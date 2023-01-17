# Install pytorchcv

!pip install ../input/pytorchcv/pytorchcv-0.0.55-py2.py3-none-any.whl --quiet
import os

import sys

import numpy as np

import pandas as pd

import cv2

from tqdm import tqdm

from timeit import default_timer as timer

import skimage.io



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torch.utils.data.sampler import *



if True:

    DATA_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'

    SUBMISSION_CSV_FILE = 'submission.csv'



import warnings

warnings.filterwarnings('ignore')



#### net #########################################################################



def do_predict(net, inputs):

    def logit_to_probability(logit):

        probability=[]

        for l in logit:

            p = F.softmax(l)

            probability.append(p)

        return probability

    

    num_ensemble = len(net)

    for i in range(num_ensemble):

        net[i].eval()



    probability=[0,0,0,0]

    for i in range(num_ensemble):

        logit = net[i](inputs)

        prob = logit_to_probability(logit)

        probability = [p+q for p,q in zip(probability,prob)]

    

    #----

    probability = [p/num_ensemble for p in probability]

    predict = [torch.argmax(p,-1) for p in probability]

    predict = [p.data.cpu().numpy() for p in predict]

    predict = np.array(predict).T

    predict = predict.reshape(-1)



    return predict



## load net -----------------------------------



from pytorchcv.model_provider import get_model



class Head(torch.nn.Module):

  def __init__(self, in_f, out_f, dropout):

    super(Head, self).__init__()

    

    self.f = nn.Flatten()

    self.d = nn.Dropout(0.25)

    self.dropout = dropout

    self.o = nn.Linear(in_f, out_f)



  def forward(self, x):

    x = self.f(x)

    if self.dropout:

      x = self.d(x)



    out = self.o(x)

    return out



class FCN(torch.nn.Module):

  def __init__(self, base, in_f, num_classes, dropout=True):

    super(FCN, self).__init__()

    self.base = base

    self.h1 = Head(in_f, num_classes, dropout)

  

  def forward(self, x):

    x = self.base(x)

    return self.h1(x)



def create_model():

    model = get_model("seresnext50_32x4d", pretrained=False)

    model.load_state_dict(torch.load('../input/seresnext50-32x4d-pretrained/seresnext50_32x4d-0521-b0ce2520.pth'))

    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer

    model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))

    model = FCN(model, 2048, 6, dropout=True)

    return model



#----------------------------------------------

net = []



model = create_model()

model = model.cuda()

state = torch.load('../input/panda-5fold-702/model-fld1.pth') # .

model.load_state_dict(state)

net.append(model)



model = create_model()

model = model.cuda()

state = torch.load('../input/panda-5fold-702/model-fld2.pth') # .

model.load_state_dict(state)

net.append(model)



model = create_model()

model = model.cuda()

state = torch.load('../input/panda-5fold-702/model-fld3.pth') # .

model.load_state_dict(state)

net.append(model)



model = create_model()

model = model.cuda()

state = torch.load('../input/panda-5fold-702/model-fld4.pth') # .

model.load_state_dict(state)

net.append(model)



model = create_model()

model = model.cuda()

state = torch.load('../input/panda-5fold-702/model-fld5.pth') # .

model.load_state_dict(state)

net.append(model)



#------------------------------------------



# Use this to test inference

train = pd.read_csv(f'{DATA_DIR}train.csv')[:1000]

# submission = train



submission = pd.read_csv(f'{DATA_DIR}sample_submission.csv')



WIDTH = 512

HEIGHT = 512



from torch.utils.data import Dataset, DataLoader

from PIL import Image



class ImageDataset(Dataset):

    def __init__(self, dataframe, root_dir, transform=None):

        self.df = dataframe

        self.root_dir = root_dir

        self.transform = transform



        self.paths = self.df.image_id.values



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        img_name = self.paths[idx]

        file_path = f'{self.root_dir}{img_name}.tiff'

        

        image = skimage.io.MultiImage(file_path)

        image = cv2.resize(image[-1], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = (255 - image).astype(np.float32) / 255.

        

        if self.transform is not None:

          image = self.transform(image=image)['image']

        

        image = np.rollaxis(image, -1, 0)

        

        return image

#---------------------------------------------



def run_make_submission_csv():

    target=[]

    batch_size= 4



    if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    # Use below lines to test inference

#     if True:

#         test_dataset = ImageDataset(train, f'{DATA_DIR}train_images/', None)

        test_dataset = ImageDataset(submission, f'{DATA_DIR}test_images/', None)

        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        

        t = tqdm(test_loader)

        with torch.no_grad():

            for b, image_batch in enumerate(t):

                image_batch = image_batch.cuda().float()

                predict = do_predict(net, image_batch)

                target.append(predict)

        print('')

    #---------

    else:

        target = [[1],[1],[1]]

    target = np.concatenate(target)



    submission['isup_grade'] = target

    submission['isup_grade'] = submission['isup_grade'].astype(int)

    submission.to_csv(SUBMISSION_CSV_FILE, index=False)

    print(submission.head())



if __name__ == '__main__':

    run_make_submission_csv()



    print('\nsucess!')