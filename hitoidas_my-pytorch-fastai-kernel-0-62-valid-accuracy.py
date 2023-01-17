# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import fastai as fa



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

import torch

from torch import optim,nn

import numpy as np



PATH = '../input'



df_train = pd.read_csv(PATH+'/train.csv', dtype={'id_code':str, 'diagnosis':int})



norm_values = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataBunch = ImageDataBunch.from_df(path=PATH,df=df_train,folder='train_images',suffix='.png',size=224,

                                   ds_tfms=get_transforms(do_flip=True,

                                                          max_warp=0,

                                                          max_rotate = 0,

                                                          max_zoom = 0,

                                                          max_lighting=0,

                                                          p_lighting=0,

                                                          p_affine=0,

                                                          xtra_tfms=[crop_pad()]), 

                                   test='test_images',

                                   bs=16,device=device).normalize(norm_values)

learn = cnn_learner(data=dataBunch,base_arch=models.densenet201,pretrained=False,model_dir='/tmp/models',

                   metrics=[accuracy])

# learn.lr_find()

# learn.recorder.plot(suggestion=True)







learn.unfreeze()

learn.fit_one_cycle(8,max_lr=(1e-05,1e-04,1e-03),wd=(1e-01,1e-01,1e-04))

learn.show_results()

net = learn.model
from PIL import Image

from torchvision import transforms

net.eval()

ids=[]

class CreateTrainDataset(Dataset):

    def __init__(self, csv_file, root_dir,transform=None):

        self.data_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data_frame)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,

                                self.data_frame.iloc[idx, 0]+'.png')

        image = Image.open(img_name)

        image = transforms.RandomHorizontalFlip()(image)

        image = transforms.Grayscale(3)(image)

        image = transforms.CenterCrop(224)(image)

        sample = {'image':transforms.ToTensor()(image)}

        return sample



testset = CreateTrainDataset(csv_file='../input/test.csv',root_dir='../input/test_images')

testloader = torch.utils.data.DataLoader(testset,batch_size=1,num_workers=4)

p=[]

for idx,image in enumerate(testloader):

    data = image["image"]

    data = data.to(device)

    output = net(data)

    preds = torch.max(torch.exp(output),1)

    p.append(int(preds.indices))



df = pd.read_csv('../input/test.csv')

ids = df["id_code"]

ddf = pd.DataFrame(data={'id_code':ids,'diagnosis':p})

ddf.to_csv('./submission.csv',sep=',',index=False)