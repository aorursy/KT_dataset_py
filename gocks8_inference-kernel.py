# saewon님 kernel을 사용했음을 알려드립니다.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
!pip install efficientnet_pytorch

!pip install pretrainedmodels
TEST_IMAGE_PATH = Path('../input/testdata/test_crop')

test_csv = pd.read_csv('../input/testdata/test.csv')
import os

import time

import random

import numpy as np

import pandas as pd

from pathlib import Path

import glob

from PIL import Image, ImageEnhance, ImageOps



from tqdm import tqdm, tqdm_notebook



import torch

from torch import nn, cuda

from torch.autograd import Variable 

import torch.nn.functional as F

import torchvision as vision

import torchvision.models as models

from torch.utils.data import Dataset, DataLoader



from sklearn.metrics import f1_score
# seed value fix

# seed 값을 고정해야 hyper parameter 바꿀 때마다 결과를 비교할 수 있습니다.

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 2019

seed_everything(SEED)
class TestDataset(Dataset):

    def __init__(self, df, mode='test', transforms=None):

        self.df = df

        self.mode = mode

        self.transform = transforms[self.mode]

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        new_idx = idx % len(self.df)

        image = Image.open(TEST_IMAGE_PATH / self.df[new_idx]).convert("RGB")

        

        if self.transform:

            image = self.transform(image)

            

        return image    
target_size = (299, 299)



data_transforms = {

    'test': vision.transforms.Compose([

        vision.transforms.Resize(target_size),

        vision.transforms.RandomRotation(5),

        vision.transforms.RandomHorizontalFlip(),

        vision.transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),

        vision.transforms.ToTensor(),

        vision.transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

}
x_test = test_csv['img_file'][:100]

num_classes = 196
weights_path = Path('../input/careff7')

weight_list = os.listdir(weights_path)
from efficientnet_pytorch import EfficientNet

tta = 5
from collections import OrderedDict

batch_size = 256

test_dataset = TestDataset(x_test, mode='test', transforms=data_transforms)

test_loader = DataLoader(test_dataset, 

                        batch_size=batch_size,

                        num_workers=2,

                        shuffle=False,

                        pin_memory=True,

                        )





total_num_models = len(weight_list)*tta 



all_prediction1 = np.zeros((len(test_dataset), num_classes))



for i, weight in enumerate(weight_list):

    print("fold {} prediction starts".format(i+1))

    

    for _ in range(tta):

        print("tta {}".format(_+1))

        

        model = EfficientNet.from_pretrained('efficientnet-b7')

        # Unfreeze model weights

        for param in model.parameters():

            param.requires_grad = True

        model._fc = nn.Sequential(

            nn.Dropout(p=0.4),

            nn.Linear(in_features=2560, out_features=196, bias=True)

        )

        

        # DataParallel 학습된 weight를 문제 없이 단일 gpu로 불러오는 code입니다.

        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686

        state_dict = torch.load(weights_path / weight)

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k[7:] # remove module.

            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        

        # multi gpu를 사용하는 code입니다. kaggle은 단일 gpu여서 의미가 없는 코드입니다.

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:

            model = nn.DataParallel(model)

        model.to(device)

        

        model.eval()

        

        prediction = np.zeros((len(test_dataset), num_classes)) # num_classes=196

        with torch.no_grad():

            for i, images in enumerate(tqdm_notebook(test_loader)):

                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

                all_prediction1 = all_prediction1 + prediction

    

all_prediction1 /= total_num_models
weights_path = Path('../input/carrb6') # careff6에 들어있는 weight인데 오타난 파일명을 바꾸는걸 깜박했네요

weight_list = os.listdir(weights_path)
batch_size = 256



test_dataset = TestDataset(x_test, mode='test', transforms=data_transforms)

test_loader = DataLoader(test_dataset, 

                        batch_size=batch_size,

                        num_workers=24,

                        shuffle=False,

                        pin_memory=True,

                        )





total_num_models = len(weight_list)*tta 



all_prediction2 = np.zeros((len(test_dataset), num_classes))



for i, weight in enumerate(weight_list):

    print("fold {} prediction starts".format(i+1))

    

    for _ in range(tta):

        print("tta {}".format(_+1))

        

        model = EfficientNet.from_pretrained('efficientnet-b6')

        # Unfreeze model weights

        for param in model.parameters():

            param.requires_grad = True

        model._fc = nn.Sequential(

                    nn.Dropout(p=0.4),

                    nn.Linear(in_features=2304, out_features=196, bias=True)

                )

        state_dict = torch.load(weights_path / weight)

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k[7:] # remove module.

            new_state_dict[name] = v

        

        model.load_state_dict(new_state_dict)

        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1: 

            model = nn.DataParallel(model)

        model.to(device)

        

        model.eval()

        

        prediction = np.zeros((len(test_dataset), num_classes)) # num_classes=196

        with torch.no_grad():

            for i, images in enumerate(tqdm_notebook(test_loader)):

                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

                all_prediction2 = all_prediction2 + prediction

    

all_prediction2 /= total_num_models
weights_path = Path('../input/careff5')

weight_list = os.listdir(weights_path)
batch_size = 256



test_dataset = TestDataset(x_test, mode='test', transforms=data_transforms)

test_loader = DataLoader(test_dataset, 

                        batch_size=batch_size,

                        num_workers=24,

                        shuffle=False,

                        pin_memory=True,

                        )





total_num_models = len(weight_list)*tta 



all_prediction3 = np.zeros((len(test_dataset), num_classes))



for i, weight in enumerate(weight_list):

    print("fold {} prediction starts".format(i+1))

    

    for _ in range(tta):

        print("tta {}".format(_+1))

        

        model = EfficientNet.from_pretrained('efficientnet-b5')

        # Unfreeze model weights

        for param in model.parameters():

            param.requires_grad = True

        model._fc = nn.Sequential(

                    nn.Dropout(p=0.4),

                    nn.Linear(in_features=2048, out_features=196, bias=True)

                )

        state_dict = torch.load(weights_path / weight)

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k[7:] # remove module.

            new_state_dict[name] = v

        

        model.load_state_dict(new_state_dict)

        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1: 

            model = nn.DataParallel(model)

        model.to(device)

        

        model.eval()

        

        prediction = np.zeros((len(test_dataset), num_classes)) # num_classes=196

        with torch.no_grad():

            for i, images in enumerate(tqdm_notebook(test_loader)):

                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

                all_prediction3 = all_prediction3 + prediction

    

all_prediction3 /= total_num_models
torch.cuda.empty_cache()
target_size = (331, 331)



data_transforms = {

    'test': vision.transforms.Compose([

        vision.transforms.Resize(target_size),

        vision.transforms.RandomRotation(5),

        vision.transforms.RandomHorizontalFlip(),

        vision.transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),

        vision.transforms.ToTensor(),

        vision.transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

}
weights_path = Path('../input/carnas')

weight_list = os.listdir(weights_path)
from pretrainedmodels import nasnetalarge
batch_size = 128



total_num_models = len(weight_list)*tta 

all_prediction4 = np.zeros((len(test_dataset), num_classes))



for i, weight in enumerate(weight_list):

    print("fold {} prediction starts".format(i+1))

    

    for _ in range(tta):

        print("tta {}".format(_+1))

        test_dataset = TestDataset(x_test, mode='test', transforms=data_transforms)

        test_loader = DataLoader(test_dataset, 

                                batch_size=batch_size,

                                num_workers=24,

                                shuffle=False,

                                pin_memory=True,

                                )



        model = nasnetalarge(num_classes=1000, pretrained='imagenet')

        model.last_linear = nn.Sequential(

                    nn.Dropout(p=0.4),

                    nn.Linear(in_features=4032, out_features=196, bias=True)

                )

        state_dict = torch.load(weights_path / weight)

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k[7:] # remove module.

            new_state_dict[name] = v

        

        model.load_state_dict(new_state_dict)

        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1: 

            model = nn.DataParallel(model)

        model.to(device)

        

        model.eval()

        

        prediction = np.zeros((len(test_dataset), num_classes)) # num_classes=196

        with torch.no_grad():

            for i, images in enumerate(tqdm_notebook(test_loader)):

                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

                all_prediction4 = all_prediction4 + prediction

    

all_prediction4 /= total_num_models
weights_path = Path('../input/carpnas')

weight_list = os.listdir(weights_path)
from pretrainedmodels import pnasnet5large
batch_size = 128



total_num_models = len(weight_list)*tta 

all_prediction5 = np.zeros((len(test_dataset), num_classes))



for i, weight in enumerate(weight_list):

    print("fold {} prediction starts".format(i+1))

    

    for _ in range(tta):

        print("tta {}".format(_+1))

        

        test_dataset = TestDataset(x_test, mode='test', transforms=data_transforms)

        test_loader = DataLoader(test_dataset, 

                                batch_size=batch_size,

                                num_workers=24,

                                shuffle=False,

                                pin_memory=True,

                                )



        

        model = pnasnet5large(num_classes=1000, pretrained='imagenet')

        model.last_linear = nn.Sequential(

                    nn.Dropout(p=0.4),

                    nn.Linear(in_features=4320, out_features=196, bias=True)

                )

        state_dict = torch.load(weights_path / weight)

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k[7:] # remove module.

            new_state_dict[name] = v

        

        model.load_state_dict(new_state_dict)

        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1: 

            model = nn.DataParallel(model)

        model.to(device)

        

        model.eval()

        

        prediction = np.zeros((len(test_dataset), num_classes)) # num_classes=196

        with torch.no_grad():

            for i, images in enumerate(tqdm_notebook(test_loader)):

                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

                all_prediction5 = all_prediction5 + prediction

    

all_prediction5 /= total_num_models
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python



def softmax(z):

    assert len(z.shape) == 2

    s = np.max(z, axis=1)

    s = s[:, np.newaxis] # necessary step to do broadcasting

    e_x = np.exp(z - s)

    div = np.sum(e_x, axis=1)

    div = div[:, np.newaxis] # dito

    return e_x / div
all_prediction1 = softmax(all_prediction1)

all_prediction2 = softmax(all_prediction2)

all_prediction3 = softmax(all_prediction3)

all_prediction4 = softmax(all_prediction4)

all_prediction5 = softmax(all_prediction5)



# soft-voting

total = (all_prediction1 + all_prediction2 + all_prediction3 + all_prediction4 + all_prediction5)



# argmax를 통하여 가장 높은 확률의 class 추출

result = np.argmax(total, axis=1)
# predict하는 시간이 너무 길어서 csv 파일로 복원하겠습니다.

# load result using local gpu

submission = pd.read_csv('../input/final-csv-for-car/soft_voting-5-tta_plus_PNAS.csv')

submission.to_csv("submission.csv", index=False)

submission.head()