import os

path = '../input/kaggle_csv'

print(os.listdir(path))
import os

import numpy as np

import glob

from PIL import Image, ImageEnhance, ImageOps

import torch

from torch import nn, cuda

from torch.autograd import Variable 

import torch.nn.functional as F

import torchvision.models as models

from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader





device = torch.device("cuda:0")





def test_model():



    data_transforms = transforms.Compose([

        transforms.Resize([224, 224]),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    cuda0 = torch.device('cuda:0')

    cuda1 = torch.device('cuda:1')



    model_ft = torch.load('../input/kaggle_csv/renext8d_DANN.pth', map_location='cuda:0')

    #model_4 = torch.load('../input/kaggle_csv/renext8d_aug_no_gray_100_real.pth',map_location='cuda:0')

    #model_5 = torch.load('../input/kaggle_csv/save_Fixrenext101.pth', map_location='cuda:0')

    #model_6 = torch.load('../input/kaggle_csv/renext32d_aug_gray_40.pth', map_location='cuda:0')



    path_test = '../input/kaggle_csv/test/test'

    image_list = os.listdir(path_test)

    

    predictions = []

    outputs = []

    for i, image in enumerate(image_list):

        path_image = os.path.join(path_test, image)

        image = Image.open(path_image)

        imgblob = data_transforms(image)

        imgblob.unsqueeze_(dim=0)

        imgblob = Variable(imgblob)

        imgblob = imgblob.cuda(cuda0)

        #print(imgblob.shape)

        torch.no_grad()



        output = model_ft(imgblob)

        #output4 = model_4(imgblob)

        #output5 = model_5(imgblob)

        #output6 = model_6(imgblob)





        #sum =output+output4*1.5+output5+output6*1.5

        sum = output





        prediction = int(torch.max(sum.data, 1)[1].cpu().numpy())



        prediction = prediction +1

        predictions.append(prediction)



        if i%100 ==0:

            print(i)

    return predictions

        



predictions = test_model()





import pandas as pd

submission = pd.read_csv(os.path.join(os.getcwd(), '../input/kaggle_csv/en_4_10.csv'))

submission["class"] = predictions

#submission.to_csv("../input/kaggle_csv/en_4_10.csv", index=False)