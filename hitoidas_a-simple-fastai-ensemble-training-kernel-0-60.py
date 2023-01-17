# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from torch import nn

from fastai.vision import *

import torchvision



df = pd.read_csv('../input/train.csv')

path = '../input'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnext101_32x8d(pretrained=True)

iB = ImageDataBunch.from_df(path=path,

                   df=df,

                   folder = 'train_images',

                   seed = 42,

                   suffix = '.png',

                   test = 'test_images',

                   size=224,

                    bs=32,

                   ds_tfms=get_transforms(do_flip=True,

                                      max_warp=0,

                                      max_rotate=0,

                                      max_lighting=0,

                                      p_affine=0,

                                      xtra_tfms=[crop_pad()]))



#Ensembling ResNeXt101_32x8d, Densenet201, Resnet152 and VGG16(with BatchNorm)

model1 = torchvision.models.resnext101_32x8d(pretrained=True)

model1.fc = nn.Sequential(nn.BatchNorm1d(2048),

                          nn.Dropout(p=0.25),

                          nn.Linear(2048,512),

                          nn.ReLU(),

                          nn.BatchNorm1d(512),

                          nn.Dropout(p=0.5),

                          nn.Linear(512,5))

model1.to(device)

learn1 = Learner(data=iB,model=model1,model_dir='/tmp/models',metrics=[accuracy])

learn2 = cnn_learner(data=iB,base_arch=models.resnet152,model_dir='/tmp/models',metrics=[accuracy])

learn3 = cnn_learner(data=iB,base_arch=models.densenet201,model_dir='/tmp/models',metrics=[accuracy])

learn4 = cnn_learner(data=iB,base_arch=models.vgg16_bn,model_dir='/tmp/models',metrics=[accuracy])

print("Training ResNeXt101_32x8d....")

learn1.fit_one_cycle(7,slice(8e-04))

model1 = learn1.model

print("Training Resnet152....")

learn2.unfreeze()

learn2.fit_one_cycle(7,slice(3e-03))

model2 = learn2.model

print("Traning Densenet201....")

learn3.unfreeze()

learn3.fit_one_cycle(7,slice(3e-03))

model3 = learn3.model

print("Training VGG16......")

learn4.unfreeze()

learn4.fit_one_cycle(7,slice(3e-03))

model4 = learn4.model

torch.save(model1, './model1.pth')

torch.save(model2, './model2.pth')

torch.save(model3, './model3.pth')

torch.save(model4, './model4.pth')
dff = pd.read_csv("../input/test.csv")

src = (ImageList.from_df(dff, path='../input', folder='test_images', suffix='.png')

               .split_none()

               .label_empty())

model1.eval()

model2.eval()

model3.eval()

model4.eval()

iB = ImageDataBunch.create_from_ll(src,size=224,bs=32,

                                  ds_tfms=get_transforms(do_flip=True,

                                      max_warp=0,

                                      max_rotate=0,

                                      max_lighting=0,

                                      p_affine=0.2,

                                      xtra_tfms=[crop_pad()]))

predictor1 = Learner(data=iB,model=model1,model_dir='/')

preds1 = predictor1.get_preds(ds_type=DatasetType.Fix)

predictor2 = Learner(data=iB,model=model2,model_dir='/')

preds2 = predictor2.get_preds(ds_type=DatasetType.Fix)

predictor3 = Learner(data=iB,model=model3,model_dir='/')

preds3 = predictor3.get_preds(ds_type=DatasetType.Fix)

predictor4 = Learner(data=iB,model=model4,model_dir='/')

preds4 = predictor4.get_preds(ds_type=DatasetType.Fix)

labels1,labels2,labels3,labels4 = [],[],[],[]

print("Predicting from model1....")

for pr in preds1[0]:

    p = pr.tolist()

    labels1.append(np.argmax(p))

print("Predicting from model2....")

for pr in preds2[0]:

    p = pr.tolist()

    labels2.append(np.argmax(p))

print("Predicting from model3....")

for pr in preds3[0]:

    p = pr.tolist()

    labels3.append(np.argmax(p))

print("Predicting from model4....")

for pr in preds4[0]:

    p = pr.tolist()

    labels4.append(np.argmax(p))

finalPreds = []

for i in range(len(labels1)):

    predd = (labels1[i]+labels2[i]+labels3[i]+labels4[i])/4.0

    predd = np.floor(predd)

    predd = int(predd)

    finalPreds.append(predd)

ids = list(dff["id_code"])

submit = pd.DataFrame(data={'id_code':ids,'diagnosis':finalPreds})

submit.to_csv('./submission.csv',index=False)