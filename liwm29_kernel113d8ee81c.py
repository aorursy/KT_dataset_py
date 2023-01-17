# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import torch

import torchvision

from torchvision import transforms, models

import torch.nn as nn

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob



use_gpu = torch.cuda.is_available()

if use_gpu:

    print("Using CUDA")

else:

    print("No CUDA")
dir = r"../input/chest-xray-pneumonia/chest_xray"

print(os.listdir("../input/chest-xray-pneumonia/chest_xray"))
# define const

TEST = "test"

VAL = "val"

TRAIN = "train"
datasets = {

    x : torchvision.datasets.ImageFolder(os.path.join(dir , x) , transform=transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225]),

    ]))

    for x in [ VAL ,TRAIN, TEST]

}



batch_size = 64



dataloaders = {

    x : torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,

        shuffle=True) # num_workers=4

    for x in [TRAIN , VAL , TEST]

}



datasizes = {

    x : len(datasets[x])

    for x in [TRAIN , TEST , VAL]

}
categories = datasets[TRAIN].classes

for x in [TRAIN, VAL, TEST]:

    print("Loaded {} images under {}".format(datasizes[x], x))

    print("类别:",datasets[x].class_to_idx)
imgs , label = next(iter(dataloaders[TRAIN]))

# batch size 8

print(label.shape)

for i in range(8):

    img = np.transpose(imgs[i] , (1,2,0))

    plt.subplot(2,4,i+1)

    plt.title(categories[label[i]])

    plt.imshow(img )

    plt.axis("off")

    

plt.tight_layout()

plt.show()
vgg16 = models.vgg16_bn()



if use_gpu:

    vgg16.cuda() 
# for i in vgg16.parameters(): # 随机初始化的参数

#     print(i)
pretrained_path = "/kaggle/input/vgg16bn/vgg16_bn.pth"

vgg16.load_state_dict(torch.load(pretrained_path))
# print("参数状态:")

# for name, value in vgg16.named_parameters():

#     print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
print("features froze")

for name , param in vgg16.features.named_parameters():

    param.requires_grad = False
print("参数状态:")

for name, value in vgg16.named_parameters():

    print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
fc_list = list(vgg16.classifier.children())[:-1]

print(fc_list)
# (6): Linear(in_features=4096, out_features=1000, bias=True)

fc_list.extend([torch.nn.Linear(4096 , 2 )])

# print(fc_list)

vgg16.classifier = torch.nn.Sequential(*fc_list)

print(vgg16)
n_epoch = 10

criterion = torch.nn.CrossEntropyLoss() # 不用one-hot

optimizer = torch.optim.Adam(vgg16.classifier.parameters())
import time

n_epoch = 5



if use_gpu:

    vgg16 = vgg16.cuda()

    criterion = criterion.cuda()



for i in range(n_epoch):

    print("epoch:{} , begin at : {}".format(i , time.time()))

    running_loss = 0 

    running_acc = 0

    running_recall = 0

    running_fscore = 0

    running_precision = 0

    n_iter = 0

    for index , data in enumerate(dataloaders[TRAIN]):

        print("iteration:{}".format(index))

        optimizer.zero_grad()

        x , y = data

        if use_gpu:

            x = x.cuda()

            y = y.cuda()

        vgg16.train()

        outcome = vgg16(x)

        loss = criterion(outcome , y)

        loss.backward()

        optimizer.step()

        pred = torch.argmax(outcome , dim = 1)

        

        this_acc = accuracy_score(y.cpu() , pred.cpu())

        this_precision , this_recall , this_fscore,_ = precision_recall_fscore_support(y.cpu() , pred.cpu(), average="binary")

        this_loss = loss.item()

        

        running_acc += this_acc

        running_precision += this_precision

        running_recall += this_recall

        running_fscore += this_fscore

        running_loss += this_loss

        

        print("acc per iter:" , this_acc)

        print("precision per iter:" , this_precision)

        print("recall per iter:" , this_recall)

        print("fscore per iter:" , this_fscore)

        print("avg running_loss per iter:", this_loss / datasizes[TRAIN])

        n_iter = index+1

    

    print("avg running loss:",running_loss / datasizes[TRAIN])

    print("avg running acc:",running_acc / n_iter)

    print("avg running precision:",running_precision / n_iter)

    print("avg running recall:",running_recall / n_iter)

    print("avg running fscore:",running_fscore / n_iter)

    

    #val

    val_loss = 0

    val_acc = 0

    val_precision = 0

    val_recall = 0

    val_fscore = 0

    n_iter_val = 0

    print("Start val --- begin at:",time.time())

    for index,data in enumerate(dataloaders[VAL]):

        with torch.no_grad():

            x , y = data

            if use_gpu:

                x = x.cuda()

                y = y.cuda()

            vgg16.eval()

            outcome = vgg16(x)

            pred = torch.argmax(outcome , dim = 1)

            

            this_acc = accuracy_score(y.cpu() , pred.cpu() )

            this_precision , this_recall , this_fscore,_ = precision_recall_fscore_support(y.cpu() , pred.cpu()  , average="binary")

            this_loss = loss.item()

            

            val_acc += this_acc

            val_precision += this_precision

            val_recall += this_recall

            val_fscore += this_fscore

            val_loss += this_loss

        n_iter_val = index + 1

    print("epoch:{} , val_acc:{}".format( i , val_acc / n_iter_val))

    print("epoch:{} , val_loss:{}".format( i ,val_loss / datasizes[VAL]))

    print("epoch:{} , val_precision:{}".format( i , val_precision / n_iter_val))

    print("epoch:{} , val_recall:{}".format( i , val_recall / n_iter_val))

    print("epoch:{} , val_fscore:{}".format( i , val_fscore / n_iter_val))

    print("epoch:{} , end at : {}".format(i , time.time()))



            
torch.save(vgg16.state_dict , "/kaggle/working/myTrain.pth")
# TEST

test_acc = 0

test_loss = 0

test_precision = 0

test_recall = 0

test_fscore = 0

n_iter_test = 0

print("Start TEST --- begin at:",time.time())

for index,data in enumerate(dataloaders[TEST]):

    with torch.no_grad():

        x , y = data

        if use_gpu:

            x = x.cuda()

            y = y.cuda()

        vgg16.eval()

        outcome = vgg16(x)

        pred = torch.argmax(outcome , dim = 1) # 因为没有keepdim 此时pred是一维的

        

        this_acc = accuracy_score(y.cpu() , pred.cpu())

        this_precision , this_recall , this_fscore,_ = precision_recall_fscore_support(y.cpu() , pred.cpu() , average="binary")

        this_loss = loss.item()

        

        test_acc += this_acc

        test_precision += this_precision

        test_recall += this_recall

        test_fscore += this_fscore

        test_loss += this_loss

    n_iter_test = index+1

print("TEST : acc:" , test_acc / n_iter_test)

print("TEST : loss:" , test_loss / datasizes[TEST])

print("TEST : precision:" , test_precision / n_iter_test)

print("TEST : recall:" , test_recall / n_iter_test)

print("TEST : fscore:" , test_fscore / n_iter_test)
imgs , label = next(iter(dataloaders[TEST]))

imgs = imgs.cuda()

vgg16 = vgg16.cuda()

vgg16.eval()

outcome = vgg16(imgs)

pred = torch.argmax(outcome , dim = 1)

plt.figure(figsize=(15,8))

for i in range(8):

    img = np.transpose(imgs.cpu().numpy()[i] , (1,2,0))

    plt.subplot(2,4,i+1)

    plt.title(categories[label[i]]+"->pred as->"+categories[pred[i]])

    plt.imshow(img )

    plt.axis("off")

    print(i,"ok")

    

plt.tight_layout()

plt.show()
# PNEUMONIA样本太多了

print(categories.index("PNEUMONIA") , "PNEUMONIA ->太多了")

list1 = [x for x in datasets[TRAIN].imgs if x[1] == 1]

list2 = [x for x in datasets[TRAIN].imgs if x[1] == 0]

print("Train data:", categories[1],len(list1))

print("Train data:", categories[0],len(list2))