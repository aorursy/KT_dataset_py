## This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import torch

import torch.nn

from sklearn.metrics import classification_report



cuda = torch.device('cuda')



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

path_to_train = "../input/fruits-360_dataset/fruits-360/Training/"

path_to_test = "../input/fruits-360_dataset/fruits-360/Test/"

classes = os.listdir(path_to_train)

value_of_img = 0

for ii in classes:

    value_of_img += len(os.listdir(path_to_train+ii))

print(value_of_img)







# read the image

classes_lib = path_to_train + classes[0]

ims = os.listdir(classes_lib)

im = plt.imread(classes_lib+"/"+ims[0])

# show the image

print(im.shape)

plt.imshow(im)

plt.show()



classes.sort()



def new_train_data(a): 

    x_train = torch.zeros(a,100,100,3)

    y_train = torch.zeros(a).type(torch.LongTensor)

    for ii in torch.arange(a):

        rndcls = torch.randint(0,len(classes),(1,))[0]

        clsdir = os.listdir(path_to_train + classes[rndcls])

        rndim = plt.imread(path_to_train + classes[rndcls] + "/" + clsdir[torch.randint(0,len(clsdir),(1,))[0]])

        y_train[ii] = rndcls

        x_train[ii] = torch.tensor(rndim)

    pooling = torch.nn.AvgPool2d(5, stride=5)

    retensor = x_train.transpose(1,3)

    pooling1 = pooling(retensor)

    repooling1 = pooling1.transpose(1,3)/255

    return repooling1, y_train

def new_test_data(a):  

    x_test = torch.zeros(a,100,100,3)

    y_test = torch.zeros(a).type(torch.LongTensor)

    for ii in torch.arange(a):

        rndcls = torch.randint(0,len(classes),(1,))[0]

        clsdir = os.listdir(path_to_test + classes[rndcls])

        rndim = plt.imread(path_to_test + classes[rndcls] + "/" + clsdir[torch.randint(0,len(clsdir),(1,))[0]])

        y_test[ii] = rndcls

        x_test[ii] = torch.tensor(rndim)

    pooling = torch.nn.AvgPool2d(5, stride=5)

    retensor = x_test.transpose(1,3)

    pooling1 = pooling(retensor)

    repooling1 = pooling1.transpose(1,3)/255

    return repooling1, y_test

def test_model(a):

    xx, yy = new_test_data(a)

    xxlen = xx#.reshape(a,-1)

    h = torch.matmul(xxlen.cuda(), best_wh)+best_bh

    h1 = torch.matmul(h.relu(), best_wh1)+best_bh1

    h2 = torch.matmul(h1.relu(), best_wh2)+best_bh2

    p = torch.matmul(h2.relu().reshape(a,-1), best_wp)+best_bp

    m = torch.nn.Softmax(dim = 1)

    output = m(p)

    pred = p.max(1)[1]

    print(classification_report(yy, pred.cpu(), labels=torch.arange(len(classes))))



def calculate_filters(xx,filters,step):

    batch_count = xx.shape[0]

    filters_count = filters.shape[0]

    filters_size = filters.shape[2]

    #xx, yy = new_train_data(batch_count)

    xx_t = xx.transpose(1,3).cuda()

    

    #filters = torch.randn(filters_count,3,filters_size,filters_size)

    box = torch.zeros(1+int((xx_t.shape[2]-filters_size)/step),1+int((xx_t.shape[3]-filters_size)/step),batch_count,filters_count).cuda()

    for i in torch.arange(0,xx_t.shape[2],step):

        for j in torch.arange(0,xx_t.shape[3],step):

            for ii in range(batch_count):

                #print((xx_t[ii,:,i:i+filters_size,j:j+filters_size]*filters).sum((1,2,3)))

                box[int(i/step),int(j/step),ii] = (xx_t[ii,:,i:i+filters_size,j:j+filters_size]*filters).sum((1,2,3))

            #box[int(i/step),int(j/step)] = torch.as_tensor([list((xx_t[:,:,i:i+filters_size,j:j+filters_size]*filters[c]).sum((1,2,3))) for c in range(filters_count)]).t()

    return box.transpose(0,2).transpose(1,3) # batch X filter X x X y

#def model(wf,wp,bp)
dx = 3

nc = len(classes)

trains = 3000

t = 1

batch_size = 300

filters_count = 50

filters_size = 5

step = 5

xx_shape = [batch_size,dx,20,20]



filters = torch.randn(filters_count,dx,filters_size,filters_size).cuda()

wp = torch.randn(filters_count*(1+int((xx_shape[2]-filters_size)/step))*(1+int((xx_shape[3]-filters_size)/step)),nc).cuda()

bp = torch.randn(nc).cuda()



best_filters = torch.randn_like(filters)

best_wp = torch.randn_like(wp)

best_bp = torch.randn_like(bp)

best_loss = 1



filters.requires_grad_(True)

wp.requires_grad_(True)

bp.requires_grad_(True)



prev = -1



for ii in range(trains):

    xx, yy = new_train_data(batch_size)

    

    p = torch.matmul(calculate_filters(xx,filters,step).relu().reshape(batch_size,-1), wp)+bp

    m = torch.nn.Softmax(dim = 1)

    output = m(p)

    loss = (1/len(yy))*torch.pow((output-1)[torch.arange(len(yy)),yy],2).sum()

    if loss < best_loss:

        best_filters = filters

        best_wp = wp

        best_bp = bp

        best_loss = loss

        print(loss)

    if round(ii*100/trains) % 10 == 0 and round(ii*100/trains) > prev: 

        print(round(ii*100/trains),"%")

        prev = round(ii*100/trains)

    loss.backward()



    with torch.no_grad():

        filters -= t * filters.grad

        wp -= t * wp.grad

        bp -= t * bp.grad

    

    filters.grad.zero_()

    wp.grad.zero_()

    bp.grad.zero_()

test_model(300)