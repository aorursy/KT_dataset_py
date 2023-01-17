# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import time 

import mxnet as mx

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from mxnet import nd, autograd, gluon, init

from mxnet.gluon import data as gdata, loss as gloss, nn

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#data processing, cited from https://www.kaggle.com/karanjakhar/facial-keypoint-detection

Train_Dir = '../input/training/training.csv'

Test_Dir = '../input/test/test.csv'

lookid_dir = '../input/IdLookupTable.csv'

train_data = pd.read_csv(Train_Dir)  

test_data = pd.read_csv(Test_Dir)

lookid_data = pd.read_csv(lookid_dir)

os.listdir('../input')

#fill null with the previous values in that row

train_data.fillna(method = 'ffill',inplace = True)

# fill the missing values in some images

imag = []

for i in range(0,7049):

    img = train_data['Image'][i].split(' ')

    img = ['0' if x == '' else x for x in img]

    imag.append(img)

#reshape the face images in [96,96]

image_list = np.array(imag,dtype = 'float')

# train data

X_train = image_list.reshape(-1,96,96)

#

training = train_data.drop('Image',axis = 1)

y_train = []

for i in range(0,7049):

    y = training.iloc[i,:]

    y_train.append(y)

#y data

Y_train = np.array(y_train,dtype = 'float')
fig = plt.figure(figsize=(10, 10))

for i in range(9):

    ax=fig.add_subplot(3,3,i+1)

    ax.imshow(X_train[i+1],cmap='gray')

plt.show()
# # keras model from  https://www.kaggle.com/karanjakhar/facial-keypoint-detection

# from keras.layers import Conv2D,Dropout,Dense,Flatten

# from keras.models import Sequential



# model = Sequential([Flatten(input_shape=(96,96)),

#                          Dense(128, activation="relu"),

#                          Dropout(0.1),

#                          Dense(64, activation="relu"),

#                          Dense(30)

#                          ])



# model.compile(optimizer='adam', 

#               loss='mean_squared_error',

#               metrics=['mae'])

# model.fit(X_train,y_train,epochs = 500,batch_size = 128,validation_split = 0.2)



# timag = []

# for i in range(0,1783):

#     timg = test_data['Image'][i].split(' ')

#     timg = ['0' if x == '' else x for x in timg] 

#     timag.append(timg)

    

# timage_list = np.array(timag,dtype = 'float')

# X_test = timage_list.reshape(-1,96,96)

# pred = model.predict(X_test)


def log_test(net,test_iter,ctx):

    test_l = 0

    for X,y in test_iter:

        X, y = X.as_in_context(ctx), y.as_in_context(ctx)

        l = loss(net(X),y)

    test_l = l.mean().asscalar()

    return test_l



def train(net, X_train, Y_train,num_epochs, trainer, batch_size, ctx):

    train_files,val_files = train_test_split(range(len(X_train)),test_size=0.1,shuffle=True)

    train_data,train_label = nd.array(X_train[train_files]),nd.array(Y_train[train_files])

    val_data,val_label = nd.array(X_train[val_files]),nd.array(Y_train[val_files])

    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_data,train_label),batch_size,shuffle=True)

    test_iter = gdata.DataLoader(gdata.ArrayDataset(val_data,val_label),batch_size)

    train_ls,test_ls=[],[]

    print('train on:',ctx)

    for epoch in range(num_epochs):

        start = time.time()

        train_sum_l =0

        for X,y in train_iter:

#             print(X.shape)

            X, y = X.expand_dims(axis=1).as_in_context(ctx), y.as_in_context(ctx)

            with autograd.record():

                l = loss(net(X),y)

            l.backward()

            trainer.step(batch_size)

        train_loss = loss(net(train_data.expand_dims(axis=1).as_in_context(ctx)),train_label.as_in_context(ctx)).mean().asscalar()

        train_ls.append(train_loss)

        print()

        if val_files:

            test_ls.append(loss(net(val_data.expand_dims(axis=1).as_in_context(ctx)),val_label.as_in_context(ctx)).mean().asscalar())

        else:

            test_ls.append(train_loss)

        print('epoch %d, train loss %.4f, test loss %.3f, time %.1f sec' % (epoch + 1,train_ls[-1], test_ls[-1], time.time() - start))

    return train_ls, test_ls
from mxnet.gluon.model_zoo import vision

#use gpu for training

ctx=mx.gpu(0)

resnet = vision.resnet34_v1(pretrained=False, ctx=mx.cpu())
fine_net = resnet.features

# fine_net.add(nn.Conv2D(64,7,strides=(2,2),padding=(3,3)),resnet18.features[1:])

fine_net.add(nn.Dense(30))

lr, num_epochs = 0.001, 500

batch_size=128

loss = gloss.L2Loss()

net = fine_net

# fine-tuning

# net[0].initialize(force_reinit=True,ctx=ctx, init=init.Xavier())

# net[2].initialize(force_reinit=True,ctx=ctx, init=init.Xavier())

# net[0].collect_params().setattr('lr_mult', 10)

# net[2].collect_params().setattr('lr_mult', 10)

net.initialize(force_reinit=True,ctx=ctx, init=init.Xavier())

net.collect_params().reset_ctx(ctx)

trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':lr})
train_ls,test_ls = train(net, X_train, Y_train,num_epochs, trainer, batch_size, ctx)
# train_ls,test_ls = train(net, X_train, Y_train,500, trainer, batch_size, ctx)

net.save_parameters('fine_tune_resnet34_20190616_adam.params')

%matplotlib inline

plt.plot(range(num_epochs),train_ls)

plt.plot(range(num_epochs),test_ls)

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['train','test'])

plt.show()
#preparing test data, cited from https://www.kaggle.com/karanjakhar/facial-keypoint-detection

timag = []

for i in range(len(test_data)):

    timg = test_data['Image'][i].split(' ')

    timg = ['0' if x == '' else x for x in timg]

    timag.append(timg)

timage_list = np.array(timag,dtype = 'float')

X_test = timage_list.reshape(-1,96,96)

plt.imshow(X_test[0],cmap = 'gray')

plt.show()

Y_test = net(nd.array(X_test).expand_dims(axis=1).as_in_context(ctx))

Y_test[0]
# pred=Y_test.asnumpy().flatten()

# rowid= pd.Series(range(1,len(pred)+1),name = 'RowId')

# loc = pd.Series(pred,name = 'Location')

# submission = pd.concat([rowid,loc],axis = 1)

# submission.to_csv('resnet18_submission_1.csv',index = False)
lookid_list = list(lookid_data['FeatureName'])

imageID = list(lookid_data['ImageId']-1)

pre_list = list(Y_test.asnumpy())

rowid = lookid_data['RowId']

rowid=list(rowid)

feature = []

for f in list(lookid_data['FeatureName']):

    feature.append(lookid_list.index(f))

preded = []

for x,y in zip(imageID,feature):

    # sometimes preded will be larger than 96

    preded.append(pre_list[x][y] if pre_list[x][y]<96 else 96) 

rowid = pd.Series(rowid,name = 'RowId')

loc = pd.Series(preded,name = 'Location')

submission = pd.concat([rowid,loc],axis = 1)

submission.to_csv('resnet18_submission_0616.csv',index = False)