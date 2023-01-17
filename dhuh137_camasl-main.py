import numpy as np

import pandas as pd

import os

import tensorflow as tf

import cv2

import PIL

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from keras.preprocessing import sequence

import matplotlib.pyplot as plt

from tqdm import tqdm

#from keras.initializers import lecun_uniform
from fastai.vision import *

path = Path("../input/data/data")

dim = 224

vocab = ['alarm' , 'lock', 'movie', 'rain', 'weather']
X = np.array([])

X_tensor = np.array([])

for cat in tqdm(vocab):

    path_tmp = path/cat

    imgs = path_tmp.ls()

    cat_imgs = []

    cat_imgs_ten = []

    for im in imgs:

        seq = []

        seq_ten = []

        for i in im.ls():

            img = np.array(PIL.Image.open(i).resize((dim,dim)))/255

            seq.append(img)

            img_ten = tf.image.convert_image_dtype(img, dtype=tf.float32)

            seq_ten.append(img_ten)

        cat_imgs.append(np.array(seq))

        cat_imgs_ten.append(np.array(seq_ten, dtype=tf.Tensor))

    cat_imgs = np.array(cat_imgs)

    cat_imgs_ten = np.array(cat_imgs_ten)

    X = np.append(X,cat_imgs)

    X_tensor = np.append(X_tensor, cat_imgs_ten)



print(X_tensor.shape, X.shape)
col_time = False

if col_time:

    for idx,seq in enumerate(X):

        X[idx]=np.array(

                [list(np.arange(len(seq))),list(seq)]

        )
Y=[]

for u in range(50):

    label = vocab[u//10]

    for _ in range(len(X[u])):

        Y.append(label)

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)

dummy_y.shape
Y = []

vocab_enc = pd.get_dummies(vocab)

for u in range(50):

    label = np.array(vocab_enc.iloc[u//10])

    seq = []

    for _ in range(len(X[u])):

        seq.append(label)

    Y.append(np.array(seq))



#Y = np.argmax(np.array(Y),axis=1)

Y = np.array(Y)

Y[0].shape
import time

from IPython.display import clear_output

seq=random.randint(0,49)

for idx in range(len(X[seq])):

    plt.imshow(X[seq][idx].reshape(224,224,3))

    plt.title(vocab[np.argmax(Y[seq])])

    plt.show()

    time.sleep(0.1)

    plt.close()

    clear_output(wait=True)
print(

    '''Data contain 1644 (224*224*3 images)\n

    Information about X:\n

    \tShape: {}\n

    \tShape of first sequence: {}\n

    \tShape of image: {}\n

    \tData Type: {}\n

    Information about X_tensor:\n

    \tShape: {}\n

    \tShape of first sequence: {}\n

    \tData type: {}\n

    Information about Y:\n

    \tShape: {}\n

    \tShape of first sequence: {}\n

    \tOne hot encoded information: {}\n

    Information about dummy_Y:\n

    \tShape: {}\n

    '''.format(X.shape,X[0].shape,X[0][0].shape, type(X[0][0]),

               X_tensor.shape,X_tensor[0].shape, type(X_tensor[0][0]),

               Y.shape, Y[0].shape, Y[0][0],

               dummy_y.shape))
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

from __future__ import unicode_literals



import torch

import torchvision

from torch.autograd import Variable

import torchvision.transforms as transforms

import torch.utils.data as D

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.nn.utils.rnn as R



torch.__version__
X_torch = []

for imgs in X:

    seq = []

    for img in imgs:

        seq.append(torch.Tensor(img))

    X_torch.append(np.array(seq,dtype=torch.Tensor))

X_torch = np.array(X_torch, dtype = torch.Tensor)

X_torch.shape
path_working = Path("../working")

path_working.ls()
np.save(path_working/"X_torch.npy", X_torch)

np.save(path_working/"Y.npy", Y)
class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)
class CLSTM_Cell(nn.Module):

    def __init__(self, input_size,

                 hidden_dim, kernel_size,

                 output_size, bias=True, use_cuda=False):

        super(CLSTM_Cell, self).__init__()

        

        self.input_dim, self.height, self.width = input_size

        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size

        self.bias = bias

        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.use_cuda = use_cuda

        self.conv = self.localize(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,

                              out_channels= 4 * self.hidden_dim,

                              kernel_size=self.kernel_size,

                              padding=self.padding,

                              bias=self.bias))

        self.pool = torch.nn.Conv2d(in_channels = 2*self.hidden_dim, out_channels=1, kernel_size=1)                                                                                                                                                                             

        self.flatten = Flatten()

        self.fc = nn.Linear(224*224, output_size)

        

    def forward(self, x, chs):

        x = torch.cat([x, chs[1]], dim=1)

        x = self.conv(x)

        cc_i, cc_f, cc_o, cc_g = torch.split(x, self.hidden_dim, dim=1) 

        i = torch.sigmoid(cc_i)

        f = torch.sigmoid(cc_f)

        o = torch.sigmoid(cc_o)

        g = torch.tanh(cc_g)

        

        c_next = f * chs[0] + i * g

        h_next = o * torch.tanh(c_next)

        

        return c_next, h_next



    def predict(self, x):

        return F.relu(self.fc(self.flatten(x)))

    

    def combine_chs(self, c, h):

        ch = torch.cat([c, h], dim=1)

        ch = F.relu(self.pool(ch))

        return ch



    def init_hidden(self, batch_size):

        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),

                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))

    

    def localize(self, x):

        if self.use_cuda:

            return x.cuda()

        else:

            return x

test = CLSTM_Cell((3, 224,224), 1, [3,3], use_cuda=False, output_size=5)

for param in test.parameters():

    print(type(param.data), param.size())
ch = test.init_hidden(1)

c_next, h_next = test.forward(X_torch[0][0].transpose(0,2).reshape(1,3,224,224),ch)
prediction = test.predict(test.combine_chs(c_next, h_next))

prediction
!pip install torchviz

from torchviz import make_dot
g = make_dot(prediction)

g
criterion = nn.CrossEntropyLoss()

optimizer=optim.Adam(test.parameters(),lr=0.00001)

optimizer.zero_grad()
loss = criterion(prediction, 

                 torch.argmax(torch.tensor(Y[0][0], dtype=torch.long).reshape(1,5)).reshape(1))

loss.backward(retain_graph=True)
optimizer.step()
loss.item()
(c_next,h_next) = test.init_hidden(1)

criterion = nn.MSELoss()

optimizer=optim.Adam(test.parameters(),lr=0.01)

optimizer.zero_grad()

num_epochs = 20

losses = []

for epoch in range(num_epochs):

    for img in tqdm(X_torch[0]):

        c_next, h_next = test.forward(img.transpose(0,2).reshape(1,3,224,224),

                                      (c_next,h_next))

    prediction = test.predict(test.combine_chs(c_next, h_next))

    loss = criterion(prediction, torch.tensor(Y[0][0], dtype=torch.float).reshape(1,5))

    loss.backward(retain_graph = True)

    optimizer.step()

    losses.append(loss.item())
plt.plot(np.arange(num_epochs), losses)
class camASL(nn.Module):

    def __init__(self, input_size,

                 hidden_dim, kernel_size,

                 num_layers, batch_first,

                 bias):

        super(camASL, self).__init__()

        

        self.clstm1 = CLSTM_Cell(input_size,hidden_dim,kernel_size)



        



    def forward(self, x):

        return x
import keras as K

from keras.layers import LSTM, Flatten, Embedding, Dense, TimeDistributed, DepthwiseConv2D, Dropout, Conv2D

from keras.callbacks import ModelCheckpoint
model = K.models.Sequential(

    [

        Conv2D(16,kernel_size = 3, strides=1, activation = 'relu', padding='valid', 

                            kernel_initializer='he_uniform', bias_initializer='zeros',

                        input_shape=(dim,dim,3)),

        Flatten(),

        Embedding(dim*dim, 500),

        LSTM(500, kernel_initializer='glorot_uniform',

             recurrent_initializer='orthogonal', return_sequences=True),

        Dropout(rate = 1-0.2),

        TimeDistributed(Dense(5, activation="softmax"))

    ]

    )
model = K.models.Sequential(

    [

        Conv2D(16,kernel_size = 3, strides=1, activation = 'relu', padding='valid', 

                            kernel_initializer='he_uniform', bias_initializer='zeros',

                        input_shape=(dim,dim,3)),

        Flatten(),

        Embedding(dim*dim, 500),

        LSTM(500, kernel_initializer='glorot_uniform',

             recurrent_initializer='orthogonal', return_sequences=False),

        Dropout(rate = 1-0.2),

        Dense(5, activation="softmax")

    ]

    )
model = K.models.Sequential(

    [

        LSTM(500, kernel_initializer='glorot_uniform',

             recurrent_initializer='orthogonal', return_sequences=False,input_shape=(2,43)),

        Dropout(rate = 1-0.2),

        Dense(1)

    ]

    )
model = K.models.Sequential(

    [

        Conv2D(16,kernel_size = 3, strides=1, activation = 'relu', padding='valid', 

                            kernel_initializer='he_uniform', bias_initializer='zeros',

                        input_shape=(dim,dim,3)),

        Flatten(),

        Dense(1, activation="softmax")

    ]

    )
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.compile(optimizer='adam', 

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
#model.train_on_batch(X[0],Y_test)