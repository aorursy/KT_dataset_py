# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.applications.inception_resnet_v2 import InceptionResNetV2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Model,Sequential

from keras.preprocessing.image import img_to_array,load_img

from keras.layers import Dense,GlobalAveragePooling2D,Input,Embedding,InputLayer,Activation,Flatten,Conv2D,MaxPooling2D

import os

print(os.listdir("../input"))

import glob

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm,tqdm_notebook

import tensorflow as tf

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/flickr30k_images/flickr30k_images/results.csv",delimiter='|')
df.head()
df.shape
imgbasedir="../input/flickr30k_images/flickr30k_images/flickr30k_images/"
imgdir=glob.glob(imgbasedir+"*.jpg")
imgdir[:5]
img=cv2.imread(imgbasedir+df.image_name.iloc[0])

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)

print(df[' comment'].iloc[0])

print(img.shape)
img=cv2.imread(imgbasedir+df.image_name.iloc[15])

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)

print(df[' comment'].iloc[15])

print(img.shape)
imagesize=(299,299,3)
def fun1(df1):

    m=list(df1[" comment"].values)

    return m
df1=df.groupby(by='image_name').apply(fun1)
index1=df1.index
values=df1.values
index1[1]
dict1=dict([(index1[i],values[i]) for i in range(len(values))])
model=InceptionResNetV2(include_top=False,weights='imagenet',input_shape=(299,299,3))
for layer in model.layers:

    layer.trainable=False
bottommodel=model.output

topmodel=GlobalAveragePooling2D()(bottommodel)
model1=Model(model.input,topmodel)
index2=index1[:6000]   # taking 5000 images
imgbasedir
len(index1)
xtrain=[]

for i in range(len(index2)):

    img=cv2.imread(imgbasedir+index2[i])

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=cv2.resize(img,(299,299)).astype('float16')

    xtrain.append(img)

xtrain=np.array(xtrain).astype('float16')
xtrain.dtype
xtrain=xtrain.astype(np.float16)/255
pred=model1.predict(xtrain)
xtrain=np.zeros([0,0])
pred=pred.astype('float16')
pred[0]
pred.shape
Imgbottleneck=120

wordembedsize=32

rnnsize=256

ns=1536
import random

tokendata=[random.sample(dict1[index2[i]],1)[0] for i in range(6000)]
tokendata[:5]
img=cv2.imread(imgbasedir+index2[0])

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)
import re
# Text Preprocessing

def fun(text):

    text=text.lower()

    text=re.sub(r"[^\w\d]"," ",text)

    text=re.sub(r"\s{2,}"," ",text)

    text=text.strip()

    return text



tokendata1=[fun(i) for i in tokendata]
tokendata1[:5]
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer()
tokenizer.fit_on_texts(tokendata1)
tokendata2=tokenizer.texts_to_sequences(tokendata1)
word2index=tokenizer.word_index
word2index['<pad>']=0
word2index['<start>']=len(word2index)

word2index['<end>']=len(word2index)
index2word=dict([(i,j) for j,i in word2index.items()])
len(index2word)
index2word[0]
tokendata3=[[word2index['<start>']]+tokendata2[i]+[word2index['<end>']] for i in range(len(tokendata2))]
len(tokendata3)
text=pad_sequences(tokendata2,padding='post')
imgembedsize=pred.shape[1]
logitsbottleneck=200
ns=400
Imgbottleneck=128
imgemb=tf.placeholder(shape=[None,1536],dtype=tf.float32)

sentences=tf.placeholder(shape=[None,None],dtype=tf.int32)

imgembed_bottleneck=Dense(Imgbottleneck,input_shape=(None,imgembedsize),activation='elu')

imgbottle_h=Dense(ns,input_shape=(None,Imgbottleneck),activation='elu')

embedding=Embedding(len(word2index),wordembedsize)

lstm=tf.nn.rnn_cell.LSTMCell(ns)

token_logits_bottleneck=Dense(logitsbottleneck,input_shape=(None,ns),activation='elu')

tokenlogits=Dense(len(word2index),input_shape=(None,logitsbottleneck))
c0=h0=imgbottle_h(imgembed_bottleneck(imgemb))
c0.get_shape()
word_embeds=embedding(sentences[:,:-1])

word_embeds.get_shape()
hiddenstates,_=tf.nn.dynamic_rnn(lstm,word_embeds,initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0,h0))
hiddenstates.get_shape()
flat_token_logits=tf.reshape(hiddenstates,(-1,ns))

flat_token_logits=tokenlogits(token_logits_bottleneck(flat_token_logits))

flat_ground_truth=tf.reshape(sentences[:,1:],[-1,])

flat_loss_mask=tf.not_equal(word2index['<pad>'],flat_ground_truth)

entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_ground_truth,logits=flat_token_logits)
loss=tf.reduce_mean(tf.boolean_mask(entropy,flat_loss_mask))
optimizer=tf.train.AdamOptimizer(learning_rate=0.001)

trainstep=optimizer.minimize(loss)

s=tf.InteractiveSession()
s.run(tf.global_variables_initializer())
def batch_matrix(batchcaption,padidx,maxlen=None):

    maxlen=max(map(len,batchcaption))

    matrix=np.zeros((len(batchcaption),maxlen))+padidx

    for i in range(len(batchcaption)):

        matrix[i,:len(batchcaption[i])]=batchcaption[i]

    return matrix
def generate_batch(imgemb1,indxcaption,batchsize,maxlen=None):

    m=np.random.choice(len(imgemb1),size=batchsize,replace=False)

    batch_img_embed=imgemb1[m]

    batch_captions=[tokendata3[i] for i in m]

    

    batch_padcaption=batch_matrix(batch_captions,0)

    

    return {imgemb:batch_img_embed,sentences:batch_padcaption}   
batchsize=64

n_epochs=8

n_batches_per_epoch=1000
lstm_c=tf.Variable(tf.zeros([1,400]))

lstm_h=tf.Variable(tf.zeros([1,400]))
s = tf.InteractiveSession()

tf.set_random_seed(42)
s.run(tf.global_variables_initializer())
trainlosslist=[]

for epoch in range(n_epochs):

    trainloss=0

    count=0

    for i in range(n_batches_per_epoch):

        trainloss1,_=s.run([loss,trainstep],feed_dict=generate_batch(pred,tokendata3,batchsize))

        count+=1

        trainlosslist.append(trainloss1)

        if i%50==0:

            

            print(trainloss1)
imgs=tf.placeholder('float32',[1,299,299,3])
testpred=model1(imgs)
in_h=in_c=imgbottle_h(imgembed_bottleneck(testpred))
in_h.get_shape()
in_c.get_shape()
lstm_c.assign(in_c)

lstm_h.assign(in_h)
currentword=tf.placeholder("int32",shape=[1])
wordembed=embedding(currentword)
wordembed.get_shape()
lstm.weights
lstm(wordembed,state=tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]
