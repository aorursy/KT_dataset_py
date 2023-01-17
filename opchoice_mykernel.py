import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import tensorflow as tf
import keras
import librosa
import librosa.display
import glob
from tqdm import tqdm_notebook as tqdm
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, LeakyReLU, Conv2D
from keras.layers import Dropout
from keras.utils import np_utils
from keras.utils.vis_utils import model_to_dot
%matplotlib inline 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
print(os.listdir("../input"))
train_dir='../input/comp/train/'
test_dir='../input/comp/test/'
submit_dir = "../input/comp/submission"
train_list=os.listdir(train_dir)
test_list=os.listdir(test_dir)
print(train_list)
print(os.listdir(submit_dir))
def mfcc_extract(filename):
  y,sr  = librosa.load(filename,sr = 44100)
# n_mfcc: number of MFCCs to return
# n_fft: length of the FFT window
# hop_length: number of samples between successive frames.
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,n_fft=int(0.02*sr),hop_length=int(0.01*sr))
  #delta=librosa.feature.delta(mfcc)
  #delta2=librosa.feature.delta(mfcc,order=2)
  #con_mfcc=np.concatenate((mfcc,delta,delta2),axis=0)
  return mfcc
def parse_audio_files(parent_dir, sub_dirs):
  labels = []
  features = []
  for label,sub_dir in enumerate(tqdm(sub_dirs)):
    for fn in glob.glob(os.path.join(parent_dir,sub_dir,"*.wav")):
      #print mfcc_extract(fn).shape
      features.append(mfcc_extract(fn))
      labels.append(label)
  return features,labels
train_features, train_labels = parse_audio_files(train_dir,train_list)
test_features, test_labels = parse_audio_files(test_dir,test_list)
submit_features=[]
submit_dir = "../input/comp/submission/"
for name in os.listdir(submit_dir):
      submit_features.append(mfcc_extract(submit_dir+name))
print( len(train_features), train_features[0].shape)
print( len(test_features), test_features[0].shape)
print( len(submit_features), submit_features[0].shape)
fig = plt.figure(figsize=(28,24))
for i,mfcc in enumerate(tqdm(train_features)):
  if i%40 < 3 : 
    #print i
    sub = plt.subplot(10,3,i%40+3*(i/40)+1)
    librosa.display.specshow(mfcc,vmin=-700,vmax=300)
    if ((i%40+3*(i/40)+1)%3==0) : 
      plt.colorbar()
    sub.set_title(train_list[train_labels[i]])
plt.show()  
train_features=np.asarray(train_features)
test_features=np.asarray(test_features)
submit_features=np.asarray(submit_features)
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
print(train_labels[0])
train_features= train_features.reshape(len(train_features),13,501,1)
test_features= test_features.reshape(len(test_features),13,501,1)
submit_features= submit_features.reshape(len(submit_features),13,501,1)
print(train_features.shape)
print(test_features.shape)
print(submit_features.shape)
train_features, train_labels = shuffle(train_features, train_labels, random_state=0)
test_features, test_labels = shuffle(test_features, test_labels, random_state=0)
model = Sequential()
model.add(Conv2D(512, kernel_size=2, input_shape=(13, 501, 1),))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(0.2))
model.add(Conv2D(128, kernel_size=2,strides=2))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(0.2)) 
model.add(Conv2D(8, kernel_size=2,strides=2))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(0.2)) 
model.add(Flatten())
model.add(Dense(units=128))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2)) 
model.add(Dense(units=32))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2)) 
model.add(Dense(units=10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
hist=model.fit(train_features, train_labels, batch_size=50, epochs=100, validation_data=(test_features, test_labels))
plt.style.use('ggplot')

fig,loss_ax =plt.subplots()
acc_ax = loss_ax.twinx()
acc_ax.grid(None)
loss_ax.plot(hist.history['loss'],'b',label='train_loss')
#loss_ax.grid(None)
acc_ax.plot(hist.history['acc'],'r',label='train_acc')
fig.set_dpi(100)
fig.suptitle('Train acc&loss')
fig.legend()
plt.show()
submit_labels=model.predict_classes(submit_features)
print(submit_labels.shape,submit_labels[0])
submit_label=model.predict(submit_features)
print(submit_label.shape,submit_label[0])
f = open('submission.csv','w')     #temp.csv 파일을 write모드로 open
f.write("filename,class\n")  #csv파일의 첫번째 row에는 column명이 들어갑니다.  f.write는 글자수를 return하지만 별로 중요한 내용은 아닙니다. 
for fn, label_num in zip(os.listdir(submit_dir),submit_labels):
    f.write(fn+","+train_list[label_num]+'\n')     #python에서 +는 문자열을 합쳐주는데도 이용됩니다. "filename,class_name\n"의 형태로 파일에 출력됩니다. 
f.close()    
pd.read_csv('submission.csv')