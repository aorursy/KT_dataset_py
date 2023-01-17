import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import os
import random
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from tqdm import tqdm
%matplotlib inline

img_path = '../input/i2a2-bone-age-regression/images/'
train_df = pd.read_csv('../input/i2a2-bone-age-regression/train.csv')
train_df.head()
print(train_df.shape)
#i = 0
#while i < 10:
    #im = Image.open(os.path.join(img_path,train_df['fileName'][random.randint(0, 12611)]))
    #plt.figure()
    #plt.imshow(im)
    #i = i+1
#img_format = []
#img_size = []
#for i in train_df['fileName']:
    #im = Image.open(os.path.join(img_path,i))
    #img_format.append(im.format)
    #img_size.append(im.size)
#print(len(img_size))
#train_df = train_df.join(pd.DataFrame({'img_format': img_format,'img_size': img_size}, index=train_df.index))
#train_df['img_format'] = img_format
#train_df['img_size'] =  img_size
#train_df.head()
one_two_hand = []
X = []
random.seed(29)
for i in tqdm(train_df['fileName']):
    im = Image.open(os.path.join(img_path,i)).convert('RGB')
    im = im.resize((256,256))
    im = im.crop((25,25,225,225))
    im = im.resize((256,256))
    im_enh = ImageEnhance.Contrast(im)
    im = im_enh.enhance(1)
    rand_onetwo = random.randint(0,1)     
    if rand_onetwo == 1:
        im_flip_v = ImageOps.mirror(im)
        imag = Image.new('RGB',(512,512), color=(100,100,100))
        imag.paste(im,(0,128,256,384))
        imag.paste(im_flip_v,(256,128,512,384))
        imag = imag.resize((256,256))
        if random.randint(0,1) == 1:
           imag = ImageOps.flip(imag)
        #plt.imshow(imag)
        #plt.show()
        imag_arr = np.asarray(imag)
        X.append(imag_arr)
        one_two_hand.append(0)
    else:
        im = im.resize((256,256))   
        if random.randint(0,1) == 1:
           im = ImageOps.flip(im) 
        #plt.imshow(im)
        #plt.show()
        im_arr = np.asarray(im)
        X.append(im_arr)
        one_two_hand.append(1)
X_t = np.stack(X,axis=0) 
y_t = np.asarray(one_two_hand)
print(X_t.shape, y_t.shape)
X_t = np.stack(X,axis=0)
y_t = np.asarray(one_two_hand)
X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.33, random_state=42)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
input_ = Input((256, 256,3))
x = input_
x = Conv2D(10, kernel_size = (3,3), strides = (1, 1), padding = 'same', activation = 'relu')(x)
x = Flatten()(x)
x = Dropout(0.1)(x)
x = Dense(1, activation  =  'sigmoid')(x)
model = Model(input_, x)
model.summary()
batch_size = 200
num_epochs = 20
learning_rate = 1e-6
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = learning_rate), metrics = ['acc'])
history = model.fit(X_train, y_train,
          batch_size = batch_size,
          epochs = num_epochs,
          verbose = 1,
          validation_data = (X_test, y_test))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('acurácia')
plt.xlabel('época')
plt.legend(['treino', 'validação'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('época')
plt.legend(['treino', 'validação'], loc = 'upper left')
plt.show()
test_df = pd.read_csv('../input/i2a2-bone-age-regression/test.csv')
#test_df.head()
#print(test_df.shape)
X_teste = []
for i in test_df['fileName']:
    im_test = Image.open(os.path.join(img_path,i)).convert('RGB')
    im_test = im_test.resize((256,256))
    im_arr_test = np.array(im_test)
    X_teste.append(im_arr_test)
X_tteste = np.stack(X_teste,axis=0) 
resultado = model.predict(X_tteste)
#print(resultado)
false_pos = 0
for i in range(0,len(resultado)):
    if resultado[i] != 1:
        #print(resultado[i])
        #print(test_df['fileName'][i])
        im = Image.open(os.path.join(img_path,test_df['fileName'][i]))
        w, h = im.size
        if w <= 800:
            false_pos += 1
        plt.title('Image: {} '.format(img_path,test_df['fileName'][i]))
        plt.imshow(im)
        plt.show()
print("imagens classificadas com duas mãos", i)
false_neg = 0
for i in range(0,len(resultado)):
    if resultado[i] == 1:
        #print(resultado[i])
        #print(test_df['fileName'][i])
        im = Image.open(os.path.join(img_path,test_df['fileName'][i]))
        w, h = im.size
        if w > 800:
            false_neg += 1
        plt.title('Image: {} '.format(img_path,test_df['fileName'][i]))
        plt.imshow(im)
        plt.show()
print("imagens classificadas com uma mão", i)
print('false_positives', false_pos, 'false_negatives', false_neg)