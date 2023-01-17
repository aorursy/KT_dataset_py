import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np

import os

import cv2

from PIL import Image, ImageDraw 

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten,Activation

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.metrics import top_k_categorical_accuracy, categorical_accuracy

from keras.applications import MobileNet

from keras.optimizers import Adam

from keras.losses import categorical_crossentropy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape = (32,32)

training_classes = 340 # how many class we are training now

train_size = 1000

# 각 클래스당 2000개의 이미지
def top_3_accuracy(x,y): 

    t3 = top_k_categorical_accuracy(x,y, 3)

    return t3
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 

callbacks = [reduceLROnPlat, earlystop]

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(shape[0], shape[1], 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(680, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(training_classes, activation='softmax'))

model.summary()
'''base_model = MobileNet(input_shape=(shape[0], shape[1], 1), include_top=False,alpha=1, weights=None, classes=training_classes)



mm = Sequential()

mm.add(base_model)

mm.add(Flatten())

mm.add(Dropout(0.5))

mm.add(Dense(1024,activation='relu'))

mm.add(Dropout(0.5))

mm.add(Dense(training_classes,activation='softmax'))



mm.summary()'''
def draw2img(drawing, shape = shape):

    fig, ax = plt.subplots()

    drawing = ast.literal_eval(drawing)

    for x,y in drawing:

        ax.plot(x, y,'g',  marker='.') #  marker='.',

    ax.axis('off')

    fig.canvas.draw()    

    X = np.array(fig.canvas.renderer._renderer)

    plt.close(fig)

    # image resizing. Original X is of various size due to strokes variable's length

    temp = (cv2.resize(X, shape) / 255.)[::-1]

    return temp[:,:,1].astype('int8') # only green channel, as we have drawn with green, try bool
# faster conversion function

def draw_it(strokes):

    image = Image.new("P", (256,256), color=255)

    image_draw = ImageDraw.Draw(image)

    for stroke in eval(strokes):

        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i], 

                             stroke[1][i],

                             stroke[0][i+1], 

                             stroke[1][i+1]],

                            fill=0, width=5)

    image = image.resize((shape[0], shape[1]))

    return np.array(image)/255.
import time

start_time = time.time()



train = pd.DataFrame()

i = 0

#labels = dict()

for file in os.listdir('../input/train_simplified/'):

    print(f"Reading...{file}.....{i*100/340}% complete")

    temp = pd.read_csv('../input/train_simplified/' + file, nrows=train_size, 

                                    usecols = ['drawing', 'word'])

    # processing data

    temp['drawing'] = temp['drawing'].apply(draw_it)

    #global label encoding

    temp['word']    = np.int16(i)

    train = train.append(temp)

    

    i = i+1

    if i==training_classes: 

        break

    if i%10==0:

        print(f"Time elasped in reading: {(time.time() - start_time)} seconds ---") 





print(f"Total Time elasped in reading: {(time.time() - start_time)} seconds ---") 
# preparing x_train and y_train

x = np.array(train['drawing'])

y = np.array(train['word'])

# each row of x, y is a input 

# y_train to onehot encoding for making them as useful for output softmax layer

'''

array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],

       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],

'''

y =y.reshape(-1, 1)  # making it a 2d array like [[1], [1], ]

from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()

enc.fit(y)

y = enc.transform(y).toarray()



#del train 용량때매 삭제해준다.

del train



# test train split



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(

    x, y, test_size=0.1, random_state=101)



print("Taking care of dimensions--------------------")

print(f"shape of x_train: {x_train.shape}")

print(f"shape of x_val: {x_val.shape}")

print(f"shape of y_train: {y_train.shape}")

print(f"shape of y_val: {y_val.shape}")

print(f"shape of image: {x_train[1].shape}")
del x

del y

val_len = int(x_val.shape[0])

train_len = int(x_train.shape[0])

t_val = np.vstack([a for a in x_val]).reshape(val_len,shape[0],shape[0],1)

del x_val



t_train = np.vstack([a for a in x_train]).reshape(train_len,shape[0],shape[0],1)

del x_train

print(t_val.shape)

print(t_train.shape)
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy', top_3_accuracy])



hist = model.fit(x=t_train, y=y_train,

          batch_size = 680,

          epochs = 50,

          validation_data = (t_val, y_val),

          callbacks = callbacks,

          verbose = 1)
'''mm.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',

              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

hist = mm.fit(x=t_train, y=y_train,

          batch_size = 600,

          epochs = 70,

          validation_data = (t_val, y_val),

          verbose=1

          )'''
def gen_graph(history, title):

    plt.plot(history.history['top_3_accuracy'])

    plt.plot(history.history['val_top_3_accuracy'])

    plt.title('Accuracy ' + title)

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation', 'Test top 3', 'Validation top 3'], loc='upper left')

    plt.show()

gen_graph(hist,'BasicNet')
temp = pd.read_csv('../input/test_simplified.csv') 

temp['drawing'] = temp['drawing'].apply(draw_it)

ttest = np.array(temp['drawing']) 

del temp 

length = int(ttest.shape[0])

test_set = np.vstack([a for a in ttest]).reshape(length,shape[0],shape[1],1)

del ttest
ttvlist = []

testpreds = model.predict(test_set,verbose=0)

#top 3 accuracy

ttvs = np.argsort(-testpreds)[:,0:3]

ttvlist.append(ttvs)

ttvarray = np.concatenate(ttvlist)
classfiles = os.listdir('../input/train_simplified/')

numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} 
preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})

preds_df = preds_df.replace(numstonames)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']



sub = pd.read_csv('../input/sample_submission.csv', index_col=['key_id'])

sub['word'] = preds_df.words.values

sub.to_csv('subcnn_small00.csv')