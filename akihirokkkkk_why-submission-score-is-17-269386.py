import pandas as pd

import numpy as np

pic_data = pd.read_pickle("/kaggle/input/dataset-train/make_pic.pkl")

pic_data["pixcel"] = pic_data["pixcel"].apply(lambda x:np.array(x)/255 if type(x)=="list" else x)

train = pd.read_json("../input/deepfake-detection-challenge/train_sample_videos/metadata.json").T

train["label"] = train["label"].apply(lambda x:0 if x=="REAL" else 1)

train["video_file"] = train.index

train = pd.merge(train,pic_data,on="video_file")
train.head()
from keras.preprocessing import image



bb = []

for u in range(len(train[train["label"]==1]["pixcel"])):

    datagen = image.ImageDataGenerator(rotation_range=20)

    x = train[train["label"]==1]["pixcel"].values[u].reshape(1,128,128,3)

    gen = datagen.flow(x, batch_size=1)

    for i in range(3):

        batches = next(gen)

        gen_img = batches[0].astype(np.uint8)

        bb.append(gen_img)

        

train = train.drop(["original","split","video_file"],axis=1)

#a = pd.DataFrame()

#a["pixcel"] = bb

#a["label"] = 1
#train  = train.append(a)

train_data_box = []

for i in train["pixcel"].values:

    train_data_box = np.append(train_data_box,i)

train_data_box = train_data_box.reshape(-1,128,128,3)



label = train["label"]

label = pd.DataFrame(label.values)[0]



pic_data = pd.read_pickle("/kaggle/input/test-data/test_make_pic.pkl").rename(columns={"video_file":"filename"})

sample = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

test_data = pd.merge(sample,pic_data,on="filename")

test_data_box = []

for i in test_data["pixcel"].values:

    test_data_box = np.append(test_data_box,i)

test_data_box = test_data_box.reshape(-1,128,128,3)/255
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from keras.models import Sequential

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPool2D

from keras.optimizers import Adam,RMSprop

from keras.layers.core import Dense,Activation, Dropout, Flatten

import keras

from keras.layers import BatchNormalization

from IPython.display import display, HTML, clear_output



split_num = 2



kf = KFold(n_splits=split_num, shuffle=True)

oof_pred = np.zeros((train_data_box.shape[0], ))

y_pred = np.zeros((label.shape[0], ))

y_pred = y_pred.astype("float")



pp = 0



for train_index, eval_index in kf.split(train_data_box):

    x_train, x_test = train_data_box[train_index], train_data_box[eval_index]

    y_train, y_test = label[train_index], label[eval_index]



    x_train = x_train/255

    x_test = x_test/255



    # モデルの定義

    model = Sequential()



    model.add(Conv2D(64,3,input_shape=(128,128,3)))

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    #model.add(Dropout(0.3))



    model.add(Conv2D(64,3,input_shape=(128,128,3)))

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    #model.add(Dropout(0.3))



    model.add(Conv2D(64,3,input_shape=(128,128,3)))

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    #model.add(Dropout(0.3))



    model.add(Conv2D(64,3,input_shape=(128,128,3)))

    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())





    model.add(Flatten())

    model.add(Dense(32))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.5))

    #model.add(BatchNormalization())



    model.add(Dense(32))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.5))



    model.add(Dense(32))

    model.add(Activation('relu'))

    model.add(BatchNormalization())



    model.add(Dense(32))

    model.add(Activation('relu'))

    model.add(BatchNormalization())



    model.add(Dense(1, activation='sigmoid'))



    adam = RMSprop(lr=0.2e-4)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=["accuracy"])

    #model.summary()

    

    #es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')



    ep = 200

    history = model.fit(x_train,y_train, batch_size=32, nb_epoch=ep, verbose=1,validation_data=(x_test,y_test))#,callbacks=[es_cb])



    clear_output()



    oof_pred[eval_index] = y_test.values.reshape(oof_pred[eval_index].shape)

    y_pred[eval_index] = model.predict(x_test).reshape(y_pred[eval_index].shape)

    

    pp = model.predict(test_data_box) + pp

    

pp = pp/split_num

print("finished")
# 可視化

plt.plot(range(1, ep+1), history.history['loss'], label="loss")

plt.plot(range(1, ep+1), history.history['val_loss'], label="val_loss")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
test_data.head()


import glob

import os

import tqdm

filenames = glob.glob(os.path.join('/kaggle/input', 'deepfake-detection-challenge/test_videos/*.mp4'))

sub = pd.read_csv(os.path.join('/kaggle/input', 'deepfake-detection-challenge/sample_submission.csv'))

sub["label"] = 0.5

file = []

for filename in tqdm.tqdm(filenames):

    file.append(filename.split('/')[-1])
sub["filename"] = sorted(file, key=lambda s: s if s[0].isalnum() else s[1:])

sub["label"] = pp

sub.to_csv('submission.csv', index=False)
sub.head()