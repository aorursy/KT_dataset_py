# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
!ls /kaggle/input/devanagari-character-set/
data_df = pd.read_csv('/kaggle/input/devanagari-character-set/data.csv')

print(data_df.shape)

print('2000 examples of each character')

print(data_df.head())
X_cols = data_df.columns.tolist()

X_cols.remove('character')



encoder = LabelEncoder()

Y = encoder.fit_transform(data_df.character.values)

X = data_df[X_cols].astype(np.uint16).values

X = X/255

X = X.reshape((-1,32,32,1))
plt.imshow(X[100,:,:,0])
from sklearn.model_selection import train_test_split

X_train,X_test, Y_train,Y_test = train_test_split(X,Y,stratify=Y,test_size=0.2)
from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Dropout,LeakyReLU,Reshape,GaussianNoise



def get_model(l_relu = 0.3,dropout=0.2):

    model = Sequential()

    model.add(GaussianNoise(0.2,input_shape=(32,32,1)))

    model.add(Conv2D(32,3,padding='same',input_shape=(32,32,1)))

    model.add(LeakyReLU(l_relu))

    model.add(MaxPool2D(2))



    model.add(Conv2D(64,3,padding='same'))

    model.add(LeakyReLU(l_relu))

    model.add(MaxPool2D(2))



    model.add(Conv2D(128,3,padding='same'))

    model.add(LeakyReLU(l_relu))

    model.add(MaxPool2D(2))



    model.add(Conv2D(128,4,padding='valid'))

    model.add(LeakyReLU(l_relu))

    model.add(Dropout(dropout))

    model.add(Conv2D(46,1,activation='softmax'))

    model.add(Reshape((-1,)))

    return model

model = get_model()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print(X_test.shape,Y_test.shape)

print(X_train.shape,Y_train.shape)
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2)



datagen.fit(X_train)



model_file = 'model.pkl'

ckp = ModelCheckpoint(model_file,save_best_only=True)

batch_size=32

epochs=100

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                    validation_data=(X_test,Y_test),

                    steps_per_epoch=len(X_train) / 32, 

                    epochs=epochs,

                    callbacks=[ckp])



model.load_weights(model_file)
pred_train_probab = model.predict(X_train)

pred_train_Y = np.argmax(pred_train_probab,axis=1)
from sklearn.metrics import confusion_matrix

conf = confusion_matrix(Y_train,pred_train_Y)



max_i = None

max_j = None

max_err = None



for i in range(conf.shape[0]):

    for j in range(conf.shape[1]):

        if i ==j:

            continue

        if conf[i,j] >5:

            labels = encoder.inverse_transform([i,j])

            if max_err is None or max_err < conf[i,j]:

                max_i = i

                max_j = j

                max_err = conf[i,j]

            print(labels[0],': perceived as :', labels[1], conf[i,j],'many times')



labels = encoder.inverse_transform([max_i,max_j])

print(f'"{labels[0]}" perceived as "{labels[1]}"', conf[max_i,max_j],'times')

_,ax =plt.subplots(ncols=2)



i_idx = int(labels[0].split('_')[1])

j_idx = int(labels[1].split('_')[1])

ax[0].imshow(X[(i_idx-1)*2000+ 1,:,:,0])

ax[0].set_title('Actual')



ax[1].imshow(X[(j_idx-1)*2000+ 1,:,:,0])

_= ax[1].set_title('Predicted as')
def get_df(Y_act,Y_pred_probab):

    Y_pred = np.argmax(Y_pred_probab,axis=1)

    Y_pred_val = np.max(Y_pred_probab,axis=1)

    correct_pred = Y_act == Y_pred

    df = pd.DataFrame(np.vstack([Y_act,Y_pred_val,correct_pred]).T,columns=['Y_act','Y_pred_val','correct'])

    return df
def plot_probablity_histograms(df,data_type):

    fig,ax = plt.subplots(ncols=2)

    fig.suptitle('Probability histogram for correct and wrong predictions for:'+data_type)

    df[df.correct ==1].Y_pred_val.hist(ax=ax[0])

    df[df.correct ==0].Y_pred_val.hist(ax=ax[1])

    ax[0].set_title('Correct Predictions')

    ax[1].set_title('Wrong Predictions')
ts_df = get_df(Y_test,model.predict(X_test))

plot_probablity_histograms(ts_df,'Test')

tr_df = get_df(Y_train,pred_train_probab)

plot_probablity_histograms(tr_df,'Train')

def scatter_plt(df,data_type):

    X = df['Y_act']

    Y = df['Y_pred_val']

    color = ['g' if c ==1 else 'r' for c in df['correct'].values]

    _,ax= plt.subplots(figsize=(20,8))

    ax.scatter(X,Y,color=color)

    ax.set_title(data_type + ':Probablities obtained for different classes. Red is incorrect, Green is correct')
scatter_plt(tr_df,'TRAIN')
scatter_plt(ts_df,'TEST')