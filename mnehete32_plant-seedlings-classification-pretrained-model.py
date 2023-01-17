import numpy as np

import pandas as pd

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.preprocessing import image

from keras.applications import xception

from keras.layers import Dense,Dropout

import os

from tqdm import tqdm

print(os.listdir("../input"))

from sklearn.svm import SVC

from sklearn.metrics import f1_score
Category = np.sort(os.listdir('../input/plant-seedlings-classification/train'))
data_dir = '../input/plant-seedlings-classification/'

train_dir = '../input/plant-seedlings-classification/train'

test_dir = '../input/plant-seedlings-classification/test'
train = []

for label, category in enumerate(Category):

    for file in os.listdir(os.path.join(train_dir, category)):

        imag = image.load_img(os.path.join(train_dir,category, file))

        train.append(['train/{}/{}'.format(category, file), label, category,imag.size])

        

train = pd.DataFrame(train, columns=['file', 'label', 'category','shape'])

train.head()
uniq, count = np.unique(train['label'], return_counts=True)

uniq = [Category[c] for c in uniq]

uniq_data = np.c_[uniq,count]

uniq_data = pd.DataFrame(uniq_data,columns=['Labels','Count'])

lowest_num_of_samples = min(count)

uniq_data.head(12)
i = 0 

m = 0

X_train = np.zeros((221*12,299,299,3))

labels = np.zeros((221*12),dtype=np.int)

for cat in tqdm(Category):

    c = 0

    for file in os.listdir(os.path.join(train_dir, cat)):

        imag = image.load_img(os.path.join(train_dir,cat, file),target_size=(299,299))

        imag = image.img_to_array(imag)

        imag = xception.preprocess_input(np.expand_dims(imag.copy(), axis=0))

        c += 1

        if c <= lowest_num_of_samples:

            X_train[m] = imag

            labels[m] = i

            m +=1

    i += 1
uniq, count = np.unique(labels, return_counts=True)

uniq = [Category[c] for c in uniq]

uniq_data = np.c_[uniq,count]

uniq_data = pd.DataFrame(uniq_data,columns=['Labels','Count'])

uniq_data.head(12)
#Y_train = np.eye(12)[labels]
X_train,labels = shuffle(X_train,labels,random_state = 0)
X_train, X_Val, Y_train, Y_Val = train_test_split(X_train, labels, test_size=0.1, random_state=1)
xception_model = xception.Xception(weights='../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')

X_train = xception_model.predict(X_train,batch_size=32,verbose = 1)

X_Val = xception_model.predict(X_Val,batch_size=32,verbose = 1)
model = SVC()

model.fit(X_train,Y_train)

train_pred = model.predict(X_train)

val_pred = model.predict(X_Val)

training_acc = f1_score(Y_train,train_pred,average='micro')

val_acc = f1_score(Y_Val, val_pred,average='micro')    

print('Traning score :: {}'.format(training_acc))

print('Validation Score :: {}'.format(val_acc))
Y_train = np.eye(12)[Y_train]
new_model = Sequential()

new_model.add(Dense(1024, activation='relu', input_shape=(2048,)))

new_model.add(Dense(512, activation='relu'))

new_model.add(Dropout(rate=0.3))

new_model.add(Dense(256, activation='relu'))

new_model.add(Dense(128, activation='relu'))

new_model.add(Dropout(rate=0.3))

new_model.add(Dense(64, activation='relu'))

new_model.add(Dense(12, activation='softmax'))
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
new_model.fit(X_train, Y_train, epochs = 20, batch_size = 64)
Y_pred = new_model.predict(X_Val)

Y_pred = np.argmax(Y_pred, axis = 1)
acc = f1_score(Y_Val, Y_pred,average='micro')

print('The F1score on the Validation set is {}'.format(acc))