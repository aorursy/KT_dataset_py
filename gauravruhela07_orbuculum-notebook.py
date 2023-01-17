import pandas as pd

import os

import numpy as np

import cv2

from tqdm import tqdm

from ast import literal_eval

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

import pickle

import matplotlib.pyplot as plt



from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Dropout

from keras.applications import vgg19

from keras.callbacks import ModelCheckpoint



import warnings



warnings.filterwarnings("ignore")
def read_img(path):

    images = {}

    for img in tqdm(os.listdir(path)):

        img_temp = cv2.imread(os.path.join(path,img))

#         print(img_temp.shape)

        img_temp = img_temp/255

        images[img] = img_temp

    return images



def read_csv(path):

    annotation = {}

    ann_csv = pd.read_csv(path)

    for i in tqdm(range(len(ann_csv))):

        annotation[ann_csv.iloc[i,0]+'.jpg'] = literal_eval(ann_csv.iloc[i,1])

    return annotation
print('Reading Images...')

path = '../input/orb/data'

images = read_img(path)



print("Reading Image Annotations...")

path = '../input/orb/annotation.csv'

annotation = read_csv(path)
X,y = [], []

for key, value in images.items():

    X.append(value)

    y.append(annotation[key])

X = np.array(X)



temp = []

for i in range(1400):

    for j in y[i]:

        if j not in temp:

            temp.append(j)

temp.sort()

labels_to_num = {}

for i,label in enumerate(temp):

    labels_to_num[label] = i

    

Y = np.zeros(shape=[len(y), len(temp)])

for i in range(len(y)):

    for j in y[i]:

        Y[i][labels_to_num[j]] = 1
X_train, X_val, y_train, y_val = train_test_split(X,Y,test_size=0.15, random_state=42)

print("X_train's shape: ",X_train.shape)

print("y_train's shape: ", y_train.shape)

print("X_val's shape: ",X_val.shape)

print("y_val's shape: ", y_val.shape)
from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
base_model = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], 3))



# freezing layers which i dont want to train.

for layer in base_model.layers:

    layer.trainable = False

    

# adding custom layers

x = base_model.output

x = Flatten()(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(512, activation='relu')(x)

predictions = Dense(33, activation='sigmoid')(x)



# creating final model

model_final = Model(input=base_model.input, output=predictions)



# compiling the model

model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1_m])



model_final.summary()
history = model_final.fit(X_train, y_train, epochs=50, validation_split=15/85, batch_size=32)

print('Saving model...')

model_final.save('model_v1.h5')
# summarizing history for loss for both training and validation set

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for f1 score for both training and validation set

plt.plot(history.history['f1_m'])

plt.plot(history.history['val_f1_m'])

plt.title('model f1 score')

plt.ylabel('f1 score')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def make_prediction(X):

    num_to_labels = {item:key for key,item in labels_to_num.items()}

    preds = model_final.predict(X.reshape(1,224,224,3))

    for i in range(len(preds)):

        preds[i] = preds[i] >= 0.5

    p = []

    for j in range(len(preds[0])):

        if preds[0][j]==1:

            p.append(num_to_labels[j])

    return p

            
plt.imshow(images['20151127_120831.jpg'])

print('Prediction: ',make_prediction(images['20151127_120831.jpg']))

print('Real: ',annotation['20151127_120831.jpg'])
plt.imshow(images['20151127_121649.jpg'])

print('Prediction: ',make_prediction(images['20151127_121649.jpg']))

print('Real: ', annotation['20151127_121649.jpg'])
plt.imshow(images['20151127_120723.jpg'])

print('Prediction: ',make_prediction(images['20151127_120723.jpg']))

print('Real: ',annotation['20151127_120723.jpg'])