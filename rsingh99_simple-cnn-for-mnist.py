# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several 

#helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

from math import floor

import seaborn as sns

import random

from scipy import ndarray

import skimage as sk

from skimage import transform

from skimage import util

import warnings

warnings.filterwarnings('ignore')

from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

from keras.layers import Input,Conv2D,Dense, Dropout, BatchNormalization, MaxPooling2D, Activation, Flatten, AvgPool2D

from keras.layers import  BatchNormalization as btn

from keras.models import Model, Sequential

from keras.applications.resnet50 import ResNet50

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

from keras.callbacks import LearningRateScheduler

from IPython.display import HTML

import base64

from scipy.ndimage.interpolation import shift

from keras.optimizers import Adam

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



%matplotlib inline
train_csv = pd.read_csv('../input/train.csv')

test_csv = pd.read_csv('../input/test.csv')
train_csv.head()
train_csv.shape,test_csv.shape
### CHECKING DISTRIBUTION OF DATASET

sns.set(color_codes=True)

sns.distplot(train_csv.iloc[:,0],label = 'LABELS',kde=False

             ,color='red',norm_hist=False,rug=False);
size_of_img = (int(np.sqrt(train_csv.shape[1])),int(np.sqrt(train_csv.shape[1])))

fig=plt.figure(figsize=(72,72))

for i in range(60):

    ax=fig.add_subplot(12,12,i+1)

    plot_image = np.array(train_csv.iloc[i,1:]).reshape(size_of_img)

    ax.imshow(plot_image,cmap='Greys')

plt.show()
def dataset_distribution(train,distribution = [60,20,20]):

    # dividing dataset in TRAIN, DEV, TEST

    # distribution is an array which tell divide percentage

    train = np.array(train)

    np.random.shuffle(train)

    perc_train = floor(distribution[0] * 0.01*train.shape[0])

    perc_dev = perc_train + floor(distribution[1] * 0.01*train.shape[0])

    perc_test = perc_dev + floor(distribution[2] * 0.01*train.shape[0])

    train_feature = train[0:perc_train,1:]

    train_label =  train[0:perc_train,0]

    

    dev_feature = train[perc_train:perc_dev,1:]

    dev_label =  train[perc_train:perc_dev,0]

    

    test_feature = train[perc_dev:perc_test,1:]

    test_label =  train[perc_dev:perc_test,0]

    

    return train_feature/255, train_label, dev_feature/255, dev_label, test_feature/255, test_label
def one_hot_encoding(label):

    ### ONE HOT ENCODING OF LABEL

    no_of_class = np.unique(label).shape[0]

    enc_labels = np.zeros((label.shape[0],no_of_class))

    for index in range(label.shape[0]):

        enc_labels[index,label[index]] = 1

    return enc_labels
def de_encoding(prediction):

    ### DE CODING LABEL

    predict = np.zeros((prediction.shape[0]))

    for index in range(prediction.shape[0]):

        predict[index] = np.argmax(prediction[index])

    return predict
def change_to_image(in_feature):

    ### CHANGING ARRAY TO IMAGE

    ### RESHAPING ARRAY FROM (,784) TO (1,28,28,1)

    feature = np.zeros(shape = (in_feature.shape[0],size_of_img[0],size_of_img[1], 1))

    for image_index in range(in_feature.shape[0]):

        feature[image_index] = (in_feature[image_index]).reshape(size_of_img[0],size_of_img[1], 1)

    return feature


def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    ### CREATING CSV FILE

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
def acc(y_test,prediction):



    ### PRINTING ACCURACY OF PREDICTION

    ### RECALL

    ### PRECISION

    ### CLASIFICATION REPORT

    ### CONFUSION MATRIX

    cm = confusion_matrix(y_test, prediction)

    recall = np.diag(cm) / np.sum(cm, axis = 1)

    precision = np.diag(cm) / np.sum(cm, axis = 0)

    

    print ('Recall:', recall)

    print ('Precision:', precision)

    print ('\n clasification report:\n', classification_report(y_test,prediction))

    print ('\n confussion matrix:\n',confusion_matrix(y_test, prediction))

    

    ax = sns.heatmap(confusion_matrix(y_test, prediction),linewidths= 0.5,cmap="YlGnBu")
train_feature, train_labels, dev_feature, dev_labels,_,__ = dataset_distribution(train_csv,[80,20,0])

train_feature.shape, train_labels.shape, dev_feature.shape, dev_labels.shape
train_image = change_to_image(train_feature)

dev_image     = change_to_image(dev_feature)
train_label = one_hot_encoding(train_labels)

dev_label = one_hot_encoding(dev_labels)
no_of_class  = 10

no_of_class,train_image.shape,dev_image.shape
perc = 85

#no_of_image_in_train = floor(train_csv.shape[0]*perc*0.01)

no_of_class = 10

print("no_of_class : ",no_of_class)

print("train_image.shape : ",train_image.shape)

print("dev_image.shape : ",dev_image.shape)

print("train_label.shape : ",train_label.shape)

print("dev_label.shape: ",dev_label.shape)

print("train_feature: ",train_feature.shape)

print("train_labels.shape : ",train_labels.shape)

gen = ImageDataGenerator(

        rotation_range=3,  

        zoom_range = 0.070,  

        width_shift_range=0.01, 

        height_shift_range=0.01)



model = Sequential()



model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.44))



model.add(Conv2D(256, kernel_size = 4, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.65))

model.add(Dense(10, activation='softmax'))





# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
# DECREASE LEARNING RATE EACH EPOCH

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# TRAIN NETWORKS

epochs = 13

X_train2, X_val2, Y_train2, Y_val2 = train_image,dev_image, train_label, dev_label

history = model.fit_generator(gen.flow(X_train2,Y_train2, batch_size=100),

    epochs = epochs, steps_per_epoch = X_train2.shape[0]//100,  

    validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=2)

name_title = ['Loss','Accuracy']

for i in range(0,2):

    ax=fig.add_subplot(8,8,i+1)

    plt.plot(history.history[list(history.history.keys())[i]], label = list(history.history.keys())[i] )

    plt.plot(history.history[list(history.history.keys())[i+2]],label = list(history.history.keys())[i+2] )

    plt.xlabel('Epochs', fontsize=18)

    plt.ylabel(name_title[i], fontsize=18)

    plt.legend()

    plt.show()
X_test = dev_image

cnn_pred = model.predict_proba(change_to_image(np.array(test_csv)/255))

results = model.predict(X_test)

results = np.argmax(results,axis = 1)

acc(dev_labels,results)
create_download_link(pd.DataFrame(cnn_pred))