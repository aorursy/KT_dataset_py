# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

np.random.seed(3)

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import tensorflow as tf 

from tensorflow import keras 

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 

import seaborn as sns 

plt.style.use('fivethirtyeight')

import pickle 

import os 

import numpy as np

import cv2 

%matplotlib inline
from keras import layers

from keras.models import Sequential

from keras.optimizers import Adam, RMSprop

from keras.applications import VGG16

from keras.utils.np_utils import to_categorical

from keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator,image,img_to_array,load_img
def build_model(backbone, lr=1e-4):

    model = Sequential()

    model.add(backbone)

    model.add(Flatten())

    model.add(Dense(4096, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(4096, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(1000, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(1,activation="sigmoid"))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=lr),

        metrics=['acc']

    )

    return model



vgg16 = VGG16(

    weights='imagenet',

    include_top=False,

    input_shape=(150,150,3)

)



model = build_model(vgg16 ,lr = 1e-4)

model.summary()
labels = ['PNEUMONIA', 'NORMAL']

img_size = 150

def get_training_data(data_dir):

    data = [] 

    for label in labels: 

        path = os.path.join(data_dir, label)

        class_num = labels.index(label)

        for img in os.listdir(path):

            try:

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

                resized_arr = cv2.resize(img_arr, (img_size, img_size))

                data.append([resized_arr, class_num])

            except Exception as e:

                print(e)

    return np.array(data)

train = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')

test = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')

val = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')
pnenumonia = 0 

normal = 0 



for i, j in train:

    if j == 0:

        pnenumonia+=1

    else:

        normal+=1

        

print('Pneumonia:', pnenumonia)

print('Normal:', normal)

print('Pneumonia - Normal:', pnenumonia-normal)
X = []

y = []



for feature, label in train:

    X.append(feature)

    y.append(label)



for feature, label in test:

    X.append(feature)

    y.append(label)

    

for feature, label in val:

    X.append(feature)

    y.append(label)





# resize data for deep learning 

X = np.array(X).reshape(-1, img_size, img_size, 3)

y = np.array(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)



X_train = X_train/255.0

X_test = X_test/255.0
#early stoping

from keras.callbacks import EarlyStopping,ModelCheckpoint



callbacks = [ EarlyStopping(monitor='val_loss',patience=4),

             ModelCheckpoint(filepath='best_model_so_far.h5',monitor='val_loss',save_best_only=True)

    

]
history = model.fit(

    X_train,

    y_train,

    batch_size=32,

    epochs=15,

    validation_split=0.1,

    callbacks=callbacks



)
model.save_weights("vgg_with_random_state_20.h5")
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(['Train','Test'],loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.ylabel("Loss")

plt.xlabel("Epoch")

plt.legend(['Train','Test'],loc='upper left')

plt.show()
pred = model.predict(X_train)
precisions,recalls,thresholds = precision_recall_curve(y_train,pred)

fpr,tpr,thresholds2 = roc_curve(y_train,pred)
def plot_precision_recall(precisions,recalls,thresholds):

    plt.plot(thresholds,precisions[:-1],'b--')

    plt.plot(thresholds,recalls[:-1],'g-')

    plt.title("Precision Vs Recall")

    plt.xlabel('Thresholds')

    plt.legend(['Precision','Recall'],loc='best')

    plt.show()



def plot_rcc(fpr,tpr):

    plt.plot(fpr,tpr)

    plt.plot([0,1],[0,1],'k--')

    plt.title('FPR Vs TPR')

#     FPR = False Positive Rate and TPR = True Positive Rate

    plt.xlabel("FPR")

    plt.ylabel("TPR(RECALL)")

    plt.legend(['FPR','TPR'],loc='best')

    plt.show()



plot_precision_recall(precisions,recalls,thresholds)

plot_rcc(fpr,tpr)
predictions = model.predict(X_test)
binary_predictions = []

threshold = thresholds[np.argmax(precisions >= 0.90)]

for i in predictions:

    if i>= threshold:

        binary_predictions.append(1)

    else:

        binary_predictions.append(0)
print('Accuracy on testing set : ',accuracy_score(binary_predictions,y_test))

print('Precision on testing set : ',precision_score(binary_predictions,y_test))

print('Recall on testing set : ',recall_score(binary_predictions,y_test))
matrix = confusion_matrix(binary_predictions,y_test)

plt.figure(figsize=(16,10))

ax = plt.subplot()

sns.heatmap(matrix,annot=True,ax=ax)



ax.set_xlabel('Predicted Labels',size=20)

ax.set_ylabel('True Labels',size=20)

ax.set_title('Confusion Matrix',size=20)



ax.xaxis.set_ticklabels(labels)

ax.yaxis.set_ticklabels(labels)
plt.figure(figsize =(40,40))

for i in range(80):

    plt.subplot(8,10,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.title(labels[y_test[i]])

    plt.imshow(X_test.reshape(-1,img_size,img_size,3)[i])

    

    if(binary_predictions[i]==y_test[i]):

        plt.xlabel(labels[binary_predictions[i]],color='blue')

    else:

        plt.xlabel(labels[binary_predictions[i]],color='red')

plt.show()
model.save("First_Try_Vgg16_Model_With_Random_State_20.h5")