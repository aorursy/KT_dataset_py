# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from random import randint 
from sklearn.preprocessing import MinMaxScaler
train_labels=[]
train_samples=[]
for i in range(900):
    random_number=randint(13,64)
    train_samples.append(random_number)
    train_labels.append(0)
    
    random_number=randint(65,100)
    train_samples.append(random_number)
    train_labels.append(1)
    
    
for i in range(35):
    random_number=randint(13,64)
    train_samples.append(random_number)
    train_labels.append(1)
    
    random_number=randint(65,100)
    train_samples.append(random_number)
    train_labels.append(0)
    
    
    
for i in range(100):
    random_number=randint(13,64)
    train_samples.append(random_number)
    train_labels.append(0)
    
    random_number=randint(65,100)
    train_samples.append(random_number)
    train_labels.append(1)
    
    
for i in range(15):
    random_number=randint(13,64)
    train_samples.append(random_number)
    train_labels.append(1)
    
    random_number=randint(65,100)
    train_samples.append(random_number)
    train_labels.append(0)
    
    
    
train_labels=np.array(train_labels)
train_samples=np.array(train_samples)
scaler=MinMaxScaler(feature_range=(0,1))
scaled_train_samples=scaler.fit_transform((train_samples).reshape(-1,1))
for i in range(10):
    print(scaled_train_samples[i])
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
model=Sequential([
    Dense(16,input_shape=(1,),activation='relu'),
    Dense(32,activation='relu'),
    Dense(2,activation='softmax')
    
    
])
model.summary()
model.compile(Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(scaled_train_samples,train_labels,validation_split=0.1,batch_size=10,epochs=20,verbose=2,shuffle=True)
test_samples=[]
test_labels=[]
for i in range(100):
    random_number=randint(13,64)
    test_samples.append(random_number)
    test_labels.append(0)
    
    random_number=randint(65,100)
    test_samples.append(random_number)
    test_labels.append(1)
    
    
for i in range(10):
    random_number=randint(13,64)
    test_samples.append(random_number)
    test_labels.append(1)
    
    random_number=randint(65,100)
    test_samples.append(random_number)
    test_labels.append(0)
test_labels=np.array(test_labels)
test_samples=np.array(test_samples)
scaler=MinMaxScaler(feature_range=(0,1))
scaled_test_samples=scaler.fit_transform((test_samples).reshape(-1,1))
predict=model.predict(scaled_test_samples,batch_size=10,verbose=0)
for i in predict:
    print(i)
predict_class=model.predict_classes(scaled_test_samples,batch_size=10,verbose=0)
for i in predict_class:
    print(i)
%matplotlib inline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
cm=confusion_matrix(test_labels,predict_class)
cm
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
classes=['noside effect','side effect']
plot_confusion_matrix(cm,classes)
model.save('keras_intro.h5')
from keras.models import load_model
loaded_model=load_model('keras_intro.h5')

loaded_model.summary()
loaded_model.get_weights()
loaded_model.optimizer

json_model=model.to_json()
json_model

from keras.models import model_from_json
model_arch=model_from_json(json_model)
model_arch.summary()
model.save_weights('model_weights.h5')

model2=Sequential([
    Dense(16,input_shape=(1,),activation='relu'),
    Dense(32,activation='relu'),
    Dense(2,activation='softmax')
    
])
model2.load_weights('model_weights.h5')
predict2=model2.predict(scaled_test_samples,batch_size=10,verbose=0)
for i in predict2:
    print(i)
