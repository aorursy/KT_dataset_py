# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install tensorflow 

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from keras.models import Sequential

from keras.layers import Dense
dataset=pd.read_csv('../input/fetal-health-classification/fetal_health.csv')

dataset
dataset.isnull().sum()
fig=plt.figure()

ax=fig.add_subplot(1,1,1)

ax.hist(dataset['baseline value'])

plt.title('plot of fetal baseline heart rate')

plt.xlabel('baseline value of heart rate')

plt.show()
dataset['prolongued_decelerations'].unique()
a=dataset.describe()

dataset_stats=a.transpose()

dataset_stats
label=dataset['fetal_health']

label
def norm(x):

    return(x-dataset_stats['mean'])/dataset_stats['std']

normed_data=norm(dataset)

normed_data
features=normed_data.drop(columns='fetal_health')

features
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(features,label, test_size=0.3)
def build_model():

    model=keras.Sequential([layers.Dense(8, activation='relu'),

                                                 layers.Dense(6, activation='relu'),layers.Dropout(0.2),

                            layers.Dense(4, activation='softmax')])

    optimizer=tf.keras.optimizers.Adam(0.001)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
model=build_model()
history=model.fit(x_train,y_train, epochs=2000)
plt.plot(history.history['loss'])

plt.show()
x_test
y_test
test_predictions=model.predict(x_test)

test_predictions
pred_classes = model.predict_classes(x_test, verbose=0)

pred_classes
from sklearn.metrics import f1_score

f1 = f1_score(y_test, pred_classes, average=None)

f1
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred_classes)
from sklearn import metrics

print(metrics.classification_report(y_test, pred_classes, digits=3))
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

fpr = {}

tpr = {}

thresh ={}



n_class = 4



for i in range(n_class):    

    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, test_predictions[:,i], pos_label=i)

    

# plotting    

plt.figure(figsize=(15,10))

plt.plot(fpr[1], tpr[1], linestyle='--',color='orange', label='Class 1 vs Rest')

plt.plot(fpr[2], tpr[2], linestyle='--',color='green', label='Class 2 vs Rest')

plt.plot(fpr[3], tpr[3], linestyle='--',color='blue', label='Class 3 vs Rest')

plt.title('Multiclass ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive rate')

plt.legend(loc='best')

plt.savefig('Multiclass ROC',dpi=300)

score = roc_auc_score(y_test, test_predictions, average='weighted', multi_class='ovo', labels=[0,1,2,3])

score
from sklearn.metrics import precision_recall_curve

precision = {}

recall = {}

thresh ={}

n_class = 4



for i in range(n_class):    

    precision[i], recall[i], thresh[i] = roc_curve(y_test, test_predictions[:,i], pos_label=i)

    

plt.figure(figsize=(15,10))

plt.plot(recall[1], precision[1],  linestyle='--',color='orange', label='Class 1 vs Rest')

plt.plot(recall[2], precision[2],  linestyle='--',color='green', label='Class 2 vs Rest')

plt.plot(recall[3], precision[3],  linestyle='--',color='blue', label='Class 3 vs Rest') 

plt.title('Multiclass Precision Recall curve')

plt.xlabel('recall')

plt.ylabel('precision')

plt.legend(loc='upper left')   
from sklearn.metrics import recall_score

recall_score(y_test, pred_classes,  labels=None, pos_label=1, average='weighted', sample_weight=None, zero_division='warn')