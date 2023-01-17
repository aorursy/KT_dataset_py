# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

#training and testing
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

plt.style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')  #read the csv file and save it into a variable
df_train.head(n=10)   
df_test = pd.read_csv('../input/test.csv')  #read the csv file and save it into a variable
df_test.head(n=10)  
def gen_graph(history, title):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss ' + title)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
def process_data(data):
    X_train = data[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    X_train = X_train.fillna(X_train.mean())

    X_train.loc[:,'Sex'] = X_train['Sex'].replace(['female','male'],[0,1]).values
    X_train.loc[:,'Pclass'] = X_train.loc[:,'Pclass'] - 1

    sex_cols = tf.keras.utils.to_categorical(X_train['Sex'], num_classes=2)
    class_cols = tf.keras.utils.to_categorical(X_train['Pclass'], num_classes=3)

    return X_train
X_train = process_data(df_train)
X_train.head()
X_test = process_data(df_test)
X_test.head()
Y_train = df_train["Survived"]
y_train_onehot = tf.keras.utils.to_categorical(Y_train, num_classes=2)
print(y_train_onehot)
def get_mlp(input_size, output_size):
    #Random Forest Classifier
    model = RandomForestClassifier()
    return model
input_n = X_train.shape[1]
output_n = 2
model = get_mlp(input_n, output_n)
model.fit(X_train, Y_train)
preds = model.predict(X_test)
preds.shape

sub = pd.read_csv('../input/gender_submission.csv')  #read the csv file and save it into a variable
sub.head(n=10)  
sub.Survived = preds
sub.head(n=10)  
sub.to_csv("submissiondnn.csv", index=False)