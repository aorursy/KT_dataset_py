# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

kf = KFold(n_splits=5)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def clean_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Cabin'] = df['Cabin'].fillna('NULL')
    df['Embarked'] = df['Embarked'].fillna('NULL')
    return df
train_df = clean_data(pd.read_csv("/kaggle/input/titanic/train.csv"))
train_df
def onehot(a):
    return OneHotEncoder().fit_transform(np.array(a).reshape(-1, 1)).toarray()

def onehot_feature(a):
    return np.array(onehot(a)).T

def build_features_X(df):
    X = []
    
    X.extend(onehot_feature(list(df['Pclass'])))
    X.extend(onehot_feature(list(df['Sex'])))
    X.append(list(df['Age']))
    X.append(list(df['SibSp']))
    X.append(list(df['Parch']))
    X.append(list(df['Fare']))
    X.append(LabelEncoder().fit_transform(df['Cabin']))
    X.append(LabelEncoder().fit_transform(df['Embarked']))
    
    return np.array(X).T

def build_features_Y(df, oh=True):
    if not oh:
        return np.array(df['Survived'])
    return onehot(np.array(df['Survived']))

build_features_Y(train_df).shape
def train_decision_tree(X, Y):
    model = DecisionTreeClassifier()
    model.fit(X, Y)
    return model

train_decision_tree(build_features_X(train_df), build_features_Y(train_df))
def train_naive_bayes(X, Y):
    model = GaussianNB()
    model.fit(X, Y)
    return model

train_naive_bayes(build_features_X(train_df), build_features_Y(train_df))
def train_nn(X, Y):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(11,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, epochs=1500)
    
    return model

#train_nn(build_features_X(train_df), build_features_Y(train_df))
def cross_val_decision_tree(df):
    X = build_features_X(df)
    Y = build_features_Y(df)
    
    precisions = []
    recalls = []
    fscores = []
    fscore_averages = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model = train_decision_tree(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        
        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test, Y_pred)
        _, _, fscore_average, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
        
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        fscore_averages.append(fscore_average)
        
    return precisions, recalls, fscores, fscore_averages

def cross_val_naive_bayes(df):
    X = build_features_X(df)
    Y = build_features_Y(df, False)
    
    precisions = []
    recalls = []
    fscores = []
    fscore_averages = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model = train_naive_bayes(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        
        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test, Y_pred)
        _, _, fscore_average, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
        
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        fscore_averages.append(fscore_average)
        
    return precisions, recalls, fscores, fscore_averages

def cross_val_nn(df):
    X = build_features_X(df)
    Y = build_features_Y(df)
    Ylabel = build_features_Y(df, False)
    
    precisions = []
    recalls = []
    fscores = []
    fscore_averages = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Ylabel[test_index]
        
        model = train_nn(X_train, Y_train)
        
        Y_pred = np.argmax(model.predict(X_test), axis=1)
        
        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test, Y_pred)
        _, _, fscore_average, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
        
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        fscore_averages.append(fscore_average)
        
    return precisions, recalls, fscores, fscore_averages
        
#cross_val_decision_tree(train_df)
#cross_val_nn(train_df)
precisions_d, recalls_d, fscores_d, fscore_averages_d = cross_val_decision_tree(train_df)
precisions_n, recalls_n, fscores_n, fscore_averages_n = cross_val_naive_bayes(train_df)
precisions_nn, recalls_nn, fscores_nn, fscore_averages_nn = cross_val_nn(train_df)
fscore_averages_d = list(map(lambda x: [x, x], fscore_averages_d))
fscore_averages_n = list(map(lambda x: [x, x], fscore_averages_n))
fscore_averages_nn = list(map(lambda x: [x, x], fscore_averages_nn))
precisions_nn
def visualize(d, n, nn, titleprefix):
    fig, axs = plt.subplots(6, 2, figsize=(10,30))

    for i in range(5):
        for j in range(2):
            axs[i][j].bar(['DT', 'NB', 'NN'], [d[i][j], n[i][j], nn[i][j]])
            axs[i][j].set_title(titleprefix + " of Fold " + str(i+1) + " Class " + str(j))
            
    average_d = np.array(d).mean(axis=0)
    average_n = np.array(n).mean(axis=0)
    average_nn = np.array(nn).mean(axis=0)
    
    for j in range(2):
        axs[5][j].bar(['DT', 'NB', 'NN'], [average_d[j], average_n[j], average_nn[j]])
        axs[5][j].set_title("Average " + titleprefix + " of all fold " + str(i+1) + " Class " + str(j))
            
    plt.show()
visualize(precisions_d, precisions_n, precisions_nn, 'Precision')
visualize(recalls_d, recalls_n, recalls_nn, 'Recall')
visualize(fscores_d, fscores_n, fscores_nn, 'F-Measure')
visualize(fscore_averages_d, fscore_averages_n, fscore_averages_nn, 'Average F-Measure')
