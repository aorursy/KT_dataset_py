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
from sklearn import preprocessing
from scipy.stats import pearsonr

# machine learning  - supervised
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

# machine learning  - unsupervised
from sklearn import decomposition
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
dataset = pd.read_csv('../input/indian_liver_patient.csv')
dataset.head(10)
dataset.info()
dataset.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(data = dataset, x = "Dataset", label = "Count")
sns.countplot(data = dataset, x = "Gender", label = "Count")
correlations = dataset.corr()
plt.figure(figsize=(10,10))
g = sns.heatmap(correlations,cbar = True, square = True, annot=True, fmt= '.2f', annot_kws={'size': 10})

le = preprocessing.LabelEncoder()
le.fit(dataset['Gender'])


dataset['Gender_Encoded'] = le.transform(dataset['Gender'])
dataset.head()
dataset['Albumin_and_Globulin_Ratio'].fillna(dataset['Albumin_and_Globulin_Ratio'].median(), inplace=True)
g = sns.PairGrid(dataset, hue = "Dataset", vars=['Age','Gender_Encoded','Total_Bilirubin','Total_Protiens'])
g.map(plt.scatter)
plt.show()
X = dataset[['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 
        'Albumin', 'Albumin_and_Globulin_Ratio','Gender_Encoded']]
y = dataset[['Dataset']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
print(X_train.shape,X_test.shape)
#Random Forest
rf = RandomForestClassifier(n_estimators=25, random_state=0)
rf.fit(X_train, y_train)
rf_predicted = rf.predict(X_test)

random_forest_score      = round(rf.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(rf.score(X_test, y_test) * 100, 2)

print('Random Forest Score: ', random_forest_score)
print('Random Forest Test Score: ', random_forest_score_test)
print('Accuracy: ', accuracy_score(y_test,rf_predicted))
print('\nClassification report: \n', classification_report(y_test,rf_predicted))

g = sns.heatmap(confusion_matrix(y_test,rf_predicted), annot=True, fmt="d")

import tensorflow as tf
from tensorflow import keras
#Neural Network
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(32,input_shape=(10,),activation = 'relu'))
model.add(layers.Dense(8,activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(0.01),
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
model.fit(X_train,y_train,epochs = 10, batch_size = 100)
