# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# import dataset 

cancer = pd.read_csv("../input/data.csv")



# Any results you write to the current directory are saved as output.
# General

sns.set()

# getting the shape

cancer.info()



cancer.describe()

cancer.shape

cancer.dtypes
cancer.head(2)
cancer.drop(['Unnamed: 32',"id"], axis=1)

cancer.head(2)
cancer.isnull().sum()
sns.countplot(x="diagnosis", data=cancer, palette="bwr")

plt.title('Distibution of Benign and Malignant')

plt.show()
corr = cancer.corr()

corr.head()


plt.figure(figsize=(20,20))

sns.heatmap(corr)

#corelation matrix

plt.figure(figsize=(20,20))

sns.heatmap(cbar=False,annot=True,data=cancer.corr()*100,cmap='coolwarm')

plt.title('% Corelation Matrix')

plt.show()
list_mean=['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean',

      'smoothness_mean','compactness_mean','concavity_mean',

      'concave points_mean','symmetry_mean','fractal_dimension_mean']

cancer_mean=cancer[list_mean]

cancer_mean.head()

#correlation map

f,ax = plt.subplots(figsize=(9, 8))

sns.heatmap(cancer_mean.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
list_SE=['diagnosis','radius_se','texture_se','perimeter_se','area_se',

      'smoothness_se','compactness_se','concavity_se',

      'concave points_se','symmetry_se','fractal_dimension_se']

cancer_SE=cancer[list_SE]

cancer_SE.head()

#correlation map

f,ax = plt.subplots(figsize=(9, 8))

sns.heatmap(cancer_SE.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
list_worst=['diagnosis','radius_worst','texture_worst','perimeter_worst','area_worst',

      'smoothness_worst','compactness_worst','concavity_worst',

      'concave points_worst','symmetry_worst','fractal_dimension_worst']

cancer_worst=cancer[list_worst]

cancer_worst.head()

#correlation map

f,ax = plt.subplots(figsize=(9, 8))

sns.heatmap(cancer_worst.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);


from sklearn import datasets

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

cancer_SE_y = cancer_SE.loc[:, ["diagnosis"]] ### Selective columns based slicing

cancer_SE_y.head(2)

cancer_SE_x = cancer_SE.iloc[:,2:11]

cancer_SE_x.head(2)
#splitting the dataset into training and testing sets



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(cancer_SE_x, cancer_SE_y, test_size = 0.25, random_state = 16)



print("Shape of x_train :", x_train.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_test :", y_test.shape)

print("Description of x_train",x_train.describe())
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# creating a model

model = RandomForestClassifier(n_estimators = 400, max_depth = 10)



# feeding the training set into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# Calculating the accuracies

print("Training accuracy :", model.score(x_train, y_train))

print("Testing accuarcy :", model.score(x_test, y_test))



# classification report

cr = classification_report(y_test, y_pred)

print(cr)



# confusion matrix 

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, cmap = 'winter')

plt.title('Confusion Matrix', fontsize = 20)

plt.show()