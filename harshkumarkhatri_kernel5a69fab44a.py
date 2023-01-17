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
dataset=pd.read_csv('../input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv')

dataset.head()
dataset.info()
dataset.isnull().sum()
# First finding the mean of a column to fill na values with it

mean=dataset['education'].mean()

dataset['education']=dataset['education'].fillna(mean)
dataset.isnull().sum()
mean=dataset['cigsPerDay'].mean()

dataset['cigsPerDay']=dataset['cigsPerDay'].fillna(mean)
mean=dataset['BPMeds'].mean()

dataset['BPMeds']=dataset['BPMeds'].fillna(mean)
mean=dataset['glucose'].mean()

dataset['glucose']=dataset['glucose'].fillna(mean)
mean=dataset['totChol'].mean()

dataset['totChol']=dataset['totChol'].fillna(mean)
mean=dataset['BMI'].mean()

dataset['BMI']=dataset['BMI'].fillna(mean)
mean=dataset['heartRate'].mean()

dataset['heartRate']=dataset['heartRate'].fillna(mean)
# Importing the train test 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset.drop("TenYearCHD",axis=1),dataset['TenYearCHD'],test_size=0.33,random_state=10)
x_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression

reg=LogisticRegression(solver='liblinear',random_state=1)
reg.fit(x_train,y_train)
predict=reg.predict(x_test)
print(predict)
from sklearn.metrics import accuracy_score
accuracy_score(predict,y_test)
dataset.shape
# ?Visualizing the result

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(4,4))

sns.pairplot(dataset)
from sklearn.metrics import confusion_matrix
confusion_matrix(predict,y_test)
sns.heatmap(confusion_matrix(predict,y_test),annot=True,cmap='YlGnBu')