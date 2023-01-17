# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

file = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
# Any results you write to the current directory are saved as output.
file.head()
file = file.drop('Serial No.',axis=1)       #serial no is not required
file.columns
#checking null values
file.isnull().sum()
file.boxplot(grid=True)
#finding correlation among features
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(file.corr(), ax=ax, annot=True, linewidths= 0.05,fmt = '.2f', cmap='magma')
plt.show()
plt.hist(file['Chance of Admit '].values)
target = file['Chance of Admit ']
feature = file.drop('Chance of Admit ',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(feature, target, test_size = 0.2,random_state = 41)
import seaborn as sns
plt.figure(figsize=(15,10))
sns.scatterplot(file['CGPA'],file['Chance of Admit '])
plt.show()
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
y_predict = regr.predict(X_test)
print(y_predict)
print("Mean squared error in Linear Regression:",mean_squared_error(y_test,y_predict))