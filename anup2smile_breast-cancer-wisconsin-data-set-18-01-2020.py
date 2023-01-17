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



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
Data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
Data.head()
Data.info()
Data.describe()
Data.isna().sum()
sns.heatmap(Data.isnull())
Data_target= Data["diagnosis"]
Data_target.head()
from sklearn.preprocessing import LabelEncoder

lc = LabelEncoder()

y = lc.fit_transform(Data_target)
y = pd.DataFrame(y, columns=["diagnosis"])
y.head()
Data.drop("diagnosis", axis=1, inplace=True)
Data = pd.concat([Data,y], axis=1)
plt.figure(figsize=(25,20))



sns.heatmap(Data.corr(), annot=True)
print(Data.columns)
relation = []

a = list(Data.columns)

b = []

for i in range(len(Data.columns)):

    

    rel = Data["diagnosis"].corr(Data.iloc[:,i])

    if rel>0.5:

        print("{} :{}".format(a[i], rel))

        b.append(a[i])

print(b)   
X = Data[['radius_mean', 'texture_mean', 'smoothness_mean','compactness_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'smoothness_se', 'compactness_se','symmetry_se', 'fractal_dimension_se']]
Trail = pd.concat([X,y], axis=1)
sns.pairplot(Trail, hue="diagnosis")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression



LR_model = LogisticRegression()
LR_model.fit(X_train,y_train)
prediction = LR_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix



print(classification_report(y_test, prediction))

print(confusion_matrix(y_test,prediction))