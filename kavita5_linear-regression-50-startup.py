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


# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import os

import sklearn
print(os.listdir("../input"))
# Importing the dataset

dataset = pd.read_csv('../input/startup-logistic-regression/50_Startups.csv')
dataset.head()
## EDA on Dataset -



#Histgram on Profit

sns.distplot(dataset['Profit'],bins=5,kde=True)
#Correlation  chart on different variables for comparision 

# Profit Vs R & Spend is very linear and almost same for Marketing spend

# Profit spend vs Administration distribution is very scattered 

sns.pairplot(dataset)
# profit split in State level - Looks Florida has the maximum Profit

sns.barplot(x='State',y='Profit',data=dataset, palette="Blues_d")

#sns.lineplot(x='State',y='Profit',data=dataset)
#gives positive & negative relation between categories

sns.heatmap(dataset.corr(), annot=True)
# spread of profit against state 

g=sns.FacetGrid(dataset, col='State')

g=g.map(sns.kdeplot,'Profit')
# spliting Dataset in Dependent & Independent Variables

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 4].values

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()



# Avoiding the Dummy Variable Trap

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

import sklearn



#from sklearn import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



print('Coefficients: \n', regressor.coef_) 

regressor.score(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)
print(regressor)
y_pred