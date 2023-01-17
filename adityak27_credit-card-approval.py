# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/svm-classification/UniversalBank.csv')

data.head()
# Drop columns which are not necessary

data.drop(["ID","ZIP Code"],axis=1,inplace=True)

data.head()
data.shape
data.info()
data.describe() # Summary Stats of the Data Set
data[data['Experience'] < 0] = 0

data.describe()
# Plotting the Heatmap of the Dataset

plt.rcParams['figure.figsize'] = (15, 15)

plt.style.use('ggplot')

sns.heatmap(data.corr() ,annot = True)

plt.title('Heatmap', fontsize = 30)
sns.distplot(data['Age'])
sns.distplot(data['Experience'])
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,mean_squared_error,accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



# Split the data into X and y

X = data.iloc[:,:-1]

y = data["CreditCard"]



# Split the data into trainx, testx, trainy, testy 

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.20)



## Print the shape of X_train, X_test, y_train, y_test

print(trainx.shape)

print(testx.shape)

print(trainy.shape)

print(testy.shape)
# Logistic Regression

X = trainx

y = trainy

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 

model = LogisticRegression()

model.fit(X , y)

predicted_classes = model.predict(X)

accuracy = accuracy_score(y,predicted_classes)

parameters = model.coef_
print(accuracy)

print(parameters)

print(model)
model.fit(testx , testy)

predicted_classes_test = model.predict(testx)

accuracy = accuracy_score(testy,predicted_classes_test)

print(accuracy)
from sklearn.naive_bayes import GaussianNB



NB = GaussianNB()



NB.fit(X , y)



NB_train_pred = NB.predict(X)

print(accuracy_score(y,NB_train_pred))



NB_test_pred = NB.predict(testx)

print(accuracy_score(testy,NB_test_pred))