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


# Random Forest Classification



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd







# Importing the dataset

dataset = pd.read_csv('/kaggle/input/titanic/train.csv')



cols = list(dataset.columns.values) #Make a list of all of the columns in the df

cols.pop(cols.index('Survived')) #Remove b from list

dataset = dataset[cols+['Survived']] #Create new dataframe with columns in the order you want





dataset = dataset.fillna(dataset.mean())

dataset['Sex'].replace(['female','male'],[0,1],inplace=True)







X = dataset.iloc[:, [1,3,4,5,6,8]].values

y = dataset.iloc[:, 11].values



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



cm
#Entire Training set



X_train = X

y_Train = dataset.iloc[:, 11].values



X_test = pd.read_csv('/kaggle/input/titanic/test.csv')



X_test = X_test.fillna(X_test.mean())

X_test['Sex'].replace(['female','male'],[0,1],inplace=True)





X_test = X_test.iloc[:, [1,3,4,5,6,8]].values



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_Train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)