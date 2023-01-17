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
train = pd.read_csv('/kaggle/input/ska-data-challenge-test/train.csv')

# Keep all columns except the Outcome and the Id column

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction', 'Age']

X = train[feature_names]

y = train[['Outcome']]

test = pd.read_csv('/kaggle/input/ska-data-challenge-test/test.csv')

testX = test[feature_names]

# If you wanted to split the training set into training+test for cross validation

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Apply scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test  = scaler.transform(X_test)

testX   = scaler.transform(testX)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'

     .format(knn.score(X_train, y_train.values.ravel())))

print('Accuracy of K-NN classifier on test set: {:.2f}'

     .format(knn.score(X_test, y_test.values.ravel())))

labels = knn.predict(testX)

output = pd.DataFrame()

output['Outcome'] = labels

output['Id'] = test['Id']

output = output[['Id', 'Outcome']]

pd.DataFrame(output).to_csv("/kaggle/working/rosie-submission.csv", index=False)