# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv('../input/train.csv') # Read the data to train the model
train.head()
x_train = train.drop(['id', 'species'], axis=1).values #Drop the label column
x_train[0]
le = LabelEncoder().fit(train['species']) 
y_train = le.transform(train['species']) #Assign label column
y_train
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial') #Simple Logistic Regression
clf.fit(x_train, y_train) #Fit the model
test = pd.read_csv('../input/test.csv') #Read the data to test the model
test.head()
test_ids = test.pop('id') #Id column for submission file
x_test = test.values #Need data in array format
x_test = scaler.transform(x_test)

y_test = clf.predict_proba(x_test) #Get probability values for each class
submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_) #Create a dataframe to create submission file
submission.head(2)
submission.to_csv('submission_leaf_classification.csv') #Export dataframe into CSV file