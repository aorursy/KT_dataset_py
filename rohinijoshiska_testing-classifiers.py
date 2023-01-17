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
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

labels = logreg.predict(testX)

#Also generate labels by fitting all the training data, no scaling/normalising

logreg.fit(X, y)

labels_useAll = logreg.predict(test[feature_names])



output = pd.DataFrame()

output['Outcome'] = labels

output['Id'] = test['Id']

output = output[['Id', 'Outcome']]

pd.DataFrame(output).to_csv("/kaggle/working/rohini-submission.csv", index=False)



output_useAll = pd.DataFrame()

output_useAll['Outcome'] = labels_useAll

output_useAll['Id'] = test['Id']

output_useAll = output_useAll[['Id', 'Outcome']]

pd.DataFrame(output_useAll).to_csv("/kaggle/working/rohini-submission-use-all.csv", index=False)
# Alternatively, user could also upload their result as an input and write it back out in the outputs to serve as a submission

resultComputedElsewhere = pd.read_csv('/kaggle/input/submission/rohini-submission-remove-column.csv')

pd.DataFrame(resultComputedElsewhere).to_csv("/kaggle/working/rohini-external-submission.csv", index=False)