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
train_path = '../input/learn-together/train.csv'

test_path = '../input/learn-together/test.csv'



train_data = pd.read_csv(train_path, index_col='Id')

test_data = pd.read_csv(test_path, index_col='Id')



train_data.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



y = train_data['Cover_Type']

X = train_data.drop(['Cover_Type'], axis=1)



# split the dataset in train/validation

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)



# create and train the classifier

clf = RandomForestClassifier(n_estimators=100, 

                             random_state=0)

clf.fit(X_train, y_train)



# get predictions for validation data

y_preds = clf.predict(X_valid)
from sklearn.metrics import accuracy_score



# get accuracy

accuracy = accuracy_score(y_valid, y_preds)

print('Accuracy: ', accuracy)
# get feature importance

feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)



feature_imp.head()
# create a new model, just using the most important features

n_features = 25

top_features = feature_imp.head(n_features).keys()

X2 = X[top_features]



# split the dataset in train/validation

X_train, X_valid, y_train, y_valid = train_test_split(X2, y, test_size=0.2, random_state=0)



# create and train the classifier

clf2 = RandomForestClassifier(n_estimators=100, 

                             random_state=0)

clf2.fit(X_train, y_train)



# get predictions for validation data

y_preds2 = clf2.predict(X_valid)



# get accuracy

accuracy = accuracy_score(y_valid, y_preds2)

print('Accuracy: ', accuracy)
# get the predictions for the test data

final_preds = clf.predict(test_data)



# create output file

output = pd.DataFrame({'Id': test_data.index,

                       'Cover_Type': final_preds})

output.to_csv('submission.csv', index=False)