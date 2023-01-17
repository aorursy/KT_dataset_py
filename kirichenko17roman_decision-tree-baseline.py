import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_train.csv")
train.head()
train.info()
train_y = train.covid_19.values
train = train.drop("covid_19", axis='columns')
test = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_test.csv")

test.head()
X = pd.concat([train, test]).select_dtypes(exclude=['object']).fillna(-99).values
X.shape
train_X = X[:4000, :]

test_X = X[4000:, :]
from sklearn.tree import DecisionTreeClassifier



tree_clf = DecisionTreeClassifier(

    max_depth=10, 

    min_samples_leaf=21, 

    max_features=0.9, 

    criterion="gini",                                    

    random_state=1)



dt = tree_clf.fit(train_X, train_y)

y_pred = dt.predict_proba(test_X)
y_pred
sub = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_submission.csv")



sub["covid_19"] = y_pred[:, 1]



sub.to_csv("submission.csv", index=False)