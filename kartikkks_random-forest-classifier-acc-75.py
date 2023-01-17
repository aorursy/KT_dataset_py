# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook

from sklearn.metrics import accuracy_score, mean_squared_error

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample = pd.read_csv("../input/sample_submission.csv")
train.tail()
sample.head()
Id = test.iloc[:,0]

Y_train = train.iloc[:,-1]

X_train = train.iloc[:,1:-1]

X_test = test.iloc[:,1:]
X_test.describe()
#splitting into test and validation data

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 0)
X_train_scaling = X_train.iloc[:,0:10]

X_val_scaling = X_val.iloc[:,0:10]

X_test_scaling = X_test.iloc[:,0:10]

#scaling required classes

from sklearn.preprocessing import StandardScaler, OneHotEncoder

sc_X = StandardScaler()

X_train_scaling = sc_X.fit_transform(X_train_scaling)

X_val_scaling = sc_X.transform(X_val_scaling)

X_test_scaling = sc_X.transform(X_test_scaling)
X_train = np.concatenate((X_train_scaling, X_train.values[:,10:]), axis=1)

X_val = np.concatenate((X_val_scaling, X_val.values[:,10:]), axis=1)

X_test = np.concatenate((X_test_scaling, X_test.values[:,10:]), axis=1)
#training on random forest classifier

from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, Y_train)

#accuracy check

Y_pred_train = classifier.predict(X_train)

Y_pred_val = classifier.predict(X_val)

accuracy_train = accuracy_score(Y_pred_train, Y_train)

accuracy_val = accuracy_score(Y_pred_val, Y_val) 



print("Training accuracy", round(accuracy_train, 5))

print("Validation accuracy", round(accuracy_val, 5))

#predictiond results

preds = classifier.predict(X_test)



sub = pd.DataFrame({"Id": Id,"Cover_Type": preds})

sub.to_csv("forest.csv", index=False) 