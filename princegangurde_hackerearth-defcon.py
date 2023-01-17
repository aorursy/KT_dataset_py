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
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error,confusion_matrix,accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
import pandas as pd

sample_submission = pd.read_csv("../input/hackerearth-defcon/sample_submission.csv")

X_test = pd.read_csv("../input/hackerearth-defcon/test.csv")

train = pd.read_csv("../input/hackerearth-defcon/train.csv")
train['DEFCON_Level'].value_counts()
parameter_train = X_test.columns

X = train[parameter_train]

Y = train['DEFCON_Level']

Y_test = sample_submission.DEFCON_Level
#splitting into train and test set 

train_X,val_x,train_Y,val_y = train_test_split(X,Y,random_state =1)
rfc_model = RandomForestClassifier(n_estimators = 500,random_state = 0)

rfc_model.fit(train_X,train_Y)

print(mean_absolute_error(rfc_model.predict(val_x),val_y))

print(confusion_matrix(rfc_model.predict(val_x),val_y))
print("Random Forest model accuracy(in %):", metrics.accuracy_score(val_y,rfc_model.predict(val_x))*100)
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 30)

KNN.fit(train_X,train_Y)
print(mean_absolute_error(KNN.predict(val_x),val_y))

print(confusion_matrix(KNN.predict(val_x),val_y))

print("\n Random Forest model accuracy(in %):", metrics.accuracy_score(val_y,KNN.predict(val_x))*100)
DTC = DecisionTreeClassifier(min_samples_split=100,random_state = 0)

DTC.fit(train_X,train_Y)
print(mean_absolute_error(DTC.predict(val_x),val_y))

print(confusion_matrix(DTC.predict(val_x),val_y))

print("\n Random Forest model accuracy(in %):", metrics.accuracy_score(val_y,DTC.predict(val_x))*100)
preds = np.round(DTC.predict(X_test))

output = pd.DataFrame({"ID": X_test.ID,"DEFCON_Level": preds })

output.to_csv("hackerearth-defcon.csv",index = False)