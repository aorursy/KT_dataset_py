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
import pandas as pd

sample_submisiion = pd.read_csv('../input/forest-cover-type-prediction/sampleSubmission.csv') 

test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')

train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
train.info()
test.info()
Y = train['Cover_Type']

X = train.drop(['Cover_Type', 'Id'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
train
train['Soil_Type32'].unique()
#https://medium.com/@rismitawahyu/diabetes-classification-using-k-nearest-neighbors-knn-in-phyton-2b22e6c41f3b
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')

RFC.fit(X_train, Y_train)

Y_pred = RFC.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=100, bootstrap = True)

RFC.fit(X_train, Y_train)

Y_pred = RFC.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))
#https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
X = train.drop(['Id', 'Cover_Type'], axis=1)

Y = train['Cover_Type']

X_testing = test.drop(['Id'], axis=1)

Id = test['Id']
RFC = RandomForestClassifier(n_estimators=100)

RFC.fit(X,Y)

Y_pred = RFC.predict(X_testing)
submission = pd.DataFrame({

    'Id' : Id,

    'Cover_Type' : Y_pred

})
submission