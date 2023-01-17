# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train = train.drop(['Soil_Type7', 'Soil_Type15'], axis=1)

test = test.drop(['Soil_Type7', 'Soil_Type15'], axis=1)
test.describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Cover_Type',axis=1), 

                                                    train['Cover_Type'], test_size=0.30, 

                                                    random_state=101)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50)

rfc.fit(X_train,y_train)

pred_rfc = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,pred_rfc))

print(confusion_matrix(y_test,pred_rfc))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50)

rfc.fit(train.drop('Cover_Type',axis=1),train['Cover_Type'])

pred_rfc = rfc.predict(test)
submission= pd.DataFrame({

    "Id" : test["Id"],

    'Cover_Type' : pred_rfc

})



submission.to_csv('submission.csv',index=False)