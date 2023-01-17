# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
features = pd.concat([data.get(['Fare', 'Age']),

                      pd.get_dummies(data.Sex, prefix='Sex'),

                      pd.get_dummies(data.Pclass, prefix='Pclass'),

                      pd.get_dummies(data.Embarked, prefix='Embarked')],

                     axis=1)

features = features.drop('Sex_male', 1)

features = features.fillna(-1)
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='mean', missing_values=-1)

imputer.fit(features)
features.values
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C = 100.)

train = features.values

target = data.values[:,1].astype(float)
logreg.fit(train,target)
X = pd.concat([test.get(['Fare', 'Age']),

                      pd.get_dummies(test.Sex, prefix='Sex'),

                      pd.get_dummies(test.Pclass, prefix='Pclass'),

                      pd.get_dummies(test.Embarked, prefix='Embarked')],

                     axis=1)

X = X.drop('Sex_male', 1)

X = X.fillna(-1)

imputer.fit(X)
prediction = logreg.predict(X)
prediction
test = pd.read_csv('../input/test.csv')