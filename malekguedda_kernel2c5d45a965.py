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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.drop(['Name','Ticket','Cabin','Parch','SibSp','PassengerId'], axis = 1, inplace = True)
train.dropna(axis = 0, inplace = True)
train['Sex']=train['Sex'].replace(['male','female'], [0,1])
train['Embarked']=train['Embarked'].replace(['S','C','Q'], [1,2,3])
x = train.drop(['Survived'], axis = 1)
y = train['Survived']

test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.drop(['Name','Ticket','Cabin','Parch','SibSp','PassengerId'], axis = 1, inplace = True)
test.dropna(axis = 0, inplace = True)
test['Sex']=test['Sex'].replace(['male','female'], [0,1])
test['Embarked']=test['Embarked'].replace(['S','C','Q'], [1,2,3])

from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler().fit_transform(x)
x_data = pd.DataFrame(x_scale)
x_s = StandardScaler().fit_transform(test)
x_test = pd.DataFrame(x_s)

from sklearn.ensemble import  RandomForestClassifier
model = RandomForestClassifier( n_estimators = 100, max_depth = 30 )
model.fit(x_data,y)
model.score(x_data,y)
print(model.predict(x_test))
#i tried a lot of algorithms and i even tried to combine them but this gave me highest score so i don't know if this is okay or i'm in overfitting


import numpy as np
def predict_titanic(model,pclass,sex,age,fare,embarked):
    data = np.array([pclass,sex,age,fare,embarked ]).reshape(1,5)
    return (model.predict(data))
predict_titanic(model,1,0,14,75000,2)
