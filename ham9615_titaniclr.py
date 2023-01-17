# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
result = pd.read_csv('../input/gender_submission.csv')

data_cols = ['PassengerId', 'Pclass','Age','Sex','SibSp','Parch','Fare','Cabin','Embarked']



le = LabelEncoder()
imr = Imputer(missing_values='NaN',strategy='median',axis=0)
imr1 = Imputer(missing_values='NaN',strategy='mean',axis=0)
X_train = train_data[data_cols]
X_test = test_data[data_cols]
y_train = train_data[['PassengerId','Survived']]
y_test = result[['PassengerId','Survived']]

X_train['Sex']=le.fit_transform(X_train['Sex'].values)
#print(X_train.loc[:,3])
X_test['Sex'] = le.fit_transform(X_test['Sex'].values)

X_train['Age'] =imr.fit_transform(X_train[['Age']].values)
X_test['Age'] = imr.fit_transform(X_test[['Age']].values)

X_train['Cabin'] =X_train['Cabin'].fillna('0')
X_test['Cabin'] = X_test['Cabin'].fillna('0')

X_train['Cabin'] =le.fit_transform(X_train['Cabin'].values)
X_test['Cabin'] = le.fit_transform(X_test['Cabin'].values)


X_train['Embarked'] = X_train['Embarked'].fillna('0')
X_test['Embarked'] = X_test['Embarked'].fillna('0')

X_train['Embarked'] = le.fit_transform(X_train['Embarked'].values)
X_test['Embarked'] = le.fit_transform(X_test['Embarked'].values)
X_train['Fare'] = imr1.fit_transform(X_train[['Fare']].values)
X_test['Fare'] = imr1.fit_transform(X_test[['Fare']].values)

sc = StandardScaler()
#sc.fit(X_train)
#X_test = sc.transform(X_test)
#X_train = sc.transform(X_train)



knn =LogisticRegression(C=0.56)
knn.fit(X_train,y_train['Survived'])
y_pred = knn.predict(X_test)
y_output = {'PassengerId':y_test['PassengerId'],
            'Survived':y_pred}
print(len(y_pred),'jeez',len(y_test['PassengerId']))
data = pd.DataFrame(y_output,columns=['PassengerId','Survived'])
data.to_csv('result.csv')



print(accuracy_score(y_pred,y_test['Survived']))
#print(y_output)







