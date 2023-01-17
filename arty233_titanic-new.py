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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data.shape
data_test.shape
y_train = data.Survived #target
data.info()
corrmat = data.corr()
plt.subplots(figsize=(11, 8))
sns.heatmap(corrmat, vmax=.8, square=True);
data.drop('Cabin', axis=1, inplace=True)
data_test.drop('Cabin', axis=1, inplace=True)
le = LabelEncoder()
le.fit(data.Sex)
data.Sex = le.transform(data.Sex)
data.info()
le.fit(data_test.Sex)
data_test.Sex = le.transform(data_test.Sex)
data['Age'].fillna(data['Age'].mean(), inplace=True)
data_test['Age'].fillna(data_test['Age'].mean(), inplace=True)
X_test, X_train, y_train, y_test = train_test_split(data, data.Survived, test_size = 0.3,
                                                    random_state = 25)
#для тестирования без комита, сделаем разбиение
features = ['Pclass','Fare','Age','Sex', 'SibSp', 'Parch']
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(data[features],data.Survived)

data_test.Fare.fillna(data_test['Fare'].mean(), inplace=True)
preds = clf.predict(data_test[features])
submission = pd.DataFrame({
        "PassengerId": data_test.PassengerId,
        "Survived": preds
    })
submission.to_csv('titanic.csv', index=False)
