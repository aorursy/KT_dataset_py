# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data
labels = train_data['Survived']
features = train_data[['Pclass','Sex','SibSp','Parch']]
features
dict1 = {'male':0,'female':1}
features.Sex = features.Sex.apply(lambda x : dict1[x])
features
from sklearn.ensemble import GradientBoostingClassifier
features
features_train = features.to_numpy()
labels_train = labels.to_numpy()
labels_train
clf = GradientBoostingClassifier()
clf.fit(features_train,labels_train)
clf.score(features_train,labels_train)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
features = test_data[['Pclass','Sex','SibSp','Parch']]
#labels = test_data['Survived']
features.Sex = features.Sex.apply(lambda x : dict1[x])
features_test = features.to_numpy()
#labels_test = labels.to_numpy()
clf.predict(features_test)
pd1 = test_data['PassengerId']

pd1
pd1['Survived'] = clf.predict(features_test)
pd1

list(pd1)
pd1 = test_data[['PassengerId']]
pd1

pd1['Survived'] = clf.predict(features_test)
pd1

pd1.to_csv('results.csv')
pd1.to_csv('/kaggle/working/results.csv',index=False)
