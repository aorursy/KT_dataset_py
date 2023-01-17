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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_label = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')





# 删去无关字段

train_data.drop(labels=['Cabin','Ticket','Name'],inplace=True,axis=1)

test_data.drop(labels=['Cabin','Ticket','Name'],inplace=True,axis=1)



# 填补Age的缺失值

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())





train_data.dropna(inplace=True)

test_data.dropna(inplace=True)



from sklearn.preprocessing import OneHotEncoder

oneencoder = OneHotEncoder()

train_result = oneencoder.fit_transform(train_data['Embarked'].values.reshape(-1,1))

test_result = oneencoder.fit_transform(test_data['Embarked'].values.reshape(-1,1))





train_data.drop('Embarked',axis=1,inplace=True)

test_data.drop('Embarked',axis=1,inplace=True)



train_data = pd.concat([train_data,pd.DataFrame(train_result.toarray())],axis=1)

test_data = pd.concat([test_data,pd.DataFrame(test_result.toarray())],axis=1)





train_data['Sex'] = (train_data.Sex == 'male').astype('int')

test_data['Sex'] = (test_data.Sex == 'male').astype('int')



test_data = pd.merge(test_data,test_label,left_on='PassengerId',right_on='PassengerId',how='inner')





train_data.dropna(inplace=True)

test_data.dropna(inplace=True)



train_label = train_data.iloc[:,train_data.columns == 'Survived']

train_data = train_data.iloc[:,train_data.columns != 'Survived']

test_label = test_data['Survived']

test_data = test_data.iloc[:,test_data.columns != 'Survived']









train_label = np.array(train_label).ravel()

test_label = np.array(test_label).ravel()





from sklearn.linear_model import LogisticRegression



lg = LogisticRegression(max_iter=500

                        ,random_state=0).fit(train_data,train_label)



lg.score(test_data,test_label)