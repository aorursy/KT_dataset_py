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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
print("Training data shape = {}".format(train_data.shape))

print("Test data shape = {}".format(test_data.shape))

train_data.head()
train_data = train_data.drop(columns=['Name','Cabin'])

train_data['family_member'] = train_data['SibSp'] + train_data['Parch']

train_data = train_data.drop(columns=['SibSp', 'Parch'])
test_data = test_data.drop(columns=['Name','Cabin'])

test_data['family_member'] = test_data['SibSp'] + test_data['Parch']

test_data = test_data.drop(columns=['SibSp', 'Parch'])
train_data.fillna(0,inplace=True)

test_data.fillna(0,inplace=True)
X = train_data.drop(columns=['Survived'])

Y = train_data['Survived']



cate_features_index = np.where(X.dtypes != float)[0]
from catboost import CatBoostClassifier, cv, Pool

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.90,random_state=42)

clf =CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42,learning_rate=0.05)
clf.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest), early_stopping_rounds=50,verbose=0)
test_id = test_data.PassengerId

test_data.isnull().sum()
prediction = clf.predict(test_data)
df_sub = pd.DataFrame()

df_sub['PassengerId'] = test_id

df_sub['Survived'] = prediction.astype(np.int)



df_sub.to_csv('gender_submission.csv', index=False)