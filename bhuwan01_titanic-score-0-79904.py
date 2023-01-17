# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import catboost



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train= pd.read_csv("/kaggle/input/titanic/train.csv")

                   

test = pd.read_csv("/kaggle/input/titanic/test.csv")
#inatialize the new columns

# idont know why it show error otherwise

test['Master']='a'

train['Surname'] = 'b'

train['Master'] = 'a'

test['Surname'] = 'b'


for i in range(len(test)):

    test['Master'][i] = (test['Name'][i].split(',')[1].split('.')[0]=='Master')
for i in range(len(train)):

    train['Master'][i] = (train['Name'][i].split(',')[1].split('.')[0]=='Master')


for i in range(len(train)):

    train['Surname'][i] = train['Name'][i].split(',')[0]


for i in range(len(test)):

    test['Surname'][i] = test['Name'][i].split(',')[0]
feature = ['Sex', 'Pclass', 'Embarked', 'Master', 'Surname']

train[feature] = train[feature].fillna('')

test[feature] = test[feature].fillna('')
master = {True:1,False:0}

train['Master']=train['Master'].map(master)

test['Master']=test['Master'].map(master)
model = catboost.CatBoostClassifier(one_hot_max_size=4, iterations=100, random_seed=0, verbose=False)
model.fit(train[feature], train['Survived'], cat_features=[0, 2, 4])
pred = model.predict(test[feature]).astype('int')
model.score(train[feature],train['Survived'])
sub =pd.DataFrame()

sub['PassengerId'] = test['PassengerId']
sub['Survived'] = pred
sub.to_csv('subm.csv',index=False)