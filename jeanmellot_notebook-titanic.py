# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import numpy as np

import pandas as pd

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
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')



all_data = pd.concat((train, test)).reset_index(drop=True)

#all_data.drop(['Survived'], axis=1, inplace=True)



all_data.describe()
all_data.head()
all_data.isnull().sum()
#all_data = all_data.drop(['PassengerId'], axis=1)
all_data['Title'] = all_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

all_data = all_data.drop(['Name'], axis=1)
all_data['Age']= all_data.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median()))
all_data['#fam']= all_data['SibSp']+all_data['Parch']
all_data = all_data.drop(['Ticket','Fare','Cabin'], axis=1)
all_data
from sklearn.preprocessing import LabelEncoder

# process columns, apply LabelEncoder to categorical features

lbl= LabelEncoder()

lbl.fit(list(all_data['Title'].values)) 

all_data['Title'] = lbl.transform(list(all_data['Title'].values))

lbl.fit(list(all_data['Sex'].values)) 

all_data['Sex'] = lbl.transform(list(all_data['Sex'].values))

lbl.fit(list(all_data['Embarked'].values)) 

all_data['Embarked'] = lbl.transform(list(all_data['Embarked'].values))



all_data
ntrain = train.shape[0]

ntest = test.shape[0]



train = all_data[:ntrain]

test = all_data[ntrain:]
train.corr()
plt.subplots(figsize=(15,8))



sns.heatmap(train.corr(),annot=True,cmap='Oranges')
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier

train

Y_train = train['Survived']

X_train = train[['Pclass','Sex','Age','SibSp','Parch','Embarked','Title','#fam']]
rd=RandomForestClassifier()

rd.fit(X_train,Y_train)



RandomForestClassifier()

#finalMdR is the prediction by RandomForestClassifier

finalMdR=rd.predict(test[['Pclass','Sex','Age','SibSp','Parch','Embarked','Title','#fam']])

finalMdR
ID = test['PassengerId']



submission=pd.DataFrame()

submission['PassengerId'] = ID

submission['Survived'] = finalMdR

submission.to_csv('submissionrd.csv',index=False)
submission