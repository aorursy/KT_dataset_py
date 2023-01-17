# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
def impute_age(cols):

    Age= cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        if Pclass==2:

            return 29

        else:

            return 24

    return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)

embarked = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
train = pd.concat([train,sex,embarked],axis=1)
train.head()
sns.heatmap(train.isnull(),yticklabels=False ,cbar=False ,cmap='viridis')
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size =0.3, random_state = 102)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=9,criterion='gini',max_depth=6)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
rf.score(x_test,y_test)
from sklearn.model_selection import cross_val_score

acc_list=[]

for i in range(1,100):

    acc=cross_val_score(RandomForestClassifier(n_estimators=i,criterion='gini',max_depth=9),x_train,y_train,cv=5).mean()

    acc_list.append(acc)
import matplotlib.pyplot as plt

plt.style.use('seaborn')

plt.plot(acc_list)
np.argmax(acc_list)
rf=RandomForestClassifier(n_estimators=np.argmax(acc_list),criterion='gini',max_depth=9)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
rf.score(x_test,y_test)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
sex = pd.get_dummies(test['Sex'],drop_first=True)

embarked = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embarked],axis=1)
test.head()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test['Fare']=test['Fare'].fillna(test['Fare'].mean())
res = rf.predict(test)
new_df= pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':res})
new_df.head()
new_df.shape
new_df.to_csv('gender_submission.csv',index=False)