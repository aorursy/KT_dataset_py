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
train_data_path=("../input/train.csv")

train_data=pd.read_csv(train_data_path)

test_data_path=("../input/test.csv")

test_data=pd.read_csv(test_data_path)

train_data.head(5)
import seaborn as sns

sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train_data.drop('Cabin',axis=1,inplace=True)

test_data.drop('Cabin',axis=1,inplace=True)

sns.boxplot(x='Pclass',y='Age',data=train_data)
def impute_age(cols):

    age=cols[0]

    pclass=cols[1]

    if pd.isnull(age):

        if pclass==1:

            return 37

        elif pclass==2:

            return 29

        else:

            return 24

    else:

        return age



train_data['Age']=train_data[['Age','Pclass']].apply(impute_age,axis=1)

test_data['Age']=test_data[['Age','Pclass']].apply(impute_age,axis=1)
train_data.dropna(inplace=True)

train_data.head(5)
sex=pd.get_dummies(train_data['Sex'],drop_first=True)

embarked=pd.get_dummies(train_data['Embarked'],drop_first=True)

sex.head(5)

sex1=pd.get_dummies(test_data['Sex'],drop_first=True)

embarked1=pd.get_dummies(test_data['Embarked'],drop_first=True)
train_data=pd.concat([train_data,sex,embarked],axis=1)

test_data=pd.concat([test_data,sex1,embarked1],axis=1)

train_data.head(5)

test_data.head(5)
train_data.drop(['Ticket','Embarked','Name','Sex','Fare'],axis=1,inplace=True)

train_data.head(5)
test_data.drop(['Ticket','Embarked','Name','Sex','Fare'],axis=1,inplace=True)

test_data.head(5)
X=train_data.drop('Survived',axis=1)

y=train_data['Survived']

X.head(5)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X,y)
prediction=model.predict(test_data)
output=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':prediction})

output.to_csv('Submit.csv',index=False)