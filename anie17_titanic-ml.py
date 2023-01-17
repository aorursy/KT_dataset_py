import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.head()
sns.heatmap(X_test.isnull(),yticklabels=False,cmap='viridis')
#class 1 =38, class 2 = 29 class 3= 24

def age_calc(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 38

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
test['Age']=test[['Age','Pclass']].apply(age_calc,axis=1)
test.drop('Cabin',axis=1,inplace=True)
sex=pd.get_dummies(test['Sex'],drop_first=True)
Embarked=pd.get_dummies(test['Embarked'],drop_first=True)
Pclass=pd.get_dummies(test['Pclass'],drop_first=True)
test=pd.concat([test,sex,Embarked,Pclass],axis=1)
test.drop(['Sex','Name','Ticket','Embarked','Pclass'],axis=1,inplace=True)
test.columns
from sklearn.model_selection import train_test_split
X_train=train[['PassengerId','Age','SibSp','Parch','male','Q','S',2,3]]

y_train=train['Survived']

X_test=test[['PassengerId','Age','SibSp','Parch','male','Q','S',2,3]]
from sklearn.linear_model import LogisticRegression
lrmode=LogisticRegression()
lrmode.fit(X_train,y_train)
pre=lrmode.predict(X_test)
X_test['Survived']=pre
X_test[['PassengerId','Survived']]
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.