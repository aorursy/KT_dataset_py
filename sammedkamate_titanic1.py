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
import numpy as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()

train.dtypes
all = pd.concat([train, test], sort = False)
all.isnull().sum()

all['Parch'].unique()
sns.countplot(x='Survived',hue='Sex',data=train)
train['Age'].plot.hist()
train['Fare'].plot.hist()
sns.countplot(x='Survived',hue='Parch',data=train)
all.shape
all['Age'].groupby(all['Pclass']).mean()
def fillAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 39
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age
all['Age'] = all[['Age','Pclass']].apply(fillAge,axis=1)
all.dropna(subset=['Embarked'],inplace=True)
all.shape

all=pd.get_dummies(all,columns=['Pclass','Sex','Embarked'],drop_first=True)
all.head()
all.drop(['Name','Ticket','Cabin','Fare'],axis=1,inplace=True)
all.head()
all.isnull().sum()
all_train = all[all['Survived'].notna()]
all_test = all[all['Survived'].isna()]
y=all_train['Survived']
x=all_train.drop(['Survived','PassengerId'],axis=1)
x.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

xtrain,xtest,ytrain,ytest=train_test_split(x,y, test_size=0.3, random_state=1)

model=LogisticRegression(solver='lbfgs',max_iter=1000,random_state=101)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
accuracy_score(ytest,ypred)
from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,ypred)
ypred
all_test.head()
test_final=all_test.drop(['PassengerId','Survived'],axis=1)
test_pred=model.predict(test_final).astype(int)
PassengerId=all_test['PassengerId']
final=pd.DataFrame({'PassengerId':PassengerId,'Survived':test_pred})
final.head()
final.to_csv("submission.csv", index = False)
