# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

titanic = pd.read_csv('../input/titanic_train.csv')
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('darkgrid')

sns.countplot(x='Survived',data=titanic,palette='RdBu_r')
sns.countplot(x='Survived',data=titanic,hue='Sex',palette='RdBu_r')
sns.countplot(x='Survived',data=titanic,hue='Pclass',palette='rainbow')
sns.distplot(titanic['Age'].dropna(),kde=False,bins=30,color='darkred')
sns.countplot(x='SibSp',data=titanic)
titanic['Fare'].hist(bins=40,figsize=(8,4))
plt.figure(figsize=(12,8))

sns.boxplot(x='Pclass',y='Age',data=titanic)
def impute_age(cols):

    age=cols[0]

    pClass=cols[1]

    if(pd.isnull(age)):

        if pClass==1:

            return 37

        elif pClass == 2:

            return 29

        else:

            return 24

    else:

        return age
titanic['Age']=titanic[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
titanic.drop('Cabin',axis=1,inplace=True)
titanic.head()
sex=pd.get_dummies(titanic['Sex'],drop_first=True)

embark=pd.get_dummies(titanic['Embarked'],drop_first=True)

titanic.head()
titanic.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
titanic=pd.concat([titanic,sex,embark],axis=1)

titanic.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(titanic.drop('Survived',axis=1),titanic['Survived'],

                                                 test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
lgmodel=LogisticRegression()
lgmodel.fit(x_train,y_train)
predictions=lgmodel.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))
sns.countplot(x=predictions)