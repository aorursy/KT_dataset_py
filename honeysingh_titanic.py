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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic = pd.read_csv('../input/train.csv')
#display the top 5 rows
titanic.head()
#Display colomn details
titanic.info()
#Age,Cabin and embarked colomns are not properly filled
titanic.isnull().sum()
titanic.columns
titanic[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked']].hist()
plt.show()
plt.figure(figsize=(15,10))
titanic_draw = titanic[[ 'Pclass',  'Sex', 'Age', 'SibSp',
       'Parch',  'Fare', 'Embarked']]
p =1
for i in titanic_draw.columns:
    plt.subplot(3,3,p)
    sns.violinplot(x='Survived',y=i,data=titanic)
    p =p+1

sns.countplot(x='Survived',data=titanic)
sns.pairplot(data=titanic.drop(['PassengerId'],axis=1),hue='Survived',diag_kind='kde')
plt.show()
sns.countplot(x='Sex',data=titanic)
def men_women_kid(passenger):
    age,sex = passenger
    
    if(age>16):
        if(sex == 'male'):
            return 'man'
        else:
            return 'woman'
    else:
        if(sex == 'male'):
            return 'boy'
        else:
            return 'girl'
    
titanic['person'] = titanic[['Age','Sex']].apply(men_women_kid,axis=1)
sns.countplot('Survived',data=titanic,hue='person')
#Distribution of age based on Person
fig = sns.FacetGrid(titanic,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
fig.set(xlim=(0,100))
fig.add_legend()
#show the distribution of class in cities
sns.factorplot('Embarked',data=titanic,hue='Pclass')
titanic['Alone'] = titanic['Parch']+titanic['SibSp']
titanic['Fare'].fillna(titanic['Fare'].dropna().median(), inplace=True)
titanic['Alone'].loc[titanic['Alone']>0] = 'Family'
titanic['Alone'].loc[titanic['Alone']==0] = 'Non-Family'
titanic.describe(include=['O'])
sns.factorplot('Pclass','Survived',hue='person',data=titanic)
sns.lmplot('Age','Survived',hue='Pclass',data=titanic)
sns.factorplot('Alone','Survived',data=titanic)
titanic.columns
X = titanic[[ 'Pclass', 'Sex', 'person', 'Alone']]
Y = titanic['Survived']
clean_up = {"person":{"woman":0,"girl":1,"boy":2,"man":3},"Alone":{"Family":1,"Non-Family":0},"Sex":{"male":0,"female":1}}
X.replace(clean_up,inplace=True)
X = pd.concat([X,pd.get_dummies(X['person'], prefix='person')],axis=1)
X.drop(['person'],axis =1,inplace=True)

X.drop(['person_3'],axis=1,inplace=True)
X.head()
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.25)

abc=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    abc.append(metrics.accuracy_score(prediction,test_Y))
models_dataframe=pd.DataFrame(abc,index=classifiers)   
models_dataframe.columns=['Accuracy']
models_dataframe
titanic[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic[['person', 'Survived']].groupby(['person'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#cannot find a proper trend
titanic[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#cannot find a proper trend
titanic[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#cannot find a proper trend
titanic[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#cannot find a proper trend
titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

X= X.append(titanic['Fare'],axis=1)
X.head()
