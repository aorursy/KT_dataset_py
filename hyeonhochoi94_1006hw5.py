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
from sklearn.model_selection import train_test_split, KFold, cross_val_score #training set과 test set 을 나누는 방법

from sklearn.metrics import accuracy_score # 정확도를 계산하는 방법

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn import tree

from sklearn.linear_model import LinearRegression

import xgboost

import matplotlib.pyplot as plt

from matplotlib import pyplot

import seaborn as sns
train_data= pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.info()
sns.heatmap((train_data.loc[:,['Survived','Age', 'SibSp', 'Parch', 'Fare', 'Pclass']]).corr(), annot=True)
X_Survived = train_data.Survived.value_counts()

labels_X = ['Survived','Died']

plt.subplot(1,1,1)

plt.rcParams['figure.figsize']=[5,5]

plt.pie(X_Survived, labels = labels_X,autopct = '%.2f%%')

plt.show()
def pie(feature):

    Number_total = train_data[feature].value_counts(sort=False)

    feature_index = Number_total.index

    feature_size  = Number_total.size

    Number_died = train_data[train_data['Survived']==0][feature].value_counts()

    Number_survived = Number_total - Number_died



    for i, j in enumerate(feature_index):

        plt.subplot(1,feature_size+1,i+1)

        plt.title(feature+' / '+str(j))

        plt.rcParams['figure.figsize']=[10,10]

        ratio_survived = [Number_survived[j],Number_died[j]]

        plt.pie(ratio_survived,labels=['Survived','Died'], autopct = '%.1f%%')

    

    plt.show()

pie('Pclass')
pie('Sex')
pie('SibSp')
pie('Parch')
pie('Embarked')
f,ax=plt.subplots(figsize=(10,5))

Number_total = train_data['Age'].value_counts(sort=False)



a = sns.kdeplot(train_data[train_data['Survived']==0]['Age'],color = "red",shade=True)

a = sns.kdeplot(train_data[train_data['Survived']==1]['Age'],color = "blue",shade= True)

a.set_xlabel('Age')

a.set_ylabel('Frequency')

a= a.legend(['died', 'Survived'])

f,ax=plt.subplots(figsize=(10,5))

Number_total = train_data['Fare'].value_counts(sort=False)



a = sns.kdeplot(train_data[train_data['Survived']==0]['Fare'],color = "red",shade=True)

a = sns.kdeplot(train_data[train_data['Survived']==1]['Fare'],color = "blue",shade= True)

a.set_xlabel('Fare')

a.set_ylabel('Frequency')

a= a.legend(['died', 'Survived'])

print(train_data.info())
train_data.Age = train_data.Age.fillna(train_data.Age.mean())
test_data.Age = test_data.Age.fillna(train_data.Age.mean())

test_data.Fare = test_data.Fare.fillna(train_data.Age.mean())

train_data.Embarked = train_data.Embarked.fillna(method='ffill')
print(test_data.info())
features =['Pclass','Sex','Age','Fare','Embarked']

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

y = train_data.Survived
X_trainer, X_tester, y_trainer, y_tester = train_test_split(X,y, test_size = 0.2) #test set 만들기

sum_r=0

sum_d=0

sum_s=0

sum_x=0

for i in range (15):

    rfc = RandomForestClassifier(n_estimators = 50,max_depth=6, random_state=0)

    dtc = tree.DecisionTreeClassifier()

    svc = SVC()

    xgb = xgboost.XGBClassifier()

    rfc.fit(X_trainer,y_trainer)

    dtc.fit(X_trainer,y_trainer)

    svc.fit(X_trainer,y_trainer)

    xgb.fit(X_trainer,y_trainer)

    rfc_pred = rfc.predict(X_tester)

    dtc_pred = dtc.predict(X_tester)

    svc_pred = svc.predict(X_tester)

    xgb_pred = xgb.predict(X_tester)

    sum_r += accuracy_score(y_tester,rfc_pred)

    sum_d += accuracy_score(y_tester,dtc_pred)

    sum_s += accuracy_score(y_tester,svc_pred)

    sum_x += accuracy_score(y_tester,xgb_pred)

print(sum_r/15,sum_d/15,sum_s/15,sum_x/15)
Final_rfc_pred = rfc.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Final_rfc_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")