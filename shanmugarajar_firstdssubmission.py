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
df1= pd.read_csv("/kaggle/input/titanic/train.csv")

df1.head()
df2 = pd.read_csv("/kaggle/input/titanic/test.csv")

df2.head()
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from xgboost import XGBClassifier

import statsmodels.api as sms
df1.describe().transpose()
sns.heatmap(df1.notnull(),cbar=False,yticklabels=False)
Train_Age_avail=df1['Age'].count()/len(df1['Age'])

Train_Cabin_avail=df1['Cabin'].count()/len(df1['Cabin'])

print('Percentage of available values of Age and Cabin in Train ', Train_Age_avail, Train_Cabin_avail)

Test_Age_avail=df2['Age'].count()/len(df2['Age'])

Test_Cabin_avail=df2['Cabin'].count()/len(df2['Cabin'])

print('Percentage of available values of Age and Cabin in Train ', Test_Age_avail, Test_Cabin_avail)
sns.heatmap(df1.corr(), annot=True)
df1['Age'].hist(bins=10)
col_list=list(df1.columns)

col_list.remove('Survived')

col_list.remove('PassengerId')

col_list.remove('Name')

col_list.remove('Cabin')

for i in col_list:

    plt.figure()

    sns.countplot(x=df1[i], hue=df1['Survived'])
#Fill missing values in Age with median of the group

df1['Age'].fillna(df1['Age'].median(),inplace=True)

df2['Age'].fillna(df2['Age'].median(),inplace=True)
#Box the ages in groups

cat_age=pd.cut(df1['Age'],bins=[0,20,30,40,60,80], labels=[1,2,3,4,5])

df1.insert(6,'AgeBox',cat_age)

cat_age=pd.cut(df2['Age'],bins=[0,20,30,40,60,80], labels=[1,2,3,4,5])

df2.insert(6,'AgeBox',cat_age)
df1.groupby(['Pclass','Sex','Survived'])['Name'].count()
df1['Dependents'] = df1['SibSp']+df1['Parch']

df2['Dependents'] = df2['SibSp']+df2['Parch']

df1.drop(['SibSp','Parch'], axis=1,inplace=True)
sns.countplot(df1['Dependents'],hue=df1['Survived'])
#Convert Embarked by Label encoding

from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()

df1['Embarked_Code']=lbe.fit_transform(df1['Embarked'].astype(str))

df2['Embarked_Code']=lbe.fit_transform(df2['Embarked'].astype(str))

#Convert Sex into categories

gender={"male":1,"female":2}

df1['Sex']=[gender[x] for x in df1['Sex']]

df2['Sex']=[gender[x] for x in df2['Sex']]
sns.boxplot(df1['Fare'])
print(df1[df1['Fare']>100]['Pclass'].value_counts())

print(df2[df2['Fare']>100]['Pclass'].value_counts())
df1.loc[df1['Fare']>100,'Fare']=df1[df1['Pclass']==1]['Fare'].median()

df2.loc[df2['Fare']>100,'Fare']=df2[df2['Pclass']==1]['Fare'].median()
sns.heatmap(df1.corr(),annot=True)
df1.drop(['Name','Age','Ticket','Cabin','Embarked','Fare'], axis=1,inplace=True)

df2.drop(['Name','Age','Ticket','Cabin','Embarked','Fare'], axis=1,inplace=True)
#Re-arranging the columns

df1=df1[['PassengerId','Pclass','Sex','AgeBox','Dependents','Embarked_Code','Survived']]

df2=df2[['PassengerId','Pclass','Sex','AgeBox','Dependents','Embarked_Code']]

y=df1.iloc[:,-1]

X=df1.iloc[:,:-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#Logical Regression and check P values

logReg=LogisticRegression()

logReg.fit(X_train,y_train)

y1_pred=logReg.predict(X_test)

print("Accuracy Score for Logical Regression is ",accuracy_score(y1_pred,y_test))

print("Confusion Matrix for Logical Regression is ",confusion_matrix(y1_pred,y_test))
logres=sms.Logit(y_train,X_train.astype(float))

res=logres.fit()

print(res.summary())
rfcls= RandomForestClassifier(n_estimators=50,criterion='gini',min_samples_leaf=3,max_depth=20,random_state=42)

rfcls.fit(X_train, y_train)

y2_pred=rfcls.predict(X_test)

print("Accuracy Score for Random Forest Classifier is ",accuracy_score(y2_pred,y_test))

print("Confusion Matrix for Random Forest Classifier is ",confusion_matrix(y2_pred,y_test))
dtree= DecisionTreeClassifier(random_state=42)

dtree.fit(X_train, y_train)

y3_pred=rfcls.predict(X_test)

print("Accuracy Score for Decision Tree Classifier is ",accuracy_score(y3_pred,y_test))

print("Confusion Matrix for Decision Tree Classifier is ",confusion_matrix(y3_pred,y_test))
grdbst=GradientBoostingClassifier(n_estimators=50,min_samples_leaf=3,max_depth=20,random_state=42)

grb_mod=grdbst.fit(X_train,y_train)

y6_pred=grb_mod.predict(X_test)

print("Accuracy Score for Gradient Boost Classifier is ",accuracy_score(y6_pred,y_test))

print("Confusion Matrix for Gradient Boost Classifier is ",confusion_matrix(y6_pred,y_test))
y_test_pred=rfcls.predict(df2)

print(y_test_pred)
output=pd.DataFrame({'PassengerId':df2['PassengerId'],'Survived':y_test_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")