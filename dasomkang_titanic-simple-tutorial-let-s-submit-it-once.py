import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#visualization - which is not used in this kernel
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#configure Visualization Defaults - which is not used in this kernel
#%matplotlib inline  - show plots in Jupyter Notebook browser
%matplotlib inline 
plt.style.use('seaborn')
sns.set(font_scale=2) 


#packages for modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

import missingno as msno

import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
#add new feature - FamilySize which will replace SibSp and Parch features
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 
#change NAN values to median of fare (or mean, anything you want)
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].median() 

#log transformation
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
#check whether 'Embarked' has Null values or not.
print('Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')
#fill in Null values with S
df_train['Embarked'].fillna('S', inplace=True)
#add new feature- Initial
df_train["Title"]=0
for i in df_train:
    df_train['Title']=df_train.Name.str.extract('([A-Za-z]+)\.') 

df_test["Title"]=0
for i in df_test:
    df_test['Title']=df_test.Name.str.extract('([A-Za-z]+)\.') 
mapping = {
    'Mlle': 'Miss',
    'Ms': 'Miss', 
    'Dona': 'Mrs',
    'Mme': 'Miss',
    'Lady': 'Mrs', 
    'Capt': 'Honorable', 
    'Countess': 'Honorable', 
    'Major': 'Honorable', 
    'Col': 'Honorable', 
    'Sir': 'Honorable', 
    'Don': 'Honorable',
    'Jonkheer': 'Honorable', 
    'Rev': 'Honorable',
    'Dr': 'Honorable'
}

df_train['Initial'] = df_train.Title.replace(mapping)
df_test['Initial'] = df_test.Title.replace(mapping)

#df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
#                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr', 'Mr'],inplace=True)

#df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
#                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.head(n=5)
df_all = pd.concat([df_train, df_test])
df_all.groupby('Initial').mean()
# fill in null values with mean of each Initial's age
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'),'Age']=33
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mrs'),'Age']=37
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'),'Age']=5
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'),'Age']=22
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Honorable'),'Age']=45

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age']=33
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'),'Age']=37
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'),'Age']=5
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'),'Age']=22
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Honorable'),'Age']=45
# Age
df_train['Age_cat'] = 0
df_train.loc[df_train['Age'] < 10, 'Age_cat'] = 0
df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1
df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2
df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3
df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4
df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5
df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6
df_train.loc[70 <= df_train['Age'], 'Age_cat'] = 7

df_test['Age_cat'] = 0
df_test.loc[df_test['Age'] < 10, 'Age_cat'] = 0
df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1
df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2
df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3
df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4
df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5
df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6
df_test.loc[70 <= df_test['Age'], 'Age_cat'] = 7
# drop the original Age feature

df_train.drop(['Age'], axis=1, inplace=True)
df_test.drop(['Age'], axis=1, inplace=True)
# Initial
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3 , 'Honorable' : 4})
df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3 , 'Honorable' : 4})

# Embarked
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Sex
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
# Initial
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')
df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')

#Embarked
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.head(n=10)
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title'], axis=1, inplace=True)
df_train.info()
X_train = df_train.drop('Survived', axis=1).values
target_label = df_train['Survived'].values
X_test = df_test.values
#split taining data into training and valid dataset 
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
#size of data
print("X_train Shape: {}".format(X_train.shape))
print("target_label Shape: {}".format(target_label.shape))
print("X_tr Shape: {}".format(X_tr.shape))
print("X_vld Shape: {}".format(X_vld.shape))
print("y_tr Shape: {}".format(y_tr.shape))
print("y_vld Shape: {}".format(y_vld.shape))
# with k-fold cv ver.
kfold = StratifiedKFold(n_splits=10)
RFC = RandomForestClassifier(random_state=2, n_estimators=300, oob_score=True, n_jobs=5).fit(X_tr, y_tr)
print(RFC.oob_score_)

cv = cross_val_score(RFC, X_tr, y_tr, scoring='accuracy', cv=kfold, n_jobs=4, verbose=1)
print(cv.mean(), cv.std())
prediction = RFC.predict(X_vld)
# without k-fold cv ver.
model = RandomForestClassifier(random_state=2, n_estimators=500, oob_score=True, n_jobs=5).fit(X_tr, y_tr)
print(model.oob_score_)
prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
submission = pd.read_csv('../input/gender_submission.csv')
submission.head(n=10)
submission.info()
prediction = model.predict(X_test)
submission['Survived'] = prediction
submission.head()
submission.to_csv('fisrt_submission.csv', index=False)
