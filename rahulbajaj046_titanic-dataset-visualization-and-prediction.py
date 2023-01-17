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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('../input/titanic/train.csv')
df_train.head()
df_test = pd.read_csv('../input/titanic/test.csv')
df_test
output = pd.DataFrame({'PassengerId':df_test.PassengerId})
df_train.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)
df_test.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)
df_train.isnull().sum()
df_test.isnull().sum()
df_train.Sex = df_train.Sex.map( {'female': 1, 'male': 0} ).astype(int)
df_test.Sex = df_test.Sex.map( {'female': 1, 'male': 0} ).astype(int)
df_train.Age.fillna(int(df_train.Age.mean()),inplace=True)
df_test.Age.fillna(int(df_test.Age.mean()),inplace=True)
df_train.Embarked.unique()
df_train.Embarked.fillna(df_train.Embarked.mode()[0],inplace=True)
floor_value = []
for i in df_train['Cabin']:
    
    try:
        value = list(i)[0]
        floor_value.append(value)
        
    except:
        floor_value.append(i)
        
df_train['Cabin'] = floor_value
floor_value = []
for i in df_test['Cabin']:
    
    try:
        value = list(i)[0]
        floor_value.append(value)
        
    except:
        floor_value.append(i)
        
df_test['Cabin'] = floor_value
df_train['Cabin'].unique()
df_test.Cabin.unique()
df_train.Cabin.fillna(df_train.Cabin.mode()[0],inplace=True)
df_test.Cabin.fillna(df_test.Cabin.mode()[0],inplace=True)
df_train.Cabin = df_train.Cabin.map( {'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7} ).astype(int)
df_test.Cabin = df_test.Cabin.map( {'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7} ).astype(int)
df_train.Fare = df_train.Fare.apply(lambda x : int(x))
df_train.Age = df_train.Age.apply(lambda x : int(x))

df_test.Fare.fillna(df_test.Fare.median(),inplace=True)
df_test.Fare = df_test.Fare.apply(lambda x : int(x))
df_test.Age = df_test.Age.apply(lambda x : int(x))
df_train.head()
df_train.info()
df_test.isnull().sum()
df_test
df_train
df_train.shape
x = ['Male','Female']

f, axes = plt.subplots(1,2,figsize=(12,4))
df_train['Sex'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True,labels=x,ax=axes[0])
axes[0].set_title('Total Males/Females onboard')
axes[0].set_ylabel('')
plt.bar(x,df_train.Sex.value_counts())
axes[1].set_title('Number of males and females present')
axes[1].set_ylabel('Number of passengers')

plt.show()
y = ['Dead','Survived']
df1 = pd.crosstab(df_train.Sex,df_train.Survived)

f,axes = plt.subplots(1,2,figsize=(15,4))

sns.barplot(y=df_train.Survived.value_counts(),x=y,ax=axes[0])
axes[0].set_ylabel('Number of people')
axes[0].set_title('Number of people: Dead vs Survived')

df1.plot(kind='bar',ax=axes[1])
axes[1].set_title('Sex vs Survived Plot')
axes[1].set_ylabel('Number of people')
axes[1].set_xticklabels(['Male','Female'],rotation=0)
axes[1].legend(y)

plt.show()
df1 = pd.crosstab(df_train.Embarked,df_train.Survived)
df2  = df_train[['Embarked', 'Survived']].groupby('Embarked').mean()

f, axes = plt.subplots(1,3,figsize=(18,4))
sns.barplot(y=df_train.Embarked.value_counts(),x=df_train.Embarked.unique(),ax=axes[0])
axes[0].set_xlabel('Embarked')
axes[0].set_ylabel('Number of passengers')
axes[0].set_title('Embarked status of passengers')
df1.plot(kind='bar',ax=axes[1])
axes[1].set_xticklabels(['C','Q','S'],rotation=0)
axes[1].set_ylabel('Number of passenger')
axes[1].set_title('Embarked vs Survived Plot')
axes[1].legend(['Dead','Survived'])
df2.plot(kind='bar',ax=axes[2])
axes[2].set_xticklabels(['C','Q','S'],rotation=0)
axes[2].set_ylabel('Probability')
axes[2].set_title('Survival Probability')

plt.show()
df1 = pd.crosstab(df_train.Pclass,df_train.Survived)
df2  = df_train[['Pclass', 'Survived']].groupby('Pclass').mean()

f, axes = plt.subplots(1,3,figsize=(18,4))
sns.barplot(y=df_train.Pclass.value_counts(),x=df_train.Pclass.unique(),ax=axes[0])
axes[0].set_xlabel('Passenger class')
axes[0].set_ylabel('Number of passengers')
axes[0].set_title('Passenger Class Plot')
df1.plot(kind='bar',ax=axes[1])
axes[1].set_xlabel('Passenger class')
axes[1].set_xticklabels([1,2,3],rotation=0)
axes[1].set_ylabel('Number of passenger')
axes[1].set_title('Passenger Class vs Survived Plot')
axes[1].legend(['Dead','Survived'])
df2.plot(kind='bar',ax=axes[2])
axes[2].set_xlabel('Passenger class')
axes[2].set_xticklabels([1,2,3],rotation=0)
axes[2].set_ylabel('Probability')
axes[2].set_title('Survival Probability')

plt.show()
df1 = pd.crosstab(df_train.Cabin,df_train.Survived)
df1.plot(kind='bar')
plt.xticks(rotation=0)
plt.title('Passenger Cabin vs Survied plot')
plt.ylabel('Number of passengers')
plt.legend(['Dead','Survived'])
plt.show()

g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=10)
g.axes[0,0].set_ylabel('Number of passengers')
g.fig.set_figwidth(12)
g.fig.set_figheight(3)
plt.show()
df_train["Family"] = df_train["SibSp"] + df_train["Parch"] 
df_test["Family"] = df_test["SibSp"] + df_test["Parch"]
df_train.drop(['SibSp','Parch'],axis=1,inplace=True)
df_test.drop(['SibSp','Parch'],axis=1,inplace=True)
df_train = pd.get_dummies(df_train,drop_first=True)
df_test = pd.get_dummies(df_test,drop_first=True)
df_train
X_train = df_train.iloc[:,1:]
y_train = df_train.Survived
X_test = df_test.iloc[:,:]
!pip install --upgrade pip
!pip install lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train,test_size=0.2,random_state =123)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train1, X_test1, y_train1, y_test1)
models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
random_forest = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features='sqrt')
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(acc_random_forest)
output['Survived'] = Y_prediction
output.to_csv('my_submission_rf.csv', index=False)
reg = LogisticRegression(max_iter=1000)
reg.fit(X_train, y_train)
Y_pred = reg.predict(X_test)
acc_log = round(reg.score(X_train, y_train) * 100, 2)
print(len(Y_pred))
print(acc_log)
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree
output['Survived'] = Y_pred
output.to_csv('my_submission_dt.csv', index=False)
gaussian = GaussianNB() 
gaussian.fit(X_train, y_train)  
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian
linear_svc = SVC(gamma='scale')
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
print(acc_linear_svc)
test=KNeighborsClassifier(n_neighbors=1)
test.fit(X_train,y_train)
ypred=test.predict(X_test)
acc_Kneighbour=test.score(X_train, y_train) * 100
acc_Kneighbour
import xgboost as xgb
xgb_model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
acc_XG_boost = xgb_model.score(X_train,y_train)*100
print(acc_XG_boost)
output['Survived'] = y_pred
output.to_csv('my_submission_xgb.csv', index=False)
from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier()
ada_model.fit(X_train,y_train)
y_pred = ada_model.predict(X_test)
acc_ada_boost = ada_model.score(X_train,y_train)*100
print(acc_ada_boost)
output['Survived'] = y_pred
output.to_csv('my_submission_abc.csv', index=False)
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
acc_lgbm = lgbm.score(X_train,y_train)*100
print(acc_lgbm)
output['Survived'] = y_pred
output.to_csv('my_submission_lgbm.csv', index=False)
Model=["RandomForestClassifier","DecisionTreeClassifier","KNeighborsClassifier","LogisticRegression",
       "SVM","Naive_bayes","XG Boost","Ada Boost","LGBM"]
Accuracy=[acc_random_forest,acc_decision_tree,acc_Kneighbour,acc_log,
          acc_linear_svc,acc_gaussian,acc_XG_boost,acc_ada_boost,acc_lgbm]
plt.barh(Model,Accuracy)
plt.show()
vc_model = VotingClassifier([('clf1',random_forest),('clf2',test),('clf3',xgb_model),('clf4',lgbm)],voting='soft')
vc_model.fit(X_train,y_train)
y_pred = vc_model.predict(X_test)
acc_vc = vc_model.score(X_train,y_train)*100
acc_vc
output['Survived'] = y_pred
output.to_csv('my_submission_vc.csv', index=False)
