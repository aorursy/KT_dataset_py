import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBRFClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

test1 = test
train.isnull().sum()
print(train.info()) 

print('*'*50) 

print(train.isnull().sum())

print('*'*50) 

print(test.isnull().sum())

print("$"*50)

test.info()

test.corr()
train.describe()

train=train.drop(['Cabin','PassengerId','Ticket'], axis=1)

test=test.drop(['Cabin','PassengerId','Ticket'], axis=1)

test
print(train.isnull().sum())

print("*"*60)

print(test.isnull().sum())

print(train.columns)

print(test.columns)
train['Title']=train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title']=test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print(train.columns)

print(test.columns)

train['Embarked']= train['Embarked'].fillna(train['Embarked'].dropna().mode()[0])
train["Title"] = train["Title"].replace(['Don', 'Rev', 'Dr',

       'Major', 'Sir', 'Col', 'Capt', 'Countess','Lady',

       'Jonkheer'],'Rare')

train["Title"] = train["Title"].replace(["Mme",'Mlle','Ms'],'Miss')
test["Title"] = test["Title"].replace(['Col' ,'Rev', 'Dr', 'Dona'],'Rare')
train['Age'] = train.groupby(['Pclass','Sex'])['Age'].apply(lambda x: x.fillna(x.median()))

test['Age'] = test.groupby(['Pclass','Sex'])['Age'].apply(lambda x: x.fillna(x.median()))

test['Fare'] = test.groupby('Pclass')['Age'].apply(lambda x:x.fillna(x.median))
crr= train.corr()

print(crr)

sns.heatmap(crr,annot=True)

print(train.info())

print(test.info())
crrt = test.corr()

sns.heatmap(crrt,annot=True , annot_kws={'fontsize':9})
print(train.info()) 

print('*'*50) 

print(test.isnull().sum())

print("$"*50)

test.info()

test.corr()
def see_corr(x):

    if type(x) is not list:

        x=[x]

#     x=x.tolist()

    for col in x:

        print("*"*60)

        print(train[[col,"Survived"]].groupby([col],as_index=False).agg(['count','mean'])

              .sort_values(by=('Survived','mean'),ascending=False))

        print(train[[col,'Survived']].corr())
le = LabelEncoder()

full_data=[train,test]

for ds in full_data:

    ds['Sex'] = le.fit_transform(ds['Sex'])

    ds['camp'] = ds["SibSp"]+ds["Parch"]+1

    ds['camp'] = pd.cut(ds['camp'],5)

    ds['camp'] = le.fit_transform(ds['camp'])

#     ds['isAlone'] = (ds["SibSp"]+ds["Parch"]+1).apply(lambda x: 1 if x ==1 else 0)

    ds['FareBin']=pd.qcut(ds['Fare'],5)

    ds['AgeBin'] = pd.cut(ds['Age'],5)

    ds['FareCode']= le.fit_transform(ds["FareBin"])

    ds['AgeCode']=le.fit_transform(ds["AgeBin"])

    ds['TitleCode']=le.fit_transform(ds["Title"])

    ds['EmbarkedCode']=le.fit_transform(ds["Embarked"])

    print(ds)
print(train.columns)

print(test.columns)

train["AgeBin"].value_counts()
# lot of missing values in cabin so we are going to drop cabin and also passenger id

train_df=train.drop(['Name', 'Age', 'SibSp', 'Parch', 'Fare',

       'Embarked', 'Title', 'FareBin', 'AgeBin'],axis=1)

test_df = test.drop(['Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',

       'Title', 'FareBin', 'AgeBin'],axis=1)
sns.heatmap(train_df.corr() , annot= True)
test_df
trainX = train_df.drop(['Survived'], axis=1)

trainY = train_df['Survived']

trainY
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2, random_state=0)

X_train, X_test, y_train, y_test
logreg = LogisticRegression()

lr_model = logreg.fit(X_train,y_train)

y_pred = lr_model.predict(X_test)

acc_lr = round(accuracy_score(y_pred,y_test)*100,2)

print(acc_lr)

print(classification_report(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
gNB = GaussianNB()

g_model = gNB.fit(X_train,y_train)

y_pred = g_model.predict(X_test)

acc_gnb = round(accuracy_score(y_pred,y_test)*100,2)

print(acc_gnb)

print(classification_report(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

acc_svm = round(accuracy_score(y_pred,y_test)*100,2)

print(acc_svm)

print(classification_report(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
lsvc = LinearSVC()

y_pred = lsvc.fit(X_train,y_train).predict(X_test)

acc_lsvm = round(accuracy_score(y_pred,y_test)*100,2)

print(acc_lsvm)

print(classification_report(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
decisonT = DecisionTreeClassifier()

y_pred = decisonT.fit(X_train,y_train).predict(X_test)

acc_dt = round(accuracy_score(y_pred,y_test)*100,2)

print(acc_dt)

print(classification_report(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
rfc = RandomForestClassifier(n_estimators=1000)

y_pred = rfc.fit(X_train,y_train).predict(X_test)

acc_rfc= round(accuracy_score(y_pred,y_test)*100,3)

print(acc_rfc)

print(classification_report(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred= knn.predict(X_test)

acc_knn= round(accuracy_score(y_pred,y_test)*100,3)

print(acc_knn)

print(classification_report(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))
xgb = XGBRFClassifier(n_estimators=1000, max_depth=5, random_state=1, learning_rate=0.5)

xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)

acc_xgb = round(accuracy_score(y_pred,y_test)*100,2)

print(acc_xgb)
Y_pred = xgb.predict(test_df)
submission = pd.DataFrame({

        "PassengerId": test1["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submissions.csv', header=True, index=False)