import numpy as np

import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from warnings import filterwarnings

filterwarnings('ignore')
pwd
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
train_df=train.copy()

test_df=test.copy()
train_df.describe().T
test_df.describe().T
train_df.shape
test_df.shape
train_df.info()
test_df.info()
train_df['Sex'].value_counts()
train_df.groupby('Sex')['Survived'].mean()
sns.barplot(x="Sex", y="Survived", data=train_df)
train_df['Pclass'].value_counts()
train_df.groupby('Pclass')['Survived'].mean()
sns.barplot(x="Pclass", y="Survived", data=train_df)
fig = plt.figure(figsize=(12,6),)



ax=sns.kdeplot(train.Pclass[train.Survived == 0] , 

               color='red',

               shade=True,

               label='not survived')



ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] , 

               color='g',

               shade=True, 

               label='survived', 

              )

plt.title('Survived vs Non-Survived')

plt.ylabel("Frequency of Survived Passenger")

plt.xlabel("Class of Passenger")

## Converting xticks into words for better understanding

labels = ['Upper', 'Middle', 'Lower']

plt.xticks(sorted(train.Pclass.unique()), labels);
train_df['Embarked'].value_counts()
train_df.groupby('Embarked')['Survived'].mean()
sns.barplot(x="Embarked", y="Survived", data=train_df);
train_df['Survived'].value_counts().plot.barh().set_title('Frequency of Survived');
train_df['Ticket'].value_counts()
train_df['SibSp'].value_counts()
train_df.groupby('SibSp')['Survived'].mean()
sns.barplot(x="SibSp", y="Survived", data=train_df);
train_df['Parch'].value_counts()
train_df.groupby('Parch')['Survived'].mean()
sns.barplot(x="Parch", y="Survived", data=train_df);
train_df.isnull().sum()
test_df.isnull().sum()
100*train_df.isnull().sum()/len(train_df)
100*test_df.isnull().sum()/len(test_df)
train_df = train_df.drop(columns="Cabin")
test_df = test_df.drop(columns="Cabin")
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test_df['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train_df.head()
test_df.head()
train_df['Title'] = train_df['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')

train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')

train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
test_df['Title'] = test_df['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

test_df['Title'] = test_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')

test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')

test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
train_df['Title'].value_counts()
test_df['Title'].value_counts()
[train_df.groupby('Title')['Age'].median(),train_df.groupby('Title')['Age'].std(),train_df.groupby('Title')['Age'].mean()]
train_df['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'),inplace=True)
train_df.isnull().sum()
test_df['Age'].fillna(test_df.groupby('Title')['Age'].transform('median'),inplace=True)
test_df.isnull().sum()
train_df['Embarked'].value_counts()
train_df[train_df.Embarked.isnull()]
train_df.groupby(['Embarked','Title'])['Title'].count()
train_df["Embarked"] = train_df["Embarked"].fillna('S')
train_df.isnull().sum()
test_df[test.Fare.isnull()]
test_df[["Pclass","Fare"]].groupby("Pclass").mean()
test_df["Fare"] = test_df["Fare"].fillna(12.46)
test_df.isnull().sum()
train_df.describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
sns.boxplot(x = train_df['Fare']);
Q1=train_df['Fare'].quantile(0.25)

Q3=train_df['Fare'].quantile(0.75)

IQR=Q3-Q1

print(IQR)
lower_limit=Q1 - 1.5 * IQR

upper_limit=Q3 + 1.5 * IQR

print('lower limit: '+str(lower_limit))

print('upper limit: '+str(upper_limit))
train_df['Fare'][train_df['Fare']>upper_limit].count()
train_df['Fare'][train_df['Fare']>upper_limit].sort_values(ascending=False).head()
train_df['Fare']=train_df['Fare'].replace(512.3292,275)

test_df['Fare']=test_df['Fare'].replace(512.3292,275)
embarked_dict={'S':1,'C':2,'Q':3}
train_df['Embarked']=train_df['Embarked'].map(embarked_dict)

test_df['Embarked']=test_df['Embarked'].map(embarked_dict)
train_df.head()
train_df['Sex']=train_df['Sex'].map(lambda x:0 if x=='female' else 1).astype(int)

test_df['Sex']=test_df['Sex'].map(lambda x:0 if x=='female' else 1).astype(int)
train_df.head()
test_df.head()
train_df['Title'].unique()
test_df['Title'].unique()
# There is no Royal Category at test_df,

title_dict={'Mr':1,'Mrs':2,'Miss':3,'Master':4,'Rare':5,'Royal':5}
train_df['Title']=train_df['Title'].map(title_dict)

test_df['Title']=test_df['Title'].map(title_dict)
train_df.head()
test_df.head()
train_df=train_df.drop(['Name','Ticket','Fare'],axis=1)
train_df.head()
test_df=test_df.drop(['Name','Ticket','Fare'],axis=1)
test_df.head()
# new field family size

train_df['FamilySize']=train_df['SibSp'] + train_df['Parch'] + 1

test_df['FamilySize']=test_df['SibSp'] + test_df['Parch'] + 1
train_df.head()
train_df['is_Single']=train_df['FamilySize'].map(lambda x: 1 if x < 2 else 0)

test_df['is_Single']=test_df['FamilySize'].map(lambda x: 1 if x < 2 else 0)
train_df=pd.get_dummies(train_df,columns=['Title'],drop_first=False)

test_df=pd.get_dummies(test_df,columns=['Title'],drop_first=False)
train_df = pd.get_dummies(train_df, columns = ["Embarked"], prefix="Em")

test_df = pd.get_dummies(test_df, columns = ["Embarked"], prefix="Em")
train_df.head()
test_df.head()
train_df['Pclass'] = train_df['Pclass'].astype('category',copy=False)

train_df=pd.get_dummies(train_df,columns=['Pclass'],drop_first=False)

train_df.head()
test_df['Pclass'] = test_df['Pclass'].astype('category',copy=False)

test_df=pd.get_dummies(test_df,columns=['Pclass'],drop_first=False)

test_df.head()
train_df.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

X = train_df.drop(['Survived', 'PassengerId'], axis=1)

Y = train_df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 13)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear')

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_log_model = round(accuracy_score(y_pred, y_test) , 4)*100

print(str(acc_log_model)+str('%'))

print(confusion_matrix(y_pred, y_test))
from sklearn.svm import SVC

svm_model = SVC(kernel = "linear").fit(x_train, y_train)

y_pred = svm_model.predict(x_test)
acc_svm_model = round(accuracy_score(y_pred, y_test) , 4)*100

print(str(acc_svm_model)+str('%'))

print(confusion_matrix(y_pred, y_test))
svc_params = {"C": np.arange(1,5)}



svc = SVC(kernel = "linear")



svc_cv = GridSearchCV(svc,svc_params, 

                            cv = 10, 

                            n_jobs = -1, 

                            verbose = 2 )



svc_cv.fit(x_train, y_train)
print("Best Parameters: " + str(svc_cv.best_params_))
svc_tuned = SVC(kernel = "linear", C = 1).fit(x_train, y_train)

y_pred = svc_tuned.predict(x_test)
acc_svc_tuned = round(accuracy_score(y_pred, y_test), 4)*100

print(str(acc_svc_tuned)+str('%'))

print(confusion_matrix(y_pred, y_test))
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

cart = DecisionTreeClassifier()

cart_model = cart.fit(x_train, y_train)

y_pred = cart_model.predict(x_test)
cart_model = round(accuracy_score(y_test, y_pred),4)*100

print(str(cart_model)+str('%'))

print(confusion_matrix(y_pred, y_test))
cart_grid = {"max_depth": range(1,10),

            "min_samples_split" : list(range(2,50)) }
cart = tree.DecisionTreeClassifier()

cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)

cart_cv_model = cart_cv.fit(x_train, y_train)
print("Best Parameters: " + str(cart_cv_model.best_params_))
cart = tree.DecisionTreeClassifier(max_depth = 3, min_samples_split = 2)

cart_tuned = cart.fit(x_train, y_train)

y_pred = cart_tuned.predict(x_test)
cart_acc = round(accuracy_score(y_pred, y_test), 4)*100

print(str(cart_acc)+str('%'))

print(confusion_matrix(y_pred, y_test))
from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier()

gb.fit(X, Y)

y_pred = gb.predict(x_test)

acc_gradient = round(accuracy_score(y_pred, y_test), 4)*100

print(str(acc_gradient)+str('%'))

print(confusion_matrix(y_pred, y_test))
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [100,500,1000],

             "max_depth": [3,5,10],

             "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()



gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv.fit(x_train, y_train)
print("Best Parameters: " + str(gbm_cv.best_params_))
gbm = GradientBoostingClassifier(learning_rate = 0.001, 

                                 max_depth = 3,

                                min_samples_split = 2,

                                n_estimators = 1000)
gbm_tuned =  gbm.fit(x_train,y_train)
y_pred = gbm_tuned.predict(x_test)

gbm_acc = round(accuracy_score(y_pred, y_test), 4)*100

print(str(gbm_acc)+str('%'))

print(confusion_matrix(y_pred, y_test))
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

acc_rfc = round(accuracy_score(y_pred, y_test) , 4)*100

print(str(acc_rfc)+str('%'))

print(confusion_matrix(y_test,y_pred))
rf_params = {"max_depth": [2,5,8,10],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()



rf_cv_model = GridSearchCV(rf_model, 

                           rf_params, 

                           cv = 10, 

                           n_jobs = -1, 

                           verbose = 2) 
rf_cv_model.fit(x_train, y_train)
print("Best Parameters: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 5, 

                                  max_features = 2, 

                                  min_samples_split = 10,

                                  n_estimators = 1000)



rf_tuned.fit(x_train, y_train)
y_pred = rf_tuned.predict(x_test)

acc_rfc_tuned = round(accuracy_score(y_pred, y_test) , 4)*100

print(str(acc_rfc_tuned)+str('%'))

print(confusion_matrix(y_test,y_pred))
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier().fit(x_train, y_train)

y_pred = lgbm.predict(x_test)

acc_lgbm=round(accuracy_score(y_pred, y_test) , 4)*100

print(str(acc_lgbm)+str('%'))

print(confusion_matrix(y_test,y_pred))
lgbm = LGBMClassifier()

lgbm_params = {

        'n_estimators': [100, 500, 1000, 2000],

        'subsample': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5,6],

        'learning_rate': [0.1,0.01,0.02,0.05],

        "min_child_samples": [5,10,20]}

lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 

                             cv = 10, 

                             n_jobs = -1, 

                             verbose = 2)

lgbm_cv_model.fit(x_train, y_train)
print("Best Parameters: " + str(lgbm_cv_model.best_params_))
lgbm = LGBMClassifier(learning_rate = 0.02, 

                       max_depth = 3,

                       subsample = 0.6,

                       n_estimators = 100,

                       min_child_samples = 10)
lgbm_tuned = lgbm.fit(x_train,y_train)
y_pred = lgbm_tuned.predict(x_test)

acc_lgbm_tuned=round(accuracy_score(y_pred, y_test) , 4)*100

print(str(acc_lgbm_tuned)+str('%'))

print(confusion_matrix(y_test,y_pred))
from xgboost import XGBClassifier
xgb_model = XGBClassifier().fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)

acc_xgb=round(accuracy_score(y_test, y_pred),4)*100

print(str(acc_xgb)+str('%'))

print(confusion_matrix(y_test,y_pred))
xgb_params = {

        'n_estimators': [100, 500, 1000, 2000],

        'subsample': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5,6],

        'learning_rate': [0.1,0.01,0.02,0.05],

        "min_samples_split": [2,5,10]}
xgb = XGBClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(x_train, y_train)
print("Best Parameters: " + str(xgb_cv_model.best_params_))
xgb = XGBClassifier(learning_rate = 0.02, 

                    max_depth = 3,

                    min_samples_split = 2,

                    n_estimators = 100,

                    subsample = 0.6)
xgb_tuned =  xgb.fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test)

acc_xgb=round(accuracy_score(y_test, y_pred),4)*100

print(str(acc_xgb)+str('%'))

print(confusion_matrix(y_test,y_pred))
#set ids as PassengerId and predict survival 

ids = test_df['PassengerId']

ypred = lgbm_tuned.predict(test_df.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': ypred })

output.to_csv('submission.csv', index=False)
output.head(7)