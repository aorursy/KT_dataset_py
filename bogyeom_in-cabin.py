import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics # 모델의 평가를 위해서 씁니다

from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수입니다.

plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno

import xgboost as xgb

#ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msg)
#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(df_test["Fare"])):

    if pd.isnull(df_test["Fare"][x]):

        pclass = df_test["Pclass"][x] #Pclass = 3

        df_test["Fare"][x] = round(df_train[df_train["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

df_train['FareBand'] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4])

df_test['FareBand'] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

df_train = df_train.drop(['Fare'], axis = 1)

df_test = df_test.drop(['Fare'], axis = 1)
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

    

df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'),'Age'] = 33

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mrs'),'Age'] = 36

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'),'Age'] = 5

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'),'Age'] = 22

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Other'),'Age'] = 46



df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age'] = 33

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'),'Age'] = 36

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'),'Age'] = 5

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'),'Age'] = 22

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Other'),'Age'] = 46




df_train.Cabin.fillna('0', inplace=True)

df_train.loc[df_train.Cabin.str[0] == 'A', 'Cabin'] = 1

df_train.loc[df_train.Cabin.str[0] == 'B', 'Cabin'] = 2

df_train.loc[df_train.Cabin.str[0] == 'C', 'Cabin'] = 3

df_train.loc[df_train.Cabin.str[0] == 'D', 'Cabin'] = 4

df_train.loc[df_train.Cabin.str[0] == 'E', 'Cabin'] = 5

df_train.loc[df_train.Cabin.str[0] == 'F', 'Cabin'] = 6

df_train.loc[df_train.Cabin.str[0] == 'G', 'Cabin'] = 7

df_train.loc[df_train.Cabin.str[0] == 'T', 'Cabin'] = 8



df_test.Cabin.fillna('0', inplace=True)

df_test.loc[df_test.Cabin.str[0] == 'A', 'Cabin'] = 1

df_test.loc[df_test.Cabin.str[0] == 'B', 'Cabin'] = 2

df_test.loc[df_test.Cabin.str[0] == 'C', 'Cabin'] = 3

df_test.loc[df_test.Cabin.str[0] == 'D', 'Cabin'] = 4

df_test.loc[df_test.Cabin.str[0] == 'E', 'Cabin'] = 5

df_test.loc[df_test.Cabin.str[0] == 'F', 'Cabin'] = 6

df_test.loc[df_test.Cabin.str[0] == 'G', 'Cabin'] = 7

df_test.loc[df_test.Cabin.str[0] == 'T', 'Cabin'] = 8

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
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket','Age'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket','Age'], axis=1, inplace=True)
X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
# gaussian

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(X_tr, y_tr)

prediction = gaussian.predict(X_vld)

acc_gaussian = round(accuracy_score(prediction,y_vld) * 100, 2)

print(acc_gaussian)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_tr, y_tr)

prediction2 = logreg.predict(X_vld)

acc_logreg = round(accuracy_score(prediction2,y_vld) * 100, 2)

print(acc_logreg)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(X_tr, y_tr)

prediction3 = svc.predict(X_vld)

acc_svc = round(accuracy_score(prediction3,y_vld) * 100, 2)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_tr, y_tr)

prediction4 = svc.predict(X_vld)

acc_linear_svc = round(accuracy_score(prediction4,y_vld) * 100, 2)
 #Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(X_tr, y_tr)

prediction5 = perceptron.predict(X_vld)

acc_perceptron = round(accuracy_score(prediction5,y_vld) * 100, 2)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

decisiontree = DecisionTreeClassifier(criterion='gini', max_depth=5)

decisiontree.fit(X_tr, y_tr)

prediction6 = decisiontree.predict(X_vld)

acc_decisiontree = round(accuracy_score(prediction6,y_vld) * 100, 2)

#importing all the required ML packages

from sklearn.ensemble import RandomForestClassifier # 유명한 randomforestclassfier 입니다. 



rf = RandomForestClassifier()

rf.fit(X_tr, y_tr)

prediction7 = rf.predict(X_vld)

acc_rf = round(accuracy_score(prediction7,y_vld) * 100, 2)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_tr, y_tr)

prediction8 = knn.predict(X_vld)

acc_knn = round(accuracy_score(prediction8,y_vld) * 100, 2)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(X_tr, y_tr)

prediction9 = knn.predict(X_vld)

acc_sgd = round(accuracy_score(prediction9,y_vld) * 100, 2)

# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(X_tr,y_tr)

prediction10 = gbk.predict(X_vld)

acc_gbk = round(accuracy_score(prediction10,y_vld) * 100, 2)
xg_boost = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.65, gamma=2, learning_rate=0.3, max_delta_step=1,

       max_depth=4, min_child_weight=2, missing=None, n_estimators=280,

       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1)

xg_boost.fit(X_tr, y_tr)

prediction11 = gbk.predict(X_vld)

acc_xg_boost = round(accuracy_score(prediction11,y_vld) * 100, 2)
# xg_boost.fit(X_tr, y_tr)

# # Y_pred = xg_boost.predict(X_test)

# print(xg_boost.score(X_tr, y_tr))



# scores = model_selection.cross_val_score(xg_boost, X_tr, y_tr, cv=5, scoring='accuracy')

# print(scores)

# print("Kfold on XGBClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier', 'XGboost'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_rf, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk, acc_xg_boost]})

models.sort_values(by='Score', ascending=False)
submission = pd.read_csv('../input/sample_submission.csv')
prediction = linear_svc.predict(X_test)

submission['Survived'] = prediction
submission.to_csv('./submission7.csv', index=False)