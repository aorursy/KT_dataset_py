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
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_train.head()
data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

data_test.head()
from sklearn.ensemble import RandomForestRegressor



### 使用 RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]



    # 乘客分成已知年龄和未知年龄两部分

    known_age = np.asmatrix(age_df[age_df.Age.notnull()].values)

    unknown_age = np.asmatrix(age_df[age_df.Age.isnull()].values)



    # y即目标年龄

    y = known_age[:, 0]



    # X即特征属性值

    X = known_age[:, 1:]



    # fit到RandomForestRegressor之中

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

    

    # 用得到的模型进行未知年龄结果预测

    predictedAges = rfr.predict(unknown_age[:, 1::])

    

    # 用得到的预测结果填补原缺失数据

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df, rfr



def set_Cabin_type(df):

    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    return df

data_train, rfr = set_missing_ages(data_train)

data_train = set_Cabin_type(data_train)





data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0



tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

null_age = np.asmatrix(tmp_df[tmp_df.Age.isnull()].values)

X = null_age[:, 1:]

predictedAges = rfr.predict(X)

data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges



data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)



dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
import sklearn.preprocessing as preprocessing



scaler = preprocessing.StandardScaler()



age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)



age_scale_param = scaler.fit(test['Age'].values.reshape(-1, 1))

test['Age_scaled'] = scaler.fit_transform(test['Age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(test['Fare'].values.reshape(-1, 1))

test['Fare_scaled'] = scaler.fit_transform(test['Fare'].values.reshape(-1, 1), fare_scale_param)
df
from sklearn import linear_model

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != 'Survived'], df['Survived'])

lr = linear_model.LogisticRegression()

grid_values = {'C': [0.1,1,10], 'penalty': ['l1', 'l2']}

grid_clf = GridSearchCV(lr, param_grid=grid_values)

grid_clf.fit(X_train, y_train)

print('Grid best parameter (max. AUC): ', grid_clf.best_params_)

print('Grid best score (AUC): ', grid_clf.best_score_)

print('Test Score:', grid_clf.score(X_test, y_test))
result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':grid_clf.predict(test)})

result.to_csv("titanic_logistic_regression.csv", index=False)
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != 'Survived'], df['Survived'])

clf = SVC(kernel = 'rbf').fit(X_train, y_train)

grid_values = {'gamma': [0.01,0.1,1,100]}

grid_clf = GridSearchCV(clf, param_grid=grid_values)

grid_clf.fit(X_train, y_train)

print('Grid best parameter (max. AUC): ', grid_clf.best_params_)

print('Grid best score (AUC): ', grid_clf.best_score_)

print('Test Score:', grid_clf.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != 'Survived'], df['Survived'])

dtc = DecisionTreeClassifier().fit(X_train, y_train)

grid_values = {'max_depth': [3,5,10,50]}

grid_dtc = GridSearchCV(dtc, param_grid=grid_values)

grid_dtc.fit(X_train, y_train)

print('Grid best parameter (max. AUC): ', grid_dtc.best_params_)

print('Grid best score (AUC): ', grid_dtc.best_score_)

print('Test Score:', grid_dtc.score(X_test, y_test))
result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':grid_dtc.predict(test)})

result.to_csv("titanic_decision_trees.csv", index=False)
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != 'Survived'], df['Survived'])

rf = RandomForestClassifier().fit(X_train, y_train)

grid_values = {'max_depth': [3,5,10,50],'max_features':[1,3,5,10,15]}

grid_rf = GridSearchCV(rf, param_grid=grid_values)

grid_rf.fit(X_train, y_train)

print('Grid best parameter (max. AUC): ', grid_rf.best_params_)

print('Grid best score (AUC): ', grid_rf.best_score_)

print('Test Score:', grid_rf.score(X_test, y_test))
result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':grid_rf.predict(test)})

result.to_csv("titanic_random_forests.csv", index=False)
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != 'Survived'], df['Survived'])

grid_gnb = GaussianNB().fit(X_train, y_train)

# grid_values = {'max_depth': [3,5,10,50],'max_features':[1,3,5,10,15]}

# grid_gnb = GridSearchCV(rf, param_grid=grid_values)

# grid_gnb.fit(X_train, y_train)

# print('Grid best parameter (max. AUC): ', grid_gnb.best_params_)

# print('Grid best score (AUC): ', grid_gnb.best_score_)

print('Test Score:', grid_gnb.score(X_test, y_test))
result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':grid_gnb.predict(test)})

result.to_csv("titanic_gaussian_naive_bayes.csv", index=False)
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != 'Survived'], df['Survived'])

gbc = GradientBoostingClassifier().fit(X_train, y_train)

grid_values = {'max_depth': [3,5,10,50],'learning_rate':np.logspace(-3,3,7)}

grid_gbc = GridSearchCV(gbc, param_grid=grid_values)

grid_gbc.fit(X_train, y_train)

print('Grid best parameter (max. AUC): ', grid_gbc.best_params_)

print('Grid best score (AUC): ', grid_gbc.best_score_)

print('Test Score:', grid_gbc.score(X_test, y_test))
result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':grid_gbc.predict(test)})

result.to_csv("titanic_GBDT.csv", index=False)