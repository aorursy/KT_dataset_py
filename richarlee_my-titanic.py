#-*- coding: utf-8 -*-

%matplotlib inline

import csv as csv

from scipy import stats, integrate

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn import linear_model

from sklearn import cross_validation

from sklearn.ensemble import BaggingRegressor

import sklearn.preprocessing as pp

from sklearn.grid_search import GridSearchCV



import matplotlib.pyplot as plt

import seaborn as sns

import time



import warnings

warnings.filterwarnings('ignore')
data_train = pd.read_csv('../input/train.csv')#dataframe

data_train.isnull().any()

data_train.Embarked.value_counts()
fig, axes = plt.subplots(1,3,figsize=(20,5))



sns.countplot(x='Survived',data=data_train,ax=axes[0])

sns.countplot(x='Embarked',hue='Survived',data=data_train,ax=axes[1])

sns.countplot(x='Pclass',hue='Survived',data=data_train,ax=axes[2])



facet = sns.FacetGrid(data_train,col="Survived",hue='Pclass',aspect=2)

facet.map(sns.distplot,'Age',hist=False)

facet.set(xlim=(0, data_train['Age'].max()))

data_train.isnull().any()
def dummy_data(data):

#     dummies_Embarked = pd.get_dummies(data['Embarked'],prefix='Embarked')

    dummies_Sex = pd.get_dummies(data['Sex'],prefix='Sex')

    dummies_Pclass = pd.get_dummies(data['Pclass'],prefix='Pclass')



    df = pd.concat([data,dummies_Sex,dummies_Pclass],axis=1)

    df.drop(['Embarked','Pclass','Sex'], axis=1, inplace=True)

    return df



df = data_train.drop(['Ticket','Cabin','Name'],axis=1)

df = dummy_data(df)

df


#预测丢失年龄

def set_missing_ages(df):

    

    age_df = df[['Age','Pclass_1','Pclass_2','Pclass_3','SibSp','Parch','Fare','Sex_female','Sex_male']]

    

    known_age = age_df[age_df.Age.notnull()].as_matrix()

    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    

    y = known_age[:,0]

    

    X = known_age[:,1:]

    

    rfr = RFR(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

    if len(unknown_age) > 0:

        predictedAges = rfr.predict(unknown_age[:,1:])

        df.loc[(df.Age.isnull()),'Age'] = predictedAges

    

    return df, rfr



df, rfr = set_missing_ages(df)



df
#构建票价预测模型



fare_df = df.filter(regex='Fare|SibSp|Parch|Sex_.*|Pclass_.*')

fare_data = fare_df.Fare

fare_df.drop(['Fare'],axis=1,inplace=True)

fare_df.insert(0,'Fare',fare_data)

fare_df = fare_df.dropna().as_matrix()



y = fare_df[:,0]



X = fare_df[:,1:]



fare_rfr = RFR(random_state=0, n_estimators=2000, n_jobs=-1)

fare_rfr.fit(X, y)
scaler = pp.StandardScaler()

age_scale_param = scaler.fit(df.Age.reshape(-1,1))

df['Age_Scaled'] = scaler.fit_transform(df.Age.reshape(-1,1), age_scale_param)



fare_scale_param = scaler.fit(df.Fare.reshape(-1,1))

df['Fare_Scaled'] = scaler.fit_transform(df.Fare.reshape(-1,1), fare_scale_param)

df.drop(['Age', 'Fare'], axis=1, inplace=True)

df.isnull().any()
train_np = df.as_matrix()

df.isnull().any()



y = train_np[:,1]



X = train_np[:,2:]



lr_clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

# clf.fit(X,y)

# clf



def pretty_print_linear(coefs, names = None, sort = False):

    if names == None:

        names = ["X%s" % x for x in range(len(coefs))]

        print(names)

    lst = zip(coefs, names)

    if sort:

        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))

    return " + ".join("%s * %s" % (np.round(coef, 3), name) for coef, name in lst)



# pretty_print_linear(clf.coef_[0],names=df.columns[2:].values)

# clf.coef_



df.corrwith(df.Survived)
# pd.DataFrame({'columns':list(df.columns)[2:],'coef':list(clf.coef_.T)})

# df.value_counts()
#交叉验证特性相关性

cross_validation.cross_val_score(lr_clf,X,y,cv=5)

# split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)

# clf.fit(split_train.as_matrix()[:,2:],split_train.as_matrix()[:,1])



# #预测cv数据

# predict_cv = clf.predict(split_cv.as_matrix()[:,2:])

# bad_cases = data_train.loc[data_train.PassengerId.isin(split_cv[split_cv.as_matrix()[:,1] != predict_cv]['PassengerId'].values)]

# bad_cases

# len(bad_cases)
start_time = time.time()

params = {'penalty':['l2'],'C':[1.0,10.0,100.0],'solver':['liblinear','sag'],'tol':[1e-4,1e-3,1e-6]}

grid_clf = GridSearchCV(lr_clf,params,n_jobs=-1)

grid_clf.fit(X,y)

print(time.time()-start_time)

grid_clf.best_params_
rf = RF(n_estimators=100)

# rf.fit(X,y)

# cross_validation.cross_val_score(rf,X,y,cv=5)

# rf.fit(X,y)



#bagging

# bagging_rf = BaggingRegressor(rf,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)

# bagging_rf.fit(X,y)

bagging_clf = BaggingRegressor(grid_clf,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)

bagging_clf.fit(X,y)
clf = bagging_clf
data_test = pd.read_csv('../input/test.csv')

data_test.isnull().any()
test_df = data_test.drop(['Name','Cabin','Ticket'], axis=1)

test_df = dummy_data(test_df)

test_df
fare_test = test_df.filter(regex='Fare|SibSp|Parch|Sex_.*|Pclass_.*')

fare_temp = fare_test.Fare

fare_test.drop(['Fare'],axis=1,inplace=True)

fare_test.insert(0,'Fare',fare_temp)



predict_fare = fare_rfr.predict(fare_test[(fare_test.Fare.isnull())].as_matrix()[:,1:])

test_df.loc[test_df.Fare.isnull(),'Fare'] = predict_fare

test_df.isnull().any()
age_df = test_df.filter(regex='Age|Fare|SibSp|Parch|Sex_.*|Pclass_.*')

age_df.insert(0,'Age',age_df.pop('Age'))

predict_ages = rfr.predict(age_df[(age_df.Age.isnull())].as_matrix()[:,1:])

test_df.loc[test_df.Age.isnull(),'Age'] = predict_ages

test_df.isnull().any()
scaler = pp.StandardScaler()

age_scale_param = scaler.fit(test_df.Age.reshape(-1,1))

test_df['Age_Scaled'] = scaler.fit_transform(test_df.Age.reshape(-1,1), age_scale_param)



fare_scale_param = scaler.fit(test_df.Fare.reshape(-1,1))

test_df['Fare_Scaled'] = scaler.fit_transform(test_df.Fare.reshape(-1,1), fare_scale_param)
test_df.drop(['Age', 'Fare'], axis=1, inplace=True)

test_df
test_np = test_df.as_matrix()

X = test_np[:,1:]

# Y_pred = clf.predict(X)

# Y_pred = rf.predict(X)

# Y_pred = bagging_rf.predict(X)

Y_pred = clf.predict(X)
submission = pd.DataFrame({

        "PassengerId": data_test["PassengerId"],

        "Survived": Y_pred.astype(int)

    })

submission.to_csv('result.csv', index=False)