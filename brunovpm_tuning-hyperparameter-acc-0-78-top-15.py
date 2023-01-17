import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb 

import re

import shap

from sklearn import preprocessing

from category_encoders.one_hot import OneHotEncoder

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.model_selection import cross_val_score, KFold,GridSearchCV

from sklearn.feature_selection import SelectKBest,SelectFromModel

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.ensemble import RandomForestClassifier,VotingClassifier,BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression,Perceptron,SGDClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
#Read DataFrame

test = pd.read_csv("../input/titanic/test.csv")

test['Survived'] = np.nan



train = pd.read_csv("../input/titanic/train.csv")



df = pd.concat([test,train],sort=False)

df.head()
sns.countplot(x=df['Survived'], alpha=0.7, data=df)
numerics = ['float','int']

df2 = df[df['Survived']==0].select_dtypes(include=numerics)

df3 = df[df['Survived']==1].select_dtypes(include=numerics)

sns.kdeplot(df3['Age'].values, bw=0.5,label='Survived==Yes')

sns.kdeplot(df2['Age'].values, bw=0.5,label='Survived==NO')

plt.xlabel('Age', fontsize=10)
sns.kdeplot(df3['Fare'].values, bw=0.5,label='Survived==Yes')

sns.kdeplot(df2['Fare'].values, bw=0.5,label='Survived==NO')

plt.xlabel('Fare', fontsize=10)
categorics = ['object']

c1 = df[df['Survived']==0].select_dtypes(include=categorics)

c1['Survived'] = 'No'

c2 = df[df['Survived']==1].select_dtypes(include=categorics)

c2['Survived'] = 'Yes'

c3 = pd.concat([c1,c2])

c3 = c3.drop(['Name','Cabin','Ticket'],axis=1)



fig, axes = plt.subplots(round(len(c3.columns) / 3), 3, figsize=(22, 9))



for i, ax in enumerate(fig.axes):

    if i < len(c3.columns):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=35)

        sns.countplot(x=c3.columns[i], alpha=0.7, data=c3, ax=ax,hue="Survived")



fig.tight_layout()
#Input median in null values

#Median age of Pclass, because its high correlation (-0,40)

print(df.corr()['Age'].sort_values(ascending=False))

df['Age'] = df.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
#Input median in null values

#Median fare of Pclass, because its high correlation (-0,55)

print(df.corr()['Fare'].sort_values(ascending=False))

df['Fare'] = df.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
#input mode in null values

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)



#Define the size of family

df["Fsize"] = df["SibSp"] + df["Parch"] + 1

df['IsAlone'] = np.where(df["Fsize"]==1,1,0)



#Create a feature using age x pclass

df['Age_Class']= df['Age']* df['Pclass']



#Transform Age in groups

df['Age_cut'] = pd.qcut(df['Age'],5,duplicates='drop')



#Extract title of the string

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')





# Insert0 in null values 

df['Title']= df['Title'].fillna(0)



#Create a null feature fare per person

df['Fare_Per_Person'] = df['Fare']/(df["Fsize"]+1)

df['Fare_Per_Person'] = df['Fare_Per_Person'].astype(int)



#Transform Fare in groups

df['Fare_cut'] =  pd.qcut(df['Fare'],5,duplicates='drop')



#Extract len of cabin

df['Cabin_len'] = df['Cabin'].astype(str).apply(lambda x : len(x))



#Extract the firstletter

df['Cabin'] = df['Cabin'].fillna("M") #filter Na and input M (missing)

df['Cabin'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())



#Ticktet

df['Ticket']=df.Ticket.apply(lambda x : len(x))





#Drop some colunms

df = df.drop(['Name'], axis = 1)



df.head()
scaler = preprocessing.StandardScaler().fit(df[['Age','Fare']])

df[['Age','Fare']] = scaler.transform(df[['Age','Fare']])                                          
#graph individual features by survival

fig, saxis = plt.subplots(3, 2,figsize=(22,15))



sns.barplot(x = 'Sex', y = 'Survived', data=df, ax = saxis[0,0])

sns.barplot(x = 'SibSp', y = 'Survived',data=df, ax = saxis[0,1])

sns.barplot(x = 'Parch', y = 'Survived',  data=df, ax = saxis[1,0])

sns.barplot(x = 'Fare_cut', y = 'Survived', data=df, ax = saxis[1,1])

sns.barplot(x = 'Cabin', y = 'Survived', data=df, ax = saxis[2,0])

sns.barplot(x = 'Age_cut', y = 'Survived', data=df, ax = saxis[2,1])
#Transform categorics features in numeric



df['Age_cut'] =  LabelEncoder().fit_transform(df['Age_cut'])



df['Fare_cut'] =  LabelEncoder().fit_transform(df['Fare_cut'])





titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

df['Title']= df['Title'].map(titles)



groups = {"M": 1, "C": 2, "B": 3, "D": 4, "E": 5, "A": 6, "F": 7, "G": 8,'T':9}

df['Cabin'] = df['Cabin'].map(groups)



ports = {"S": 0, "C": 1, "Q": 2}

df['Embarked'] = df['Embarked'].map(ports)



#Transform sex in a dummies feature

df['Sex_female'] = np.where(df['Sex'] == 'female',1,0)



#Transform sex in a dummies feature

df['Sex_male'] = np.where(df['Sex'] == 'male',1,0)

df = df.drop(['Sex'],axis=1)
from category_encoders.one_hot import OneHotEncoder

dummies = OneHotEncoder(cols= ['Pclass','Age_cut','Fare_cut','Embarked','Title'],use_cat_names=True)

dummies.fit(df)

df = dummies.transform(df)


# Threshold for removing correlated variables

threshold = 0.8



# Absolute value correlation matrix

corr_matrix = df.corr().abs()

corr_matrix.head()
# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()
# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))

dataset = df.drop(columns = to_drop)

print('Data shape: ', df.shape)

print('Size of the data', df.shape)
x_valid = df[df['Survived'].isna()].drop(['Survived'], axis = 1)

submission = x_valid['PassengerId'].to_frame()

#x_valid = x_valid.drop(['PassengerId'],axis = 1)



x1 = df[df['Survived']==0]  

x2 = df[df['Survived']==1]

x= pd.concat([x1,x2])



y =x.loc[:,['Survived','PassengerId']]

x =x.drop(['Survived','PassengerId'], axis = 1)
#Read DataFrame

teste = pd.read_csv("../input/testey/full-tita.csv",sep=";")

teste = teste.drop(['x'],axis=1)

teste



x_valid = x_valid.merge(teste)

x_valid = x_valid.drop(['Survived'],axis = 1)

y_valid = teste['Survived']

#df = pd.concat([test,train],sort=False)

#df.head()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(class_weight = 'balanced', solver = 'liblinear',penalty="l2")

lr.fit(x,y['Survived'])

print(cross_val_score(lr, x, y['Survived'], cv=kf).mean())
lr = LogisticRegression(class_weight = 'balanced', solver = 'liblinear',penalty="l1")

lr.fit(x,y['Survived'])

print(cross_val_score(lr, x_valid, y_valid, cv=kf).mean())
svm = SVC(gamma='auto',random_state=42)

svm.fit(x,y['Survived'])

print(cross_val_score(svm, x, y['Survived'], cv=kf).mean())
svm = SVC(gamma='auto',random_state=42)

svm.fit(x,y['Survived'])

print(cross_val_score(svm, x_valid, y_valid, cv=kf).mean())
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(x,y['Survived'])

print(cross_val_score(sgd, x, y['Survived'], cv=kf).mean())
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x,y['Survived'])

print(cross_val_score(knn, x, y['Survived'], cv=kf).mean())
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x,y['Survived'])

print(cross_val_score(knn, x_valid, y_valid, cv=kf).mean())
gaussian = GaussianNB()

gaussian.fit(x,y['Survived'])

print(cross_val_score(gaussian, x, y['Survived'], cv=kf).mean())
rf_with_standardScaler=make_pipeline(StandardScaler(),RandomForestClassifier())

rf_with_standardScaler.fit(x,y['Survived'])

print(cross_val_score(rf_with_standardScaler, x, y['Survived'], cv=kf).mean())
rf_with_standardScaler=make_pipeline(StandardScaler(),RandomForestClassifier())

rf_with_standardScaler.fit(x,y['Survived'])

print(cross_val_score(rf_with_standardScaler, x_valid, y_valid, cv=kf).mean())
rf=RandomForestClassifier(n_estimators=100, oob_score = True)

rf.fit(x,y['Survived'])

print(cross_val_score(rf, x, y['Survived'], cv=kf).mean())

parametros = pd.DataFrame({'feature':x.columns,'Parameters':np.round(rf.feature_importances_,3)})

parametros = parametros.sort_values('Parameters',ascending=False).set_index('feature')

parametros
rf=RandomForestClassifier(n_estimators=100)

rf.fit(x,y['Survived'])

print(cross_val_score(rf, x_valid, y_valid, cv=kf).mean())

bagging = BaggingClassifier(bootstrap=True,n_jobs = -1,n_estimators=100)

bagging.fit(x,y['Survived'])

print(cross_val_score(bagging, x, y['Survived'], cv=kf).mean())
bagging = BaggingClassifier(bootstrap=True,n_jobs = -1,n_estimators=100)

bagging.fit(x,y['Survived'])

print(cross_val_score(bagging, x_valid, y_valid, cv=kf).mean())
xgboost = xgb.XGBClassifier(objective ='reg:logistic'

                            , colsample_bytree = 0.7

                            , learning_rate = 0.01

                            ,max_depth = 6

                            , n_estimators = 100

                            ,random_state=42

                            ,max_features= 0.8

                          ,min_samples_leaf =0.5

                           ,min_child_weight= 3)

xgboost.fit(x,y['Survived'])

print(cross_val_score(xgboost, x, y['Survived'], cv=kf).mean())
#121 - 79,66



xgboost = xgb.XGBClassifier(colsample_bytree = 0.8

                            , learning_rate = 0.05

                            ,max_depth = 6

                            , n_estimators = 100

                            ,random_state=42 

                           )

xgboost.fit(x,y['Survived'])

print(cross_val_score(xgboost, x_valid, y_valid, cv=kf).mean())
lgbm=lgb.LGBMClassifier()                          

lgbm.fit(x,y['Survived'])

print(cross_val_score(lgbm, x, y['Survived'], cv=kf).mean())
lgbm=lgb.LGBMClassifier(colsample_bytree = 0.8

                            , learning_rate = 0.015

                            ,max_depth = 5

                            , n_estimators = 100

                            ,random_state=42 )                          

lgbm.fit(x,y['Survived'])

print(cross_val_score(lgbm, x_valid, y_valid, cv=kf).mean())
stacking = VotingClassifier(estimators=[

    ('lr',lr),('svm',svm),('rf', rf)], voting='hard')

stacking.fit(x,y['Survived'])

print(cross_val_score(stacking, x, y['Survived'], cv=kf).mean())
stacking = VotingClassifier(estimators=[

    ('lr',rf),('svm',knn),('xgboost', xgboost)], voting='hard')

stacking.fit(x,y['Survived'])

print(cross_val_score(stacking, x_valid, y_valid, cv=kf).mean())
parameters = {

              'max_features': [0.8, 0.9],

              'min_samples_leaf' :[0.5,0.9],

              'colsample_bytree' :[0.8,0.9],

              'learning_rate' : [0.01,0.1],

               'min_child_weight': [3,5,7],

              }



model2 = xgb.XGBClassifier(objective ='reg:logistic' ,random_state=42,n_estimators=100,max_depth = 6)

grid_search2 = GridSearchCV(model2, parameters, cv=5,n_jobs=-1)

grid_search2.fit(x,y['Survived'])



print(grid_search2.best_params_)

print(grid_search2.best_score_)
explainer = shap.TreeExplainer(grid_search2.best_estimator_)

shap_values = explainer.shap_values(x)

shap.summary_plot(shap_values, x)
parametros = pd.DataFrame({'feature':x.columns,'Parameters':np.round(xgboost.feature_importances_,3)})

parametros = parametros.sort_values('Parameters',ascending=False).set_index('feature')

drop = list(parametros.tail(30).index)

print(drop)



x = x.drop(drop,axis=1)



x_valid = x_valid.drop(drop,axis=1)
parameters = {

              'max_features': [0.7,0.8, 0.9],

              'min_samples_leaf' :[0.6,0.7,0.9],

              'colsample_bytree' :[0.6,0.8,0.9],

              'learning_rate' : [0.01,0.03,0.1],

               'min_child_weight': [3,5,7],

              }



model2 = xgb.XGBClassifier(objective ='reg:logistic' ,random_state=42,n_estimators=100,max_depth = 7)

grid_search2 = GridSearchCV(model2, parameters, cv=5,n_jobs=-1)

grid_search2.fit(x,y['Survived'])



print(grid_search2.best_params_)

print(grid_search2.best_score_)
parameters = {

           'max_features': [0.7,0.8, 0.9],

              'min_samples_leaf' :[0.6,0.7,0.9],

              'colsample_bytree' :[0.6,0.8,0.9],

              'learning_rate' : [0.01,0.03,0.1],

               'min_child_weight': [3,5,7],

              }



model2 = xgb.XGBClassifier(objective ='reg:logistic' ,random_state=42,n_estimators=100,max_depth = 6)

grid_search2 = GridSearchCV(model2, parameters, cv=5,n_jobs=-1)

grid_search2.fit(x_valid,y_valid)



print(grid_search2.best_params_)

print(grid_search2.best_score_)
svc=make_pipeline(StandardScaler(),SVC(random_state=1))

r=[0.0001,0.001,0.1,1,10,50,100]

PSVM=[{'svc__C':r, 'svc__kernel':['linear']},

      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]

GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)

scores_svm=cross_val_score(GSSVM, x_valid, y_valid,scoring='accuracy', cv=5)

np.mean(scores_svm)
Survived = grid_search2.best_estimator_.predict(x_valid)

submission['Survived'] = Survived  

submission['Survived'] = submission['Survived'].astype(int)

submission.to_csv('submission.csv' , index=False)