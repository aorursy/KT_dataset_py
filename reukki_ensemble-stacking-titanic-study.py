import pandas as pd #판다스 / 데이터프레임

import numpy as np #넘파이/ 행렬연산

import re # 정규표현식

import sklearn # 사이킷런

import xgboost as xgb #xgboost

import seaborn as sns # 데이터 시각화

import matplotlib.pyplot as plt # 기본 시각화

%matplotlib inline 

# 주피터 안에서 띄우기 위함



import plotly.offline as py #plotly 

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings # 경고 무시

warnings.filterwarnings('ignore')
# Stacking 기본 모델

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,

                             GradientBoostingClassifier, ExtraTreesClassifier)



from sklearn.svm import SVC # 서포트 벡터 머신 - 커널 활용

from sklearn.model_selection import KFold# 교차 검증
train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
train.head(3)
PassengerId = test['PassengerId']

test.head(3)
train.isnull().sum()
test.isnull().sum()
print(train[['Pclass', 'Survived']].groupby(['Pclass']).mean())
plt.style.use('seaborn')

f ,ax= plt.subplots(1,2,figsize=(12,8))



#객실 등급별 생존자 수

train[['Pclass', 'Survived']].groupby(['Pclass']).sum().plot.bar(ax=ax[0])

ax[0].set_title('Survived')

ax[0].set_ylabel('Counts')



sns.countplot('Pclass',hue='Survived',data=train, ax=ax[1])
# 성별 생존자 평균 - 여자 생존자 확률이 많이 높음

print(train[['Sex','Survived']].groupby(['Sex'], as_index = False).mean())
full_data = [train, test]



for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

print(train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1,'IsAlone'] = 1

    

print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

print(train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    

train['CategoricalFare'] = pd.qcut(train['Fare'],4)

print(train[['CategoricalFare','Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg- age_std, age_avg + age_std, size = age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

    train['CategoricalAge'] = pd.cut(train['Age'],5)

    

print(train[['CategoricalAge','Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):

    title_search = re.search('([A-Za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

    

print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for dataset in full_data:

    dataset['Sex'] = dataset['Sex'].map({'female':0,'male':1}).astype(int)

    

    title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

    

    #요금가격

    dataset.loc[(dataset['Fare']<=7.91, 'Fare')] = 0

    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454),'Fare']=1

    dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31),'Fare']=2

    dataset.loc[(dataset['Fare']>31),'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    #나이

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age'] = 1

    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age'] = 2

    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64),'Age'] = 3

    dataset.loc[dataset['Age']>64, 'Age'] = 4
#Feature Selection



drop_elements = ['PassengerId','Name','Ticket','Cabin','SibSp','Parch','FamilySize']

train = train.drop(drop_elements, axis=1)

train =train.drop(['CategoricalAge', 'CategoricalFare'],axis=1)



test = test.drop(drop_elements,axis=1)

print(train.head(10))



train = train.values

test = test.values
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



#분류기 생성

classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()

]





log_cols = ['Classifier','Accuracy']

log = pd.DataFrame(columns=log_cols)



# n_splits = 반복 횟수 및 재 셔플링

# StratifiedKfold 와 shuffleSplit의 병합

# StratifiedKfold - 레이블 분포가 전체 데이터 셋의 레이블 분포 따르게끔 유지

# ShuffleSplit - train / test split의 사용자 정의 번호 생성

sss = StratifiedShuffleSplit(n_splits=10,test_size=0.1, random_state = 0)



X = train[0::, 1::]

y = train[0::,0]



acc_dict = {}



for train_index, test_index in sss.split(X,y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    for clf in classifiers:

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc

    

for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)

    

plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes('muted')

sns.barplot(x='Accuracy',y='Classifier',data=log, color='b')

    
train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head(3)
full_data = [train, test]



# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',

       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 

NFOLDS = 5 #k-fold 수

kf = KFold(n_splits = NFOLDS, random_state = SEED)



# feature_importance 생성하기위한 클래스 

class SklearnHelper(object):

    #생성

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)

    

    #모델 훈련

    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)

    #모델 예측

    def predict(self, x):

        return self.clf.predict(x)

    #모델 학습

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    # 특성 중요도

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)

        return self.clf.fit(x,y).feature_importances_
#객체 만들기

def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)

        

        #해당 폴드에서 학습된 모델에 검증 데이터로 예측

        oof_train[test_index] = clf.predict(x_te)

        

        #해당 폴드에서 원본 테스트 데이터로 예측

        oof_test_skf[i, :] = clf.predict(x_test)



    #테스트 셋에 대한 예측 반환

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#랜덤포레스트

rf_params = {

    'n_jobs':-1,

    'n_estimators' : 500,

    'warm_start' : True,

    'max_depth' : 6,

    'min_samples_leaf':2,

    'max_features' : 'sqrt',

    'verbose':0

}



#Extra Trees

et_params = {

    'n_jobs' : -1,

    'n_estimators':500,

    'max_depth' : 8,

    'min_samples_leaf' : 2,

    'verbose' : 0

}



#AdaBoost parameters

gb_params = {

    'n_estimators': 500,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



#서포터 벡터 머신 / 커널

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}
#랜덤 포레스트

rf = SklearnHelper(clf=RandomForestClassifier, seed = SEED, params=rf_params)

et = SklearnHelper(clf = ExtraTreesClassifier, seed = SEED, params=et_params)

ada = SklearnHelper(clf = AdaBoostClassifier, seed = SEED, params = ada_params)

svc = SklearnHelper(clf=SVC, seed= SEED, params = svc_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

x_train = train.values

x_test = test.values
#객체 생성 - 모델,훈련셋,훈련 레이블,테스트셋

et_oof_train, et_oof_test = get_oof(et,x_train, y_train,x_test)

rf_oof_train, rf_oof_test = get_oof(rf,x_train,y_train, x_test)

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train,x_test)

gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test)
# feature Importance 반환

rf_feature = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)
rf_features =  [0.12724711, 0.20305232, 0.03141594, 0.02082381, 0.07181766, 0.02346052,

 0.10859766, 0.06486906, 0.06655421, 0.01332034, 0.26884137]

et_features =  [0.12029486, 0.38409962, 0.02704675, 0.01711051, 0.0575024,  0.02727249,

 0.04602978, 0.0836201,  0.04460883, 0.02177028, 0.17064438]

ada_features = [0.026, 0.014, 0.02,  0.064, 0.038, 0.01,  0.694, 0.012, 0.048, 0.006, 0.068]

gb_features = [0.08770115, 0.01446721, 0.04869985, 0.01284386, 0.04989932, 0.02586046

 ,0.1723618,  0.03681898, 0.11384058, 0.00610441, 0.43140238]
#데이터 쉽게 시각화하기 위해 데이터 프레임화



cols = train.columns.values

feature_dataframe = pd.DataFrame( 

    {'features': cols,

     'Random Forest feature importances': rf_features,

     'Extra Trees  feature importances': et_features,

      'AdaBoost feature importances': ada_features,

    'Gradient Boost feature importances': gb_features

    })
trace = go.Scatter(

    y = feature_dataframe['Random Forest feature importances'].values,

    x = feature_dataframe['features'].values,

    mode = 'markers',

    marker = dict(

        sizemode = 'diameter',

        sizeref=1,

        size=25,

        color = feature_dataframe['Random Forest feature importances'].values,

        colorscale='Portland',

        showscale = True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout = go.Layout(

    autosize=True,

    title = 'Random Forest Feature Importance',

    hovermode = 'closest',

    yaxis = dict(

        title = 'Feature Importance',

        ticklen = 5,

        gridwidth = 2

    ),

    showlegend = False

)



fig = go.Figure(data = data, layout = layout)

py.iplot(fig, filename='scatter2020')



trace = go.Scatter(

    y = feature_dataframe['Extra Trees  feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Extra Trees  feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]





layout= go.Layout(

    autosize= True,

    title= 'Extra Trees Feature Importance',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2020')





trace = go.Scatter(

    y = feature_dataframe['AdaBoost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['AdaBoost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'AdaBoost Feature Importance',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



trace = go.Scatter(

    y = feature_dataframe['Gradient Boost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_dataframe['Gradient Boost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Feature Importance',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2020')
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1)

feature_dataframe.head(5)
y = feature_dataframe['mean'].values

x = feature_dataframe['features'].values

data = [go.Bar(

            x= x,

             y= y,

            width = 0.5,

            marker=dict(

               color = feature_dataframe['mean'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    autosize= True,

    title= 'Barplots of Mean Feature Importance',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar-direct-labels')
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()
data = [

    go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Viridis',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
# 하나의 분류기의 데이터 셋에 사용하기 위해 합친다.

# kfold의 결과와 x_test의 결과를 stacking

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
gbm = xgb.XGBClassifier(

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
predictions
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")