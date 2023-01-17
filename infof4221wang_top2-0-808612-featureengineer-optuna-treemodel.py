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
# Import libraries

import numpy as np

import pandas as pd

import IPython

# warning

from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning

# statistics

from scipy.stats import skew, kurtosis,probplot,norm

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import fcluster

# plot libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Preprocessing and encoder

from sklearn.preprocessing import MinMaxScaler,StandardScaler,OrdinalEncoder, LabelEncoder

# model evaluation and selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,ShuffleSplit,StratifiedKFold

from sklearn.model_selection import cross_val_score,cross_val_predict

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import optuna

# Classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn.neural_network import MLPClassifier

# Ensemble Classifiers

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier,VotingClassifier,StackingClassifier

from xgboost.sklearn import XGBClassifier

# Neural Network

import tensorflow as tf

from tensorflow import keras

import kerastuner as kt
# load dataset

titanic_raw_train = pd.read_csv('..//input/titanic/train.csv')

titanic_raw_test = pd.read_csv('..//input/titanic/test.csv')

titanic_raw_train.info()

titanic_raw_test.info()
titanic_raw_train.head()
titanic_raw_train.describe(include='all')
# make a copy for feature engineering

train = titanic_raw_train.copy()

test = titanic_raw_test.copy()

train_test = pd.concat([train,test])
corr = train.corr()

corr
sns.heatmap(data=corr,vmax=1,vmin=-1,center=0,annot=True)
train.Pclass.describe()
# Pclass Distribution by Survived

sns.catplot(data=train,x='Pclass',hue='Survived', kind="count")
sns.catplot(data=train,x='Pclass',y='Survived', kind="point")
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train)
# Fare distribution by Pclass with Survived as label

sns.catplot(data=titanic_raw_train,x='Pclass',y='Fare',hue='Survived',kind='swarm')
train[['Survived','Pclass']].groupby('Pclass').mean()
#looks for strings which lie between A-Z or a-z and followed by a .

train_test['Title']=train_test.Name.str.extract('([A-Za-z]+)\.') 
pd.crosstab(train_test.Title, train_test.Survived)
train_test.Title.replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],

                         ['Miss','Mrs','Miss','Rare','Rare','Rare','Rare','Rare','Rare','Rare','Rare','Rare','Rare','Rare'],inplace=True)
pd.crosstab(train_test.Title, train_test.Survived)
# Is_married feature based on title

train_test.loc[train_test['Title'] == 'Mrs','Is_Married'] = 1
# extract family name

train_test['Family_name'] = train_test.Name.str.extract('(\w+),', expand=False)

train_test.Family_name
# Use mean survived rate as Familiy_survived feature 

m = train_test[['Family_name', 'Survived']].groupby('Family_name').mean()

c = train_test[['Family_name', 'PassengerId']].groupby('Family_name').count()

m = m.rename(columns={'Survived': 'Family_survived'})

c = c.rename(columns={'PassengerId': 'FamilyMemberCount'})

# if family name is unique in all data, set Family_survived as -1

m = m.where(m.join(c).FamilyMemberCount > 1, other=-1).fillna(-1).join(c)

m.Family_survived = m.Family_survived.astype('int64')

train_test = train_test.join(m, on='Family_name')
train_test.Age.isnull().sum()
train_test[['Age','Title']].groupby('Title').mean()
# fill in missing ages

train_test.loc[(train_test.Age.isnull())&(train_test.Title=='Mr'),'Age']=32

train_test.loc[(train_test.Age.isnull())&(train_test.Title=='Mrs'),'Age']=37

train_test.loc[(train_test.Age.isnull())&(train_test.Title=='Master'),'Age']=5

train_test.loc[(train_test.Age.isnull())&(train_test.Title=='Miss'),'Age']=22

train_test.loc[(train_test.Age.isnull())&(train_test.Title=='Rare'),'Age']=45
train_test['Age'] = train_test['Age'].astype('int64')
sns.distplot(a=train_test.Age, kde=True)
def plot_skew(feature):

    """

    Function to plot distribution and probability(w.r.t quantiles of normal distribution)

    """

    fig, axs = plt.subplots(figsize=(20,10),ncols=2)

    sns.distplot(feature,kde=True,fit=norm,ax=axs[0])

    # Generates a probability plot of sample data against the quantiles of a specified theoretical distribution (the normal distribution by default).

    f=probplot(feature, plot=plt)

    print('Skewness: {:f}'.format(feature.skew()))

    print('Kurtosis: {:f}'.format(feature.kurtosis()))

plot_skew(train.Age)
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy?scriptVersionId=2051374

plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()
sns.catplot(x='Sex',data=train,hue = 'Survived', kind='count')
sns.catplot(x="Sex", y="Survived", kind="point", data=train);
# https://seaborn.pydata.org/tutorial/categorical.html#distributions-of-observations-within-categories

sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="point", data=train);
train[['Survived','Sex']].groupby('Sex').mean()
corr
train.SibSp.describe()
sns.distplot(a=train.SibSp, kde=False)
plt.figure(figsize=[10,6])

plt.subplot(121)

plt.hist(x = [train[train['Survived']==1]['SibSp'], train[train['Survived']==0]['SibSp']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('SibSp Histogram by Survival')

plt.xlabel('# SibSp')

plt.ylabel('# of Passengers')

plt.subplot(122)

plt.hist(x = [train[train['Survived']==1]['Parch'], train[train['Survived']==0]['Parch']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Parch Histogram by Survival')

plt.xlabel('# Parch')

plt.ylabel('# of Passengers')

plt.legend()
sns.catplot(x='SibSp',kind='count',hue='Survived', data=train)
sns.barplot(x = 'SibSp', y = 'Survived', data=train)
fig, saxis = plt.subplots(1, 2,figsize=(16,12))

sns.barplot(x = 'SibSp', y = 'Survived', data=train,ax=saxis[0])

sns.barplot(x = 'Parch', y = 'Survived', order=[1,2,3], data=train, ax=saxis[1])
train_test.Fare.isnull().sum()
train_test.Fare.fillna(test.Fare.median(), inplace=True)
# Fare distribution

plt.figure(figsize=(10,5))

sns.distplot(a=titanic_raw_train.Fare, kde=True)
plot_skew(train_test.Fare)
train_test['Fare_log'] = np.log1p(train_test.Fare)

plot_skew(train_test.Fare_log)
# extract ticket numbers 

ticket = train_test.Ticket.str.extract('(\d+$)', expand=False).fillna(0).astype(int).ravel()

# cluster data from https://www.kaggle.com/shaochuanwang/titanic-ml-tutorial-on-small-dataset-0-82296/notebook

Z = linkage(ticket.reshape(train_test.shape[0], 1), 'single')

clusters = fcluster(Z, 20, criterion='distance')

train_test['Ticket_Code'] = clusters
import itertools

count = train_test[['PassengerId', 'Ticket_Code']].groupby('Ticket_Code').count().rename(columns={'PassengerId': 'Number'})

train_test['Ticket_Code_Remap'] = train_test.Ticket_Code.replace(dict(zip(count.index[count.Number <= 10], itertools.cycle([0]))))

fig, axs = plt.subplots(figsize=(20,20),nrows=2)

sns.barplot(train_test.Ticket_Code, train_test.Survived,ax=axs[0])

sns.barplot(train_test.Ticket_Code_Remap, train_test.Survived,ax=axs[1])

# ticket frequency

train_test['Ticket_Frequency'] = train_test.groupby('Ticket')['Ticket'].transform('count')

train_test.Ticket_Frequency.astype('int64')
train_test.Cabin.isnull().sum()
# group cabin using area code

train_test['Cabin_code'] = train_test.Cabin.str.get(0).fillna('Z')

train_test[['Survived','Cabin_code']].groupby('Cabin_code').mean()
train.Embarked.isnull().sum()

train_test.Embarked.fillna(train_test.Embarked.mode()[0], inplace=True)

pd.crosstab(train_test.Embarked, train_test.Survived)
sns.catplot(data=train,x='Embarked',y='Survived', kind='point')
train[['Embarked','Survived']].groupby('Embarked').mean()
train_test['Family_size']=train_test.SibSp+train_test.Parch

train_test['Is_alone']=1

train_test.loc[train_test['Family_size'] > 1,'Is_alone'] = 0

train_test['Family_size'].astype('int64')

train_test['Is_alone'].astype('int64')
plt.figure(figsize=(10,8))

sns.catplot(x='Family_size', y='Survived',kind='point',data = train_test)
sns.catplot(x='Is_alone', y='Survived',kind='point',data = train_test)
# #choose related data to predict age

# age_df =train_test[['Age','Fare', 'Family_size', 'Title', 'Pclass','Is_alone','Sex']]

# age_df_notnull = age_df.loc[train_test['Age'].notnull()]

# age_df_isnull = age_df.loc[(train_test['Age'].isnull())]

# Xtr_age = pd.get_dummies(age_df_notnull.drop(columns=['Age']))

# Xte_age = pd.get_dummies(age_df_isnull.drop(columns=['Age']))

# Y_age = age_df_notnull.Age

# # use RandomForestRegression to train data

# RFR = RandomForestRegressor(n_estimators=100, n_jobs=-1)

# RFR.fit(Xtr_age,Y_age)

# predictAges = RFR.predict(t)

# train_test.loc[train_test['Age'].isnull(), ['Age']]= predictAges

# RFR.score(Xtr_age,Y_age)
fig, axs = plt.subplots(figsize=(10,8),ncols=2)

sns.distplot(a=train.Age, ax=axs[0])

sns.distplot(a=train.Fare, ax=axs[1])
train_test['Age_band'] = pd.cut(train_test.Age, bins=10, precision=0)
train_test['Fare_band'] = pd.qcut(train_test.Fare_log, q=13, precision=2)
fig, axs = plt.subplots(figsize=(28,8),ncols=2)

sns.pointplot(x='Age_band', y='Survived', data=train_test,ax=axs[0])

sns.pointplot(x='Fare_band', y= 'Survived', data=train_test,ax=axs[1])
MM_scaler = MinMaxScaler(feature_range=(0,1))

train_test[['Age_scaled']]=MM_scaler.fit_transform(train_test[['Age']])

sns.distplot(a=train_test.Age_scaled)
St_scaler = StandardScaler()

train_test[['Fare_scaled']]=St_scaler.fit_transform(train_test[['Fare_log']])

sns.distplot(a=train_test.Fare_scaled)
le = OrdinalEncoder()
train_test[['Age_code']] = le.fit_transform(train_test[['Age_band']])

train_test[['Fare_code']] = le.fit_transform(train_test[['Fare_band']])
for row in train_test:

    train_test.loc[train_test['Family_size']==0, 'Family_type']=1

    train_test.loc[(1<=train_test['Family_size'])&(train_test['Family_size']<=3), 'Family_type']=2

    train_test.loc[(3<=train_test['Family_size'])&(train_test['Family_size']<=6), 'Family_type']=3

    train_test.loc[7<=train_test['Family_size'], 'Family_type']=4

train_test.Family_type = train_test.Family_type.astype('int64')
# lb_cabin = LabelEncoder()

# lb_title = LabelEncoder()

# lb_embarked = LabelEncoder()

# train_test.Cabin_code = lb_cabin.fit_transform(train_test.Cabin_code)

# train_test.Title = lb_title.fit_transform(train_test.Title)

# train_test.Embarked = lb_embarked.fit_transform(train_test.Embarked)
# encode Sex

train_test.Sex.replace(['female', 'male'], [1,0],inplace=True)
train_test.info()
X = pd.get_dummies(train_test[['Pclass','Sex','Title','Family_type',

                               'Age_code','Fare_code','Embarked',

                               'Cabin_code','Ticket_Code_Remap','Family_survived']])[:891]

X_test = pd.get_dummies(train_test[['Pclass','Sex','Title','Family_type',

                                    'Age_code','Fare_code','Embarked',

                                    'Cabin_code','Ticket_Code_Remap','Family_survived']])[891:]

Y = train.Survived

X.shape
sns.heatmap(X.join(Y).corr(),annot=True,cmap='RdYlGn',annot_kws={'size':10})

fig=plt.gcf()

fig.set_size_inches(22,22)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.show()
X.shape
CV = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

n_trials = 500# no. of trials during tuning
# # hyperparameter tuning using GridSearchCV

# penalty  = ['l1', 'l2','elasticnet', 'none'] # specify the norm used in the penalization

# C = np.logspace(-2, 2, 10) # 50 nums start form 0.1 to 10 Inverse of regularization strength

# hyper={'penalty':penalty,'C':C}

# gd=GridSearchCV(estimator=LogisticRegression(),param_grid=hyper,verbose=2,n_jobs=-1,cv=CV,refit=True,scoring='accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# LR_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     penalty = trial.suggest_categorical('penalty',['l1', 'l2','elasticnet', 'none'])

#     if penalty !='none':

#         C =trial.suggest_loguniform('C', 1e-4, 1e4)

#         # define classifier

#         LR_clf = LogisticRegression(penalty=penalty, C=C)

#     else:

#         # define classifier

#         LR_clf = LogisticRegression(penalty=penalty)

#     # define evaluation matrix as objective to return

#     score = cross_val_score(LR_clf, X, Y, n_jobs=-1, cv=CV, scoring='roc_auc')

#     accuracy = score.mean()

#     return accuracy

# # create study

# study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='maximize')

# # run study to find best objective

# study.optimize(objective, n_trials=200)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# LR_param = study.best_params

LR_param = {'penalty': 'l2', 'C': 109.16587461586346} 

LR_best= LogisticRegression(**LR_param)
# # hyperparameter tuning using GridSearchCV

# C=[0.4,0.5,0.6,0.8,1,5] # Regularization parameter.

# gamma=['scale','auto',0.01,0.1,0.2,0.3,0.5,1]

# kernel=['rbf']

# degree=[3,5,7] # Degree of the polynomial kernel 

# hyper={'kernel':kernel,'C':C,'gamma':gamma,'degree':degree}

# gd=GridSearchCV(estimator=SVC(),param_grid=hyper,verbose=5,n_jobs=-1,cv=CV,refit=True,scoring='accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# SVM_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):



#     gamma = trial.suggest_loguniform('gamma',1e-4,1e2) #Kernel coefficient 

#     C =trial.suggest_loguniform('C', 1e-2, 1e3) # Regularization parameter.

#     kernel = trial.suggest_categorical('kernel',['rbf'])

#     #degree = trial.suggest_int('degree',1,3)

#     clf = SVC(gamma=gamma, C=C,kernel=kernel)

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV, scoring='roc_auc')

#     accuracy = score.mean()

#     return accuracy

# study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='maximize')

# s=study.optimize(objective, n_trials=n_trials, n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
#SVM_params = study.best_params

SVM_params = {'gamma': 0.04190895340786603, 'C': 4.615992928303742, 'kernel': 'rbf'}

SVM_best= SVC(**SVM_params,probability=True)
# criterion=['gini', 'entropy'] #The function to measure the quality of a split.

# max_depth=[3,5,10,15,20,None]  #The maximum depth of the tree. 

# splitter=['best', 'random']#The strategy used to choose the split at each node.

# min_samples_split = [1,3,0.01,0.05,0.1] #The minimum num of samples required to split an internal node.

# min_samples_leaf = [1,3,5,7,0.1] # minimum num of samples required to be a leaf node

# max_features=['auto'] #The number of features to consider when looking for the best split

# hyper={'criterion':criterion, 'max_depth':max_depth,'splitter':splitter,

#        'min_samples_split':min_samples_split,'max_features':max_features,

#       'min_samples_leaf':min_samples_leaf}

# gd=GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),param_grid=hyper,

#                 verbose=2,n_jobs=-1,cv=CV,refit=True,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# DT_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):



#     max_depth = trial.suggest_int("max_depth", 2, 32,5) #The maximum depth of the tree.

#     min_samples_split = trial.suggest_int('min_samples_split', 20,200,20) ##The minimum num of samples required to split an internal node.

#     min_samples_leaf = trial.suggest_int('min_samples_leaf', 20, 200,20) # minimum num of samples required to be a leaf node

#     clf = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth, 

#                                    min_samples_leaf=min_samples_leaf, random_state=42)

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV, scoring='roc_auc')

#     accuracy = score.mean()

#     return accuracy

# study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='maximize')

# s=study.optimize(objective, n_trials=n_trials, n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# DT_param =study.best_params

DT_param ={'max_depth': 12, 'min_samples_split': 60, 'min_samples_leaf': 20}

DT_best= DecisionTreeClassifier(**DT_param)
# Visualize Decision Tree

DT_best.fit(X,Y)

fig = plt.figure(figsize=(30,30))

_ = tree.plot_tree(DT_best, 

                   feature_names=X.columns,

                   filled=True)
# # Grid Search hyperparameter tunning

# n_neighbors=[5,10,15,20,100,200] #Number of neighbors 

# weights=['uniform','distance'] # weight function used in prediction

# p=[1,2] #Power parameter for the Minkowski metric.

# hyper={'n_neighbors':n_neighbors,'weights':weights,'p':p}

# gd=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=hyper,cv=CV,refit=True,verbose=2,n_jobs=-1,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# KNN_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_neighbors=trial.suggest_int('n_neighbors',5,305,50) #Number of neighbors 

#     weights=trial.suggest_categorical('weights',['uniform','distance']) # weight function used in prediction

#     p=trial.suggest_int('p',1,2) #Power parameter for the Minkowski metric.

#     # define classifier

#     clf = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,p=p,n_jobs=-1)

#     # define evaluation matrix as objective to return

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV, scoring='roc_auc')

#     accuracy = score.mean()

#     return accuracy

# # create study

# study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective, n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# KNN_param = study.best_params

KNN_param ={'n_neighbors': 5, 'weights': 'uniform', 'p': 1}

KNN_best= KNeighborsClassifier(**KNN_param)
# # network builder

# def model_builder(hp):

#     model = keras.Sequential()

#     # Dense layer

#     for i in range(hp.Int('layers',1,2)):

#         model.add(keras.layers.Dense(units=hp.Int('units_{:}'.format(i),min_value=32, max_value=512,step=32), 

#                                      activation = hp.Choice('actv_{:}'.format(i),['relu','tanh'])))

#     # output layer

#     model.add(keras.layers.Dense(1,activation='sigmoid'))

#     # model config

#     model.compile('adam', 'binary_crossentropy',metrics=[keras.metrics.AUC(name='auc')])

#     return model
# # build tuner

# tuner = kt.Hyperband(model_builder, 

#                     objective = kt.Objective('val_auc','max'),

#                     max_epochs=20,

#                     factor = 3,

#                     directory = 'keras_logs',

#                     project_name = 'NN')

# class ClearTrainingOutput(tf.keras.callbacks.Callback):

#     def on_train_batch_end(*args,**kwargs):

#         IPython.display.clear_output(wait=True)

# # start tuning

# tuner.search(X,Y,epochs = 20, validation_split=0.2,callbacks=[ClearTrainingOutput()])
# NN_best_params = tuner.get_best_hyperparameters(num_trials=1)[0]

# NN_best_model = tuner.hypermodel.build(NN_best_params)
# # Grid Search to tune parameters

# hidden_layer_sizes=[(40,40),(80,80),(100,),(40),(80),(120)] # the number of neurons in the ith hidden layer

# activation=['identity', 'logistic', 'tanh', 'relu'] # activation function

# solver=['lbfgs','sgd','adam'] #The solver for weight optimization.

# alpha=[0.0001,0.001,0.01,.1] #L2 penalty (regularization term) parameter.

# learning_rate=['constant','invscaling','adaptive']# Learning rate schedule for weight updates.

# hyper={'hidden_layer_sizes':hidden_layer_sizes,'activation':activation,

#        'solver':solver,'alpha':alpha,'learning_rate':learning_rate}

# gd=GridSearchCV(estimator=MLPClassifier(random_state=42,early_stopping=True),param_grid=hyper,

#                 cv=CV,refit=True,verbose=2,n_jobs=-1,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# MLP_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_layers = trial.suggest_int('n_layers', 1, 2) # no. of hidden layers 

#     layers = []

#     for i in range(n_layers):

#         layers.append(trial.suggest_int(f'n_units_{i+1}', 10, 210,50)) # no. of hidden unit

#     activation=trial.suggest_categorical('activation',['logistic', 'tanh', 'relu']) # activation function 

#     alpha=trial.suggest_loguniform('alpha',0.0001,50) #L2 penalty (regularization term) parameter.

#     # define classifier

#     clf = MLPClassifier(random_state=42,

#                         solver='adam',

#                         early_stopping=True,

#                         activation=activation,

#                         alpha=alpha,

#                         learning_rate='adaptive',

#                         learning_rate_init=0.01,

#                         batch_size=32,

#                         hidden_layer_sizes=(layers))

#     # define evaluation matrix as objective to return

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV)

#     accuracy = score.mean()

#     return accuracy

# # create study

# study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials =500,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# MLP_param =study.best_params

MLP_best= MLPClassifier(random_state=42,

                        solver='adam',

                        early_stopping=True,

                        activation='relu',

                        alpha= 0.0002746340910250398,

                        learning_rate='adaptive',

                        learning_rate_init=0.01,

                        batch_size=32,

                        hidden_layer_sizes=160)
# # Grid Search

# n_estimators=[40,50,60,500] #The number of trees in the forest.

# # criterion=['gini', 'entropy']#The function to measure the quality of a split. 

# max_depth=[3,4,5,7]#The maximum depth of the tree.

# min_samples_split = [3,0.01,0.05,0.1] #The minimum num of samples required to split an internal node.

# min_samples_leaf = [3,5,7,0.1] # minimum num of samples required to be a leaf node

# max_features=['auto'] #The number of features to consider when looking for the best split

# # oob_score=['True','False']

# hyper={'n_estimators':n_estimators, 'max_depth':max_depth,

#       'min_samples_split':min_samples_split,'max_features':max_features,

#        'min_samples_leaf':min_samples_leaf}

# gd=GridSearchCV(estimator=RandomForestClassifier(random_state=42,oob_score=True,criterion='gini'),

#                 param_grid=hyper,verbose=2,n_jobs=-1,cv=CV,refit=True,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# RF_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_estimators=trial.suggest_int('n_estimators',50,500,50) #The number of trees in the forest.

#     max_depth=trial.suggest_int('max_depth',1,5,1)#The maximum depth of the tree.

#     min_samples_split = trial.suggest_int('min_samples_split',20,200,20) #The minimum num of samples required to split an internal node.

#     min_samples_leaf = trial.suggest_int('min_samples_leaf',20,200,20) # minimum num of samples required to be a leaf node

#     # define classifier

#     clf = RandomForestClassifier(random_state=42,

#                                 criterion='gini',

#                                 oob_score=True,

#                                 max_depth=max_depth,

#                                 min_samples_split=min_samples_split,

#                                 min_samples_leaf=min_samples_leaf,

#                                 n_jobs=-1)

#     # define evaluation matrix as objective to return

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV, scoring='roc_auc')

#     accuracy = score.mean()

#     return accuracy

# # create study

# study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# RF_param =study.best_params

RF_param={'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 40, 'min_samples_leaf': 20}

RF_best= RandomForestClassifier(random_state=42,

                                criterion='gini',

                                oob_score=True,

                                **RF_param)
# # Grid Search

# n_estimators=[40,50,60,500] #The number of trees in the forest.

# # criterion=['gini', 'entropy']#The function to measure the quality of a split. 

# max_depth=[3,4,5,7,10]#The maximum depth of the tree.

# min_samples_split = [3,0.01,0.05,0.1] #The minimum num of samples required to split an internal node.

# min_samples_leaf = [3,5,7,0.1] # minimum num of samples required to be a leaf node

# max_features=['auto'] #The number of features to consider when looking for the best split

# hyper={'n_estimators':n_estimators, 'max_depth':max_depth,

#       'min_samples_split':min_samples_split,'max_features':max_features}

# gd=GridSearchCV(estimator=ExtraTreesClassifier(random_state=42,criterion='gini'),

#                 param_grid=hyper,verbose=2,n_jobs=-1,cv=CV,refit=True,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# RF_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_estimators=trial.suggest_int('n_estimators',50,500,50) #The number of trees in the forest.

#     max_depth=trial.suggest_int('max_depth',1,5,1)#The maximum depth of the tree.

#     min_samples_split = trial.suggest_int('min_samples_split',20,200,20) #The minimum num of samples required to split an internal node.

#     min_samples_leaf = trial.suggest_int('min_samples_leaf',20,200,20) # minimum num of samples required to be a leaf node

#     # define classifier

#     clf = ExtraTreesClassifier( random_state=42,

#                                 criterion='gini',

#                                 max_depth=max_depth,

#                                 min_samples_split=min_samples_split,

#                                 min_samples_leaf=min_samples_leaf,

#                                 n_jobs=-1)

#     # define evaluation matrix as objective to return

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV,scoring = 'roc_auc')

#     accuracy = score.mean()

#     return accuracy

# # create study

# study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# ET_params = study.best_params

ET_params = {'n_estimators': 250, 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 20} 

ET_best = ExtraTreesClassifier( random_state=42,

                                criterion='gini',

                                n_jobs=-1,

                               **ET_params)
# # Grid Search parameter tuning

# n_estimators=[5,10,20,50,100]

# base_estimator__C=[0.4,0.8,1,2,5] # Regularization parameter.

# base_estimator__gamma=['scale','auto',0.01,0.1,0.5,1]

# base_estimator__kernel=['rbf']

# # base_estimator__degree=[3,5,7] # Degree of the polynomial kernel 

# hyper={'n_estimators':n_estimators,

#        'base_estimator__C':base_estimator__C,

#        'base_estimator__gamma':base_estimator__gamma,

#        'base_estimator__kernel':base_estimator__kernel}

# gd=GridSearchCV(estimator=BaggingClassifier(base_estimator=SVC(),random_state=42),

#                 param_grid=hyper,verbose=2,n_jobs=-1,cv=CV,refit=True,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# SVMB_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_estimators=trial.suggest_int('n_estimators',50,500,50) #The number of base estimators

#     base_estimator__C=trial.suggest_loguniform('base_estimator__C',1e-5,1e3)# Regularization parameter.

#     base_estimator__gamma = trial.suggest_loguniform('base_estimator__gamma',1e-5,1e3) # kenerl coefficient

#     # define classifier

#     clf = BaggingClassifier(base_estimator=SVC(C=base_estimator__C,

#                                                gamma=base_estimator__gamma,

#                                                kernel = 'rbf'),

#                             random_state=42,

#                             n_jobs=-1)

#     # define evaluation matrix as objective to return

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV,scoring = 'roc_auc')

#     accuracy = score.mean()

#     return accuracy

# # create study

# study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# SVMB_params = study.best_params

SVMB_params = {'n_estimators': 300, 'base_estimator__C': 3.232901108594473, 'base_estimator__gamma': 0.07183110256410177}  

SVMB_best = BaggingClassifier(base_estimator=SVC(kernel = 'rbf',probability=True,C=3.232901108594473,gamma=0.07183110256410177),

                              random_state=42,n_jobs=-1,n_estimators=350)
# n_estimators=list(range(30,300,10))

# learn_rate=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]

# base_estimator__max_depth=[1,2,3]

# hyper={'n_estimators':n_estimators,'learning_rate':learn_rate,

#       'base_estimator__max_depth':base_estimator__max_depth,}

# gd=GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini'),

#                                              random_state=42),param_grid=hyper,verbose=2,

#                                             cv=CV,refit=True,n_jobs=-1,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# ADB_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_estimators=trial.suggest_int('n_estimators',40,500,20) #The number of base estimators

#     learn_rate=trial.suggest_loguniform('learn_rate',1e-5,0.1)

#     base_estimator__max_depth=trial.suggest_int('base_estimator__max_depth',1,5,1)

#     # define classifier

#     clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=base_estimator__max_depth,

#                                                                   criterion='gini'),

#                             learning_rate=learn_rate,

#                             n_estimators=n_estimators,

#                             random_state=42)

#     # define evaluation matrix as objective to return

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV,scoring = 'roc_auc')

#     accuracy = score.mean()

#     return accuracy

# # create study

# study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# ADB_param =study.best_params

ADB_best= AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini',

                                                                  max_depth=2),

                            learning_rate=0.03990900089241141 ,

                            n_estimators=160,

                            random_state=42)
# # Grid Search parameter tunning

# n_estimators=list(range(20,150,10))

# learn_rate=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]

# max_depth=[3,5,8,10]

# min_samples_split=np.linspace(0.1, 0.3, 4)

# criterion=['friedman_mse','mae']

# max_features=['log2','sqrt']

# hyper={'n_estimators':n_estimators,

#        'learning_rate':learn_rate,

#        'max_depth':max_depth,

#        'criterion':criterion,

#        'min_samples_split':min_samples_split,

#        'max_features':max_features}

# gd=GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),

#                 param_grid=hyper,verbose=2,refit=True,n_jobs=-1,

#                cv=CV,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# GDB_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))

# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_estimators=trial.suggest_int('n_estimators',40,500,20) #The number of base estimators

#     learning_rate=trial.suggest_loguniform('learning_rate',1e-5,0.1)

#     max_depth=trial.suggest_int('max_depth',1,5,1)

#     min_samples_split=trial.suggest_int('min_samples_split',20,200,20)

#     min_samples_leaf=trial.suggest_int('min_samples_leaf',20,200,20)

#     # define classifier

#     clf = GradientBoostingClassifier(max_depth=max_depth,

#                                      min_samples_split=min_samples_split,

#                                      min_samples_leaf=min_samples_leaf,

#                                      learning_rate=learning_rate,

#                                      n_estimators=n_estimators,

#                                      subsample=0.8,

#                                      n_iter_no_change=10,

#                                      random_state=42)

#     # define evaluation matrix as objective to return

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV,scoring = 'roc_auc')

#     accuracy = score.mean()

#     return accuracy

# # create study

# study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# GDB_param = study.best_params

GDB_param = {'n_estimators': 420, 'learning_rate': 0.08199591231901683, 

             'max_depth': 3, 'min_samples_split': 140, 'min_samples_leaf': 20} 

GDB_best= GradientBoostingClassifier(**GDB_param,

                                     subsample=0.8,

                                     n_iter_no_change=10,

                                     random_state=42)
# # Grid Search hyperparameter tuning

# #tree features

# max_depth=[5,8]

# # subsamples=[0.8,0.5]# the fraction of observations to be randomly samples for each tree.

# colsample_bytree=[0.5,0.8]# the fraction of columns to be randomly samples for each tree.

# gamma = [0.1,0.3,0.5] #Minimum loss reduction required to make a further partition on a leaf node of the tree.

# min_child_weight =[1,3,5] # Minimum sum of instance weight(hessian) needed in a child.

# # boosting features

# n_estimators=[25,50,90,120] #Number of gradient boosted trees(rounds).

# learning_rate =[0.01, 0.25, 0.1]

# # learning_rate =[0.01, 0.025, 0.05, 0.1, 0.15, 0.2]# Boosting learning rate (xgb’s “eta”), typically in [0.01,0.2].

# booster=['dart'] # tree booster always better than linear.

# # regularization

# #reg_alpha=[1e-5, 1e-2, 0.1] # L1 regularization term on weights

# reg_lambda =[1e-5, 1e-2] # L2 regularization term on weights

# hyper={'n_estimators':n_estimators,

#        'learning_rate':learn_rate,

#        'booster':booster,

#        'max_depth':max_depth,

#        'colsample_bytree':colsample_bytree,

#        'gamma':gamma,

#        'min_child_weight':min_child_weight,

#        'reg_lambda':reg_lambda} 

# gd=GridSearchCV(estimator=XGBClassifier(verbosity=1,n_jobs =-1,random_state=42),

#                 param_grid=hyper,verbose=2,refit=True,n_jobs=-1,

#                cv=CV,scoring = 'accuracy')

# gd.fit(X,Y)

# print('Best evaluation score:{:.6f}'.format(gd.best_score_))

# XGB_best=gd.best_estimator_

# print('Best parameters:{}'.format(gd.best_params_))

# # hyperparameter tuning using optuna framework

# def objective(trial):

#     # define sample space and distibution of parameters

#     max_depth = trial.suggest_int("max_depth", 1,5,1)

#     n_estimators = trial.suggest_int("n_estimators", 40, 500, 20)

#     booster = trial.suggest_categorical('booster',['gbtree','gblinear','dart'])

#     min_child_weight = trial.suggest_int('min_child_weight',5,105,10)

#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-6,1e-3)

#     gamma = trial.suggest_loguniform('gamma', 0.00001, 100)

#     reg_alpha = trial.suggest_loguniform('reg_alpha',1e-3,1e2) # L1 regularization term on weights.

#     reg_lambda = trial.suggest_loguniform('reg_lambda',1e-3,1e2) # L2 regularization term on weights.

#     colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree',0.4,0.8,0.2) # sub-features to use 

#     subsample = trial.suggest_discrete_uniform('subsample',0.8,1.0,0.1) # subsamples to use

#     # define classifier

#     clf = XGBClassifier(objective='binary:logistic',

#                         booster = booster,

#                         subsample=subsample,

#                         colsample_bytree=colsample_bytree,

#                         learning_rate=learning_rate,

#                         n_estimators=n_estimators,

#                         max_depth=max_depth,

#                         gamma=gamma,

#                         reg_alpha=reg_alpha,

#                         reg_lambda=reg_lambda,

#                         n_jobs=-1,

#                         random_state=42)

#     # defin evaluation matrix

#     score = cross_val_score(clf, X, Y, n_jobs=-1, cv=CV,scoring = 'roc_auc')

#     accuracy = score.mean()

#     return accuracy

# # define optimizor's direction and sample algorithms 

# study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())

# # run optimizor with n_trials to find best parameters

# s=study.optimize(objective, n_trials=500, n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))

    
# XGB_param = study.best_params

XGB_param = {'max_depth': 5, 'n_estimators': 500, 'booster': 'dart', 

             'min_child_weight': 45, 'learning_rate': 0.00028818174062883895, 

             'gamma': 0.0701754028803822, 'reg_alpha': 0.09673762960851098, 'reg_lambda': 0.020973617864068886, 

             'colsample_bytree': 0.6000000000000001, 'subsample': 1.0} 

XGB_best= XGBClassifier(**XGB_param,

                        objective='binary:logistic',

                        n_jobs=-1,

                        random_state=42)
@ignore_warnings(category=ConvergenceWarning)

def model_eval():

    acc_mean = []

    std = []

    acc = []

    classifiers=['Svm','LR','KNN','DT','MLP','RF','ADB','ET','XGB','GDB','SVMB']

    models=[SVM_best,LR_best,KNN_best,DT_best,MLP_best,RF_best,ADB_best,ET_best,XGB_best,GDB_best,SVMB_best]

    for model in models:

        cv_result = cross_val_score(model,X,Y, cv = 5,scoring = "roc_auc")

        acc_mean.append(cv_result.mean())

        std.append(cv_result.std())

        acc.append(cv_result)

    performance_df=pd.DataFrame({'CV_Mean':acc_mean,'Std':std},index=classifiers)

    return performance_df



performance_df = model_eval()

performance_df.sort_values(by=['CV_Mean'],ascending=False)
votingC = VotingClassifier(estimators=[('XGB', XGB_best),

                                       ('RF',RF_best),

                                       ('ADB',ADB_best),

                                       ('ET', ET_best),

                                       ('GDB',GDB_best),

                                       ('DT',DT_best)],

                           voting='hard', n_jobs=-1,verbose=True)

votingC.fit(X,Y)



cv_result = cross_val_score(votingC,X,Y, cv = 5,scoring = "accuracy")

vot_acc = cv_result.mean()

vot_std = cv_result.std()

performance_df.loc['Voting'] = {'CV_Mean':vot_acc, 'Std':vot_std}

performance_df.sort_values(by=['CV_Mean'],ascending=False)
stacking = StackingClassifier(estimators=[('XGB', XGB_best),

                                       ('RF',RF_best),

                                       ('ADB',ADB_best),

                                       ('ET', ET_best),

                                       ('GDB',GDB_best),

                                       ('DT',DT_best)],

                             final_estimator=LogisticRegression(),

                             cv=5,

                             n_jobs=-1)

cv_result = cross_val_score(stacking,X,Y, cv = 5,scoring = "accuracy")

stk_acc = cv_result.mean()

stk_std = cv_result.std()

performance_df.loc['Stacking'] = {'CV_Mean':stk_acc, 'Std':stk_std}

performance_df.sort_values(by=['CV_Mean'],ascending=False)
rf = XGB_best

rf.fit(X,Y)

features = pd.DataFrame()

features['feature'] = X.columns

features['importance'] = rf.feature_importances_

features.sort_values(by=['importance'], ascending=False, inplace=True)

features.set_index('feature', inplace=True)

features
# split data

X_tr, X_te, Y_tr, Y_te = train_test_split(X,Y)

# fit model

clf = RF_best.fit(X_tr,Y_tr)

# find misclassified data in test set

mis_df = X_te[np.logical_xor(Y_te,clf.predict(X_te))] 

# show orginal data

mis_df=train.loc[mis_df.index]

mis_df.describe()
titanic_raw_train.loc[mis_df.index]
performance_df.CV_Mean.idxmax()
sub_model = RF_best
features = pd.DataFrame()

features['feature'] = X.columns

features['importance'] = sub_model.feature_importances_

features.sort_values(by=['importance'], ascending=False, inplace=True)

features.set_index('feature', inplace=True)

features
clf = RF_best.fit(X,Y)

sub = clf.predict(X_test)

sub_pd = pd.DataFrame({'PassengerId':titanic_raw_test.PassengerId,'Survived':sub})

sub_pd.to_csv('submit.csv' ,index=False)