# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('seaborn-colorblind')
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',category=DeprecationWarning)

from IPython.display import Image
from collections import Counter
from scipy.stats.mstats import normaltest
from sklearn.preprocessing import StandardScaler
# Before we dive into the main game, let's take a look at the training set.

df = pd.read_csv("../input/train.csv")
# check the dimension and first five rows of training set

print("Dimension:",df.shape)
df.head()
# Let's see if there is any missing value in dataset and useful information for further analysis
df.info()
df.isnull().sum()
df.Age.fillna(df.Age.median(),inplace=True)
df.describe()

df['Died'] = (df['Survived'] == 0).map(lambda x: 1 if x == True else 0)
df.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',stacked=True,colors=['red','grey'],figsize=(13,5))
plt.legend(frameon=False)

ax = plt.gca()
for item in ax.xaxis.get_ticklabels():
    item.set_rotation(0)
    
plt.ylabel("Passengers")
df.groupby('Sex').agg('mean')[['Survived']]

a = sns.factorplot(x='Sex',y='Age',hue='Survived',data=df,split=True,palette={0:"grey",1:'red'},size=6,kind='violin',aspect=2)

print(normaltest(df.Fare))
fig, (axis1,axis2) = plt.subplots(ncols=2, figsize=(17,6))
gs = gridspec.GridSpec(1,2,width_ratios=[1,3])
ax1 = plt.subplot(gs[0])
sns.kdeplot(df['Fare'],color='red',shade=True,ax=ax1)

ax2 = plt.subplot(gs[1])
ax2.hist([df[df['Survived']==1]['Fare'],df[df['Survived']==0]['Fare']],stacked=True,bins=40,color=['red','grey'],label=['Survived','Died'])

plt.xlabel("Fare")
plt.ylabel("passengers")
plt.legend()


plt.figure(figsize=(10,5))
ss = sns.barplot(x='SibSp',y='Survived',data=df,ci=None)
ss = ss.set(ylabel='prob of Survival')

plt.figure(figsize=(10,5))
pc = sns.barplot(x='Parch',y='Survived',data=df,ci=None)
pc = pc.set(ylabel='prob of Survival')


# Let's check if there is any null value in the feature.
df.Embarked.isnull().sum()
df.Embarked.fillna(df.Embarked.mode()[0],inplace=True)
print(Counter(df.Embarked))
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
sns.barplot(x='Embarked', y='Survived',data=df, ci=None,ax=axis1)
axis2.set(ylabel='prob of Survival')

em = sns.factorplot(x='Embarked',y='Fare',hue='Survived',data=df,kind='violin',palette={0:'grey',1:'red'}, size=6,split=True,ax=axis2)
plt.close(2)

fig, axs = plt.subplots(ncols=2, figsize=(15,6))
sns.barplot(x='Pclass',y='Fare',data=df,ax=axs[0])
ps = sns.barplot(x='Pclass',y='Survived',data=df,ax=axs[1])
ps.set(ylabel='prob of Survival')
sns.despine(ax=axs[1], right=True, left=False)
sns.despine(ax=axs[0], right=True, left=False)

df[['Fare','Pclass','Survived']].corr()

df['Family_Size'] = df['Parch'] + df['SibSp'] + 1

fs = sns.factorplot(x='Family_Size',y='Survived',kind='bar',data=df,size=7)
fs = fs.set(ylabel='prob of Survival')

print("number of missing values:",df.Cabin.isnull().sum())
df.Cabin.describe()
# Extract only the first letter if it is not nan else replace it with 'X'
df['Cabin'].fillna('X',inplace=True)
df['Cabin'] = df['Cabin'].map(lambda x:x[0])
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(16,5))

# Distribution of cabin locations.
sns.factorplot(x='Cabin',data=df,size=6,kind='count',ci=None, order=['X','A','B','C','D','E','F','G','T'],ax=axis1)

# Survival rate according to cabin locations.
sns.factorplot(x='Cabin',y= 'Survived',data=df,size=6,kind='bar',order=['X','A','B','C','D','E','F','G','T'],ax=axis2)
axis2.set(ylabel='prob of Survival')
plt.close(2)
plt.close(3)

df['Ticket'] = df['Ticket'].map(lambda x:x.replace('/',''))
df['Ticket'] = df['Ticket'].map(lambda x:x.replace('.',''))
df['Ticket'] = df['Ticket'].map(lambda x: 'XX' if x.isdigit() else x.split(' ')[0])

t = sns.factorplot(x='Ticket',data=df,size=6,kind='count',ci=None)
t = t.set_xticklabels(rotation=90)

df['Title'] = df['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
df.Title.unique()
n = sns.factorplot(kind='count',x='Title',data=df,size=7)
n = n.set_xticklabels(rotation=45)
pd.crosstab(df['Survived'],df['Title'])
df['Title'].replace(['Miss','Mlle','Mme','Mrs','Ms'],'Ms',inplace=True)
df['Title'].replace({'Mr':0,'Ms':1,'Master':2},inplace=True)
df['Title'] = df['Title'].map(lambda x:3 if type(x)==str else x)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(16,5))

# Distribution of titles.
sns.factorplot(x='Title',kind='count',size=6,data=df,ax=axis1)
axis1.set_xticklabels(['Mr','Ms,Miss,Mme,Mlle,Mrs', 'Master','Minor'])

# Survival rate categorized by titles.
sns.factorplot(x='Title',y='Survived',data=df, kind='bar',size=6,ax=axis2)
axis2.set_xticklabels(['Mr','Ms,Miss,Mme,Mlle,Mrs', 'Master','Minor'])
axis2.set(ylabel='prob of Survival')
plt.close(2)
plt.close(3)

plt.figure(figsize=(15,5))
sns.heatmap(df[['Survived','Parch','Fare','Age','SibSp']].corr(),annot=True,linewidth=3,linecolor='yellow')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

y_id = test.iloc[:,0]
y = train.iloc[:,1]

last_train_index = len(train)
first_test_index = len(train)

comb = pd.concat([train,test]).reset_index()
comb.drop(['PassengerId','Survived','index'],inplace=True, axis=1)

print("TrainSet:",train.shape,"TestSet:",test.shape,"CombinedSet:",comb.shape)

# Let's see if there is any missing value in Embarked column.
Counter(comb.Embarked)
# Fill the missing values with the most frequent 'S'.
comb['Embarked'].fillna(comb.Embarked.mode()[0],inplace=True)

# Dummy encoding.
embarked_dummy = pd.get_dummies(comb['Embarked'],prefix='EM')

# Concatenate the combined with dummy variables.
comb = pd.concat([comb,embarked_dummy],axis=1)

# In this time, create new feature in combined dataset
comb['Family_Size'] = comb['Parch'] + comb['SibSp'] + 1

# Categorize according to the size of family
comb['FS_single'] = comb['Family_Size'].map(lambda x: 1 if x==1 else 0)
comb['FS_small'] = comb['Family_Size'].map(lambda x: 1 if x==2 else 0)
comb['FS_medium'] = comb['Family_Size'].map(lambda x: 1 if 3<=x<=4 else 0)
comb['FS_large'] = comb['Family_Size'].map(lambda x: 1 if x>=5 else 0)

# convert to male:1, female:0
comb['Sex'] = comb.Sex.map(lambda x: 1 if x =='male' else 0)


print('Missing Age Values :',comb.Fare.isnull().sum())
comb['Fare'].fillna(comb['Fare'].median(),inplace=True)

# Create new dummy variables for each Pclass
pclass_dummy = pd.get_dummies(comb['Pclass'],prefix='PC')

# Concatenate the combined with dummy variables.
comb = pd.concat([comb,pclass_dummy],axis=1)

print("number of missing values:",comb.Cabin.isnull().sum())
comb.Cabin.describe()
# If there are any missing values, fill them with 'X'.
comb['Cabin'].fillna('X',inplace=True)

# Extract only the first letter.
comb['Cabin'] = comb['Cabin'].map(lambda x:x[0])

# Create new dummy variables for each refined values.
cabin_dummy = pd.get_dummies(comb['Cabin'],prefix='CB')

# Concatenate the combined with dummy variables.
comb = pd.concat([comb,cabin_dummy],axis=1)

comb.Ticket.head()
# If the ticket is only a digit, replace them with XX else extract each prefix.
comb['Ticket'] = comb['Ticket'].map(lambda x:x.replace('/',''))
comb['Ticket'] = comb['Ticket'].map(lambda x:x.replace('.',''))
comb['Ticket'] = comb['Ticket'].map(lambda x: 'XX' if x.isdigit() else x.split(' ')[0])

# Create new dummy variables for each refined values.
ticket_dummy = pd.get_dummies(comb['Ticket'],prefix='TK')

# Concatenate the combined with dummy variables.
comb = pd.concat([comb,ticket_dummy],axis=1)

comb['Title'] = comb['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
Counter(comb['Title'])
# Convert equivalent female prefixes to Ms
comb['Title'].replace(['Miss','Mlle','Mme','Mrs','Ms'],'Ms',inplace=True)

# Convert titles to numeric values
comb['Title'].replace({'Mr':0,'Ms':1,'Master':2},inplace=True)
comb['Title'] = comb['Title'].map(lambda x:3 if type(x)==str else x)

# Create new dummy variables.
title_dummy = pd.get_dummies(comb['Title'],prefix='TLE')

# Concatenate the combined with dummy variables.
comb = pd.concat([comb,title_dummy],axis=1)

print('Missing Age Values :',comb.Age.isnull().sum())
print('MissingAgeValues / Total :', comb.Age.isnull().sum()/len(comb))
plt.figure(figsize=(9,6))

# Exclude PassengerID and Survived
col = ['Pclass','Sex','SibSp','Age','Parch','Title']

# Heatmap
a = sns.heatmap(comb[col].corr(),annot=True,linewidth=3,cmap='RdBu_r')
comb['old_age'] = comb['Age']

comb['Age'] = comb.groupby(['Pclass','SibSp','Parch','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
comb['Age'].fillna(comb['Age'].median(),inplace=True)
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
oa = sns.distplot(comb['old_age'].dropna(),ax=ax1,kde=True,bins=70)
a = sns.distplot(comb['Age'].dropna(),ax=ax2,kde=True,bins=70,color='red')
ax1.set_xlabel('old_Age')
ax1.set_ylabel('Count')
ax2.set_xlabel('Age')
ax2.set_ylabel('Count')

comb.drop(['Cabin','Embarked','Name','Ticket','Title','old_age'],axis=1,inplace=True)

# Split the combined dataset into two: train and test

X_train = comb[:last_train_index]
X_test = comb[first_test_index:]
# Import the libraries we will be using

from sklearn.model_selection import GridSearchCV,StratifiedKFold,learning_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

kfold = StratifiedKFold(n_splits=20, random_state = 2018)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

GBC = GradientBoostingClassifier(random_state = 2018)
gb_param_grid = {
              'n_estimators' : [543],
              'learning_rate': [0.05],
              'max_depth': [4],
              'min_samples_leaf': [3],
              'subsample':[0.4]
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsGBC.fit(X_train,y)
GBC_best = gsGBC.best_estimator_
print(gsGBC.best_params_)

# cv = 20, random_state = 2018 (Scaler)
# ** {'learning_rate': 0.05, 'max_depth': 4, 'min_samples_leaf': 3, 'n_estimators': 543, 'subsample': 0.4}
# {'learning_rate': 0.05, 'max_depth': 4, 'min_samples_leaf': 3, 'n_estimators': 540, 'subsample': 0.4} - 80.861
gsGBC.best_score_
#y_submission = gsGBC.predict(X_test)

DTC = DecisionTreeClassifier(random_state = 2018)

adaDTC = AdaBoostClassifier(DTC, random_state = 2018)

ada_param_grid = {"base_estimator__criterion" : ["entropy"],
                  "base_estimator__splitter" :   ["best"],
                  "base_estimator__max_features":[None],
                  "base_estimator__min_samples_split":[0.3],
                  "algorithm" : ["SAMME.R"],
                  "n_estimators" :[1100],
                  "learning_rate":  [0.005]
                 }

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1,verbose = 1)
gsadaDTC.fit(X_train,y)
ada_best = gsadaDTC.best_estimator_
print(gsadaDTC.best_params_)

## cv = 20, random_state = 2018
# * {'algorithm': 'SAMME.R', 'base_estimator__criterion': 'entropy', 'base_estimator__max_features': None, 'base_estimator__min_samples_split': 0.3, 'base_estimator__splitter': 'best', 'learning_rate': 0.005, 'n_estimators': 1000}
# ** {'algorithm': 'SAMME.R', 'base_estimator__criterion': 'entropy', 'base_estimator__max_features': None, 'base_estimator__min_samples_split': 0.3, 'base_estimator__splitter': 'best', 'learning_rate': 0.005, 'n_estimators': 1100} - 75.119

gsadaDTC.best_score_
#y_submission = gsadaDTC.predict(X_test)

gammas = [0.0019]
Cs = [165]
weight = [{1:1}]
kernels = ['rbf']
param_grid = dict(C=Cs,gamma=gammas,class_weight=weight, kernel=kernels)
gsSVMC = GridSearchCV(SVC(probability=True, random_state = 2018),param_grid=param_grid,cv=kfold,scoring="accuracy",n_jobs=-1,verbose=1)
gsSVMC.fit(X_train,y)
gsSVMC_best = gsSVMC.best_estimator_
print(gsSVMC.best_params_)

## cv = 20, random_state = 2018
# **** {'C': 165, 'class_weight': {1: 1}, 'gamma': 0.0019, 'kernel': 'rbf'} - 77.990
# *** {'C': 170, 'class_weight': {1: 1}, 'gamma': 0.002, 'kernel': 'rbf'} - 77.990
# ** {'C': 150, 'class_weight': {1: 1}, 'gamma': 0.002, 'kernel': 'rbf'} - 77.990
gsSVMC.best_score_
#y_submission = gsSVMC.predict(X_test)

RFC = RandomForestClassifier(random_state = 2018)

rf_param_grid = {"max_depth": [7],
              "max_features": [31],
              "min_samples_leaf": [8],
              "n_estimators" :[349]
                }

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsRFC.fit(X_train,y)
RFC_best = gsRFC.best_estimator_
print(gsRFC.best_params_)

# cv = 20, random_state = 2018
# * {'max_depth': 7, 'max_features': 31, 'min_samples_leaf': 8, 'n_estimators': 349} - 80.861

gsRFC.best_score_
#y_submission = gsRFC.predict(X_test)

ExtC = ExtraTreesClassifier(random_state = 2018)

ex_param_grid = {
              "n_estimators" :[311],          
              "max_depth": [6],
              "max_features": ['auto'],
              "min_samples_leaf": [1],
              "bootstrap": [True]
                }

                
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsExtC.fit(X_train,y)
ExtC_best = gsExtC.best_estimator_
print(gsExtC.best_params_)
                
## cv = 20, random_state = 2018
# **{'bootstrap': True, 'max_depth': 6, 'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 311} - 78.947
# * {'bootstrap': True, 'max_depth': 6, 'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 315} - 78.947

gsExtC.best_score_
#y_submission = gsExtC.predict(X_test)

XGB = XGBClassifier(random_state = 2018, early_stopping_rounds = 500)

xg_param_grid = {
              'n_estimators' : [465],
              'learning_rate': [0.1],
              
              'max_depth': [11],
              'min_child_weight':[10],
              
              'gamma': [0.1],
              
              'subsample':[0.5],
              'colsample_bytree':[0.6]
}

gsXGB = GridSearchCV(XGB,param_grid = xg_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
gsXGB.fit(X_train,y)
print(gsXGB.best_params_)

## cv = 20, random_state = 2018
# * {'colsample_bytree': 0.6, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 10, 'n_estimators': 465, 'subsample': 0.5} - 78.947
# ** {'colsample_bytree': 0.6, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 10, 'n_estimators': 465, 'subsample': 0.5} - 78.947

XGB_best = gsXGB.best_estimator_
gsXGB.best_score_
#y_submission = gsXGB.predict(X_test)

votingC = VotingClassifier(estimators = [('rfc',RFC_best), ('extc',ExtC_best), ('xgb',XGB_best)
                                        ],weights = [2,1,1], voting='hard',n_jobs=-1)
votingC = votingC.fit(X_train,y)

y_submission = votingC.predict(X_test)


# rfc + extc - 80.861
# W :rfc + extc + xgb - 81.339, weights:[1,1,1] - hard
# WW : rfc + extc + xgb - 81.818 weights:[2,1,1] - hard
print('Training Score:',votingC.score(X_train,y))
def lcplot(model,X,y,title,train_sizes=np.linspace(0.1,1.0,5)):
    
    train_sizes, train_scores, test_scores = learning_curve(model,X,y,cv=kfold,n_jobs=-1,train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training size")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha = 0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha = 0.1, color='g')
    plt.plot(train_sizes,train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes,test_scores_mean, 'o-', color='g', label='CV score')
    plt.legend(loc='best')
    return plt

g = lcplot(gsGBC.best_estimator_,X_train,y,"GradientBoost")
g = lcplot(gsadaDTC.best_estimator_,X_train,y,"AdaBoost")
g = lcplot(gsSVMC.best_estimator_,X_train,y,"SVM")
g = lcplot(gsRFC.best_estimator_,X_train,y,"RandomForest")
g = lcplot(gsExtC.best_estimator_,X_train,y,"ExtraTrees")
g = lcplot(gsXGB.best_estimator_,X_train,y,"XGBoost")

my_submission = pd.DataFrame({'PassengerId':y_id,'Survived':y_submission})
my_submission.to_csv('submission.csv',index=False)  