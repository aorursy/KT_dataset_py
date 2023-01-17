# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, GridSearchCV, KFold

import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost import plot_importance
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
data = pd.concat([train_data,test_data],axis=0,ignore_index=True)
print('train shape ', train_data.shape, '; test shape ',test_data.shape, '; all data',data.shape)
data.tail()
data.describe()
data.describe(include=['O'])
Pclass = data.Pclass
Survived = data.Survived
FamilySize = pd.Series(data.SibSp + data.Parch, name='FamilySize')
print(FamilySize.value_counts())
FamilySize = FamilySize.map({0:0,1:1,2:1,3:2,4:2,5:2,6:4,7:4,8:4,9:4,10:4})
print(FamilySize.value_counts())
SibSp = pd.Series(data.SibSp, name='SibSp')
Parch = pd.Series(data.Parch, name='Parch')
#largeSibSp = pd.Series((data.SibSp>2).astype('int'),name='largeSibSp')
#largeParch = pd.Series((data.Parch>2).astype('int'),name='largeParch')
Sex = data.Sex.map({'male':1,'female':0})
Embarked = data.Embarked
Embarked_ctg = pd.get_dummies(Embarked,prefix='Embarked')
Title = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
Title.iloc[1305]
Title = Title.map({'Capt':0,'Col':0,'Major':0,'Rev':0,
           'Countess':1,'Don':1,'Dona':1,'Dr':1,'Jonkheer':1,'Lady':1,'Sir':1,
           'Miss':2,'Mlle':2,
           'Mme':3,'Mrs':3,'Ms':3,
           'Mr':4,'Master':5})
Title_ctg = pd.get_dummies(Title,prefix='Title')
tmp = pd.DataFrame({'Survived':Survived,'Title':Title})
tmp.groupby('Title').Survived.value_counts()
#print(data.groupby(['Embarked','Pclass']).Fare.mean())
#print(data.describe())
Fare = data.Fare.copy(deep=True)
print(data.loc[data.Fare.isnull()])
tmp = pd.DataFrame({'Pclass':Pclass,'Fare':Fare})
guess_Fare = tmp.groupby(['Pclass']).Fare.median()
print('\n',guess_Fare,'\n')
for i in tmp.Pclass.unique():
    Fare.loc[(Fare.isnull()) & (tmp.Pclass == i)] = guess_Fare.loc[i]
print(Fare.loc[data.Fare.isnull()])
Fare.loc[ Fare <= 8] = 0
Fare.loc[(Fare > 8)  & (Fare <= 16),] = 1
Fare.loc[(Fare > 16) & (Fare <= 32),] = 2
Fare.loc[(Fare > 32) & (Fare <= 48),] = 3
Fare.loc[(Fare > 48) & (Fare <= 100)] = 4
Fare.loc[ Fare > 100] = 5
tmp = pd.DataFrame({'Survived':Survived,'Fare':Fare})
print(tmp.groupby('Survived').Fare.value_counts().sort_index())
Fare -= Fare.min()
Fare /= Fare.max()
Surname = data.Name.str.extract('(^\D+),', expand=False)
tmp = pd.DataFrame({'Surname':Surname,'name':data.Name})
tmp.head(100)
#(Surname.isna()).value_counts()
Surname.value_counts()
N = 1309
hasFamilySurvived = pd.Series(np.zeros([N]),name='hasFamilySurvived')

for i in range(0,N):
    if (FamilySize.iloc[i]==0):
        continue
    thisname = Surname.iloc[i]
    index = Survived.loc[(tmp.Surname==thisname) & (Survived==1)].index
    if (i in index):
        index = index.drop(i)
    if (index.empty==False):
        hasFamilySurvived.iloc[i]=1

print(hasFamilySurvived[hasFamilySurvived==1].index)
hasFamilySurvived.value_counts()
Cabin = data.Cabin.fillna('N')
Cabin = Cabin.str.extract('(^\D)+', expand=False)
tmp = pd.DataFrame({'Survived':Survived,'Cabin':Cabin})
print(tmp.groupby('Cabin').Survived.mean().sort_values())
Cabin = Cabin.map({'A':1,'B':4,'C':4,'D':4,'E':6,'F':3,'G':2,'N':0,'T':-1})
isalone = pd.Series(((data.Parch + data.SibSp)==0).astype('int'),name='isalone')
tmp = pd.DataFrame({'Survived':Survived,'isalone':isalone})
tmp.groupby('Survived').isalone.value_counts()
Age = data.Age.copy(deep=True)
Age.loc[(data.Age <= 8) & (data.Age.notna())] = 0
Age.loc[(data.Age > 8)  & (data.Age <= 16) & (data.Age.notna())] = 1
Age.loc[(data.Age > 16) & (data.Age <= 32) & (data.Age.notna())] = 2
Age.loc[(data.Age > 32) & (data.Age <= 48) & (data.Age.notna())] = 3
Age.loc[(data.Age > 48) & (data.Age <= 64) & (data.Age.notna())] = 4
Age.loc[(data.Age > 64) & (data.Age.notna())] = 5
tmp = pd.DataFrame({'Survived':Survived,'Age':Age})
print(tmp.groupby('Survived').Age.value_counts().sort_index())
#print(data.Age.loc[data.Age.isnull()])
#print(Age.value_counts().sort_index())

Age_X = pd.concat([Sex,Pclass,Fare,Title_ctg,Embarked_ctg,FamilySize,SibSp,Parch,isalone],axis=1)
Age_Y = Age.copy(deep=True)
Age_X_train = Age_X.loc[data.Age.notna()]
Age_Y_train = Age_Y.loc[data.Age.notna()]
Age_X_test = Age_X.loc[data.Age.isnull()]
#print(data.Age.loc[data.Age.isnull()])
print(Age_X_train.head())
#print(Age_Y_train.value_counts().sort_index())
#print(Age_X_test.describe())

AgeModel = SVC(C=5)
AgeModel.fit(Age_X_train, Age_Y_train)
print('Age training score ', round(AgeModel.score(Age_X_train, Age_Y_train) * 100, 2))
Age_Y_pred = AgeModel.predict(Age_X_test)
Age.loc[data.Age.isnull()] = Age_Y_pred

Age -= Age.min()
Age /= Age.max()

tmp = data.copy(deep=True)
tmp.loc[data.Age.isnull(),'Age'] = Age
#tmp.loc[data.Age.isnull()]
Ticket = data.Ticket.str.extract('(\d+)$', expand=False)
Ticket.fillna(0,inplace=True)
Ticket = (Ticket.astype('int')/10000).astype('int')
tmp = pd.DataFrame({'Survived':Survived,'Ticket':Ticket})
tmp.groupby('Ticket').Survived.mean().sort_values()
Ticket = Ticket.map({34:-1,4:-1,5:-1,21:-1,32:-1,26:-1,
            31:0,38:0,35:0,37:0,310:0,
            36:1,0:1,6:1,2:1,39:1,23:1,
            25:2,24:2,11:2,33:2,3:2,1:2,22:2})
#Ticket.loc[Ticket==34]=-1
#Ticket.loc[(Ticket==11)|(Ticket==24)|(Ticket==11)]
data_pruned = pd.concat([Survived,Pclass,Age,Sex,Fare,Title_ctg,Embarked_ctg,
                         FamilySize,SibSp,Parch,hasFamilySurvived,Cabin,Ticket],axis=1)
data_pruned.info()
data_pruned.head()
import seaborn as sns
sns.set(rc={'figure.figsize':(12,10)})
sns.heatmap(data_pruned.corr())
train = data_pruned.iloc[0:891]
test = data_pruned.iloc[891:1309]
train_X = train.loc[:,'Pclass':]
train_Y = train.loc[:,'Survived']
test_X = test.loc[:,'Pclass':]
print('shapes: ',train_X.shape,train_Y.shape,test_X.shape)
model = XGBClassifier()
print(model.get_params())
param_grid = {'n_estimators': [100],
             'max_depth': [6,8],
             'learning_rate': [0.002],
             'colsample_bytree': [0.5,0.8],
             'gamma': [1,2,3],
             'min_child_weight': [1,2,3],
             'subsample': [0.8,1]}
np.random.seed(42)
cv = KFold(n_splits=4, random_state=42, shuffle=True)

tuned_model = GridSearchCV(model, param_grid=param_grid, cv=cv,scoring='accuracy', n_jobs=8)
tuned_model.fit(train_X.values, train_Y.values)
print('final params ', tuned_model.best_params_)
print('best score ', tuned_model.best_score_)
model = tuned_model.best_estimator_
pred_Y_mean = np.zeros([test_X.shape[0]])
print(model.get_params())

cv = KFold(n_splits=6, random_state=24, shuffle=True)
for tidx, vidx in cv.split(train):
    print('predict with KFold')
    train_cv = train.iloc[tidx]
    val_cv = train.iloc[vidx]
    train_X = train_cv.loc[:,'Pclass':]
    train_Y = train_cv.loc[:,'Survived'] 
    val_X = val_cv.loc[:,'Pclass':]
    val_Y = val_cv.loc[:,'Survived'] 
    model.fit(train_X,train_Y,eval_set=[(val_X,val_Y)],early_stopping_rounds=50,verbose = 5)
    pred_Y = model.predict(test_X)
    pred_Y_mean = pred_Y_mean + pred_Y

pred_Y_mean = ((pred_Y_mean/8.0)>0.5).astype('int')
feature_importance = pd.Series(model.feature_importances_,index=train.columns.values[1:]).sort_values(ascending=False)
print('feature_importance\n')
print(feature_importance)  

final_pred = pd.DataFrame(pred_Y_mean, columns=['Survived'],index=test_X.index.values+1)
final_pred.index.name = 'PassengerId'
final_pred.to_csv('submission.csv')
#save results for submission
train_X = train.loc[:,'Pclass':]
train_Y = train.loc[:,'Survived']
train_Y_pred = model.predict(train_X)
train_Y_pred_sr = pd.Series(train_Y_pred,name='train_Y_predict',index=train_Y.index)
#false positive
train_data.loc[(train_Y==0)&(train_Y_pred_sr==1)]
#True positive
train_data.loc[(train_Y==1)&(train_Y_pred_sr==1)]
#True negative
train_data.loc[(train_Y==0)&(train_Y_pred_sr==0)]
#false negative
train_data.loc[(train_Y==1)&(train_Y_pred_sr==0)]