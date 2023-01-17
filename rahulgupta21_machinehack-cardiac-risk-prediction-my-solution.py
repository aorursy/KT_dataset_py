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
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier, BaggingClassifier, VotingClassifier
train = pd.read_csv(r'/kaggle/input/cardiac-risk-prediction/Train.csv')

test = pd.read_csv(r'/kaggle/input/cardiac-risk-prediction/Test.csv')

sample = pd.read_excel(r'/kaggle/input/cardiac-risk-prediction/Sample_Submission.xlsx')
train.head()
test.head()
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
for c in train.columns :

    print(train[c].value_counts())
sns.countplot(train.UnderRisk)
for c in train.columns[:-1]:

    print(train.groupby(c)['UnderRisk'].value_counts(),'\n\n')
ind_no = []

train = train.groupby('Family_history').get_group(1).reset_index(drop = True)

d = test[test.Family_history==0].index

ind_no.extend(d)

del train['Family_history']

del test['Family_history']

test.drop(index = d, inplace = True)

len(ind_no)
train = train.groupby('CABG_history').get_group(0).reset_index(drop = True)

d = test[test.CABG_history==1].index.tolist()

del train['CABG_history']

del test['CABG_history']

ind_no.extend(d)

test.drop(index = d, inplace = True)

len(ind_no)
train[train.History_of_preeclampsia==1]
train.drop(index=304, inplace = True)

train.reset_index(drop = True, inplace = True)
train = train.groupby('History_of_preeclampsia').get_group(0).reset_index(drop = True)

d = test[test.History_of_preeclampsia==1].index.tolist()

del train['History_of_preeclampsia']

del test['History_of_preeclampsia']

ind_no.extend(d)

test.drop(index = d, inplace = True)

len(ind_no)
train = train.groupby('Chain_smoker').get_group(0).reset_index(drop = True)

d = test[test.Chain_smoker==1].index.tolist()

del train['Chain_smoker']

del test['Chain_smoker']

ind_no.extend(d)

test.drop(index = d, inplace = True)

len(ind_no)
del train['Metabolic_syndrome']

del test['Metabolic_syndrome']
train.head(1)
train['disease'] = train.Gender.astype('str')+train.Consumes_other_tobacco_products.astype('str')+train.Use_of_stimulant_drugs.astype('str')+train['HighBP'].astype('str')+train.Obese.astype('str')+train.Diabetes.astype('str')+train.Respiratory_illness.astype('str')

test['disease'] = test.Gender.astype('str')+test.Consumes_other_tobacco_products.astype('str')+test.Use_of_stimulant_drugs.astype('str')+test['HighBP'].astype('str')+test.Obese.astype('str')+test.Diabetes.astype('str')+test.Respiratory_illness.astype('str')
train['any_drug'] = train.Consumes_other_tobacco_products + train.Use_of_stimulant_drugs

test['any_drug'] = test.Consumes_other_tobacco_products + test.Use_of_stimulant_drugs
tr = set(train.disease).difference(set(test.disease))

te = set(test.disease).difference(set(train.disease))



train.disease = train.disease.apply(lambda x: 'other' if x in tr else x).astype('category')

test.disease = test.disease.apply(lambda x: 'other' if x in te else x).astype('category')
val = {'no':['2010000','2001100', '2111000', '1100111', '1000101','1111000'],

         'yes':['1101110','0110000']}
d = list(test[test.disease.apply(lambda x: x in val['no'])].index)

test.drop(index = d, inplace = True)

ind_no.extend(d)
ind_yes = []

d = list(test[test.disease.apply(lambda x: x in val['yes'])].index)

test.drop(index = d, inplace = True)

ind_yes.extend(d)
for i,j in enumerate(train.columns):

    for c in train.columns[i+1:]:

        print('Columns Fixed v/s Variable:\t', j,c)

        print(train.groupby([j,c])['UnderRisk'].value_counts(),'\n\n')
train.shape
tr_dummies = pd.DataFrame(pd.get_dummies(train[['Gender','disease','any_drug']].astype('category')))

te_dummies = pd.DataFrame(pd.get_dummies(test[['Gender','disease','any_drug']].astype('category')))



train = pd.concat([train,tr_dummies],axis=1)

test = pd.concat([test,te_dummies],axis=1)



train.drop(columns=['Gender','disease','any_drug'],inplace=True)

test.drop(columns=['Gender','disease','any_drug'],inplace=True)
label = train.UnderRisk

label = label.apply(lambda x : 1 if (x=='yes') else 0)

train.drop(columns=['UnderRisk'],inplace=True)
515+184
label.shape
from keras.layers import *

from keras.utils import to_categorical

from keras.models import Sequential
model = Sequential()

model.add(Dense(32, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

#model.add(Dense(16, activation = 'relu'))

#model.add(BatchNormalization())

#model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))



model.compile(loss = 'binary_crossentropy', metrics=['accuracy'], optimizer = 'adam')
model.fit(train.values, to_categorical(label,2), batch_size=128, epochs = 100, validation_split=0.2, shuffle=True)
def model(m, split ,train, label, test, random, need = False):

    te = []

    strat = StratifiedKFold(n_splits= split, shuffle=True, random_state = random)

    for tr_index,te_index in strat.split(train,label):

        

        xtrain,xval = train.loc[tr_index,:],train.loc[te_index,:]

        ytrain,yval = label.loc[tr_index], label.loc[te_index]

        

        m.fit(xtrain,ytrain)

        tr_pred = m.predict_proba(xtrain)

        te_pred = m.predict_proba(xval)

        

        print('Training Loss :{}                                    Testing Loss : {}'.format(log_loss(ytrain,tr_pred),log_loss(yval,te_pred)))

        if need :

            te.append(pd.DataFrame(m.predict_proba(test), index = test.index))

    return(te)    
train.head()
train.shape
model(RandomForestClassifier(random_state=12312),5,train,label,test,2344)
model(DecisionTreeClassifier(random_state=124),5,train,label,test,1323)
result = model(LogisticRegression(random_state=9994),5,train,label,test,245993, need = True)
model(GradientBoostingClassifier(random_state = 65444),5,train,label,test,331022)
result = model(SVC(probability=True, random_state=5410, kernel = 'rbf', C = 0.1),5,train,label,test,331022, True)
import lightgbm as lgb

import xgboost as xgb

result = model(lgb.LGBMClassifier(random_state=2993),5,train,label,test,34995, need = True)
result = model(xgb.XGBClassifier(random_state=2304),5,train,label,test,34005, True)
from sklearn.model_selection import GridSearchCV, train_test_split



xtr, xval, ytr,yval = train_test_split(train, label, test_size=0.2, random_state = 123)

param = [{'num_leaves':[5,10,15,20,25,30,40,50,65],

         'lambda_l1':[0.1,0.3,0.6,0.8],

         'lambda_l2':[0.1,0.4,1,2,5,6,8,10]}]

gs = GridSearchCV(param_grid=param, cv=3, estimator=lgb.LGBMClassifier(random_state = 31), scoring='neg_log_loss', verbose=1,n_jobs=-1)

gs.fit(xtr, ytr)
gs.best_score_, gs.best_estimator_
est_1 = gs.best_estimator_
xgb.XGBClassifier()
param = [{'max_depth':[2,3,4,5,7,9,12],

         'colsample_bytree':[0.1,0.3,0.6,0.8],

         'subsample':[0.1,0.4,0.6,0.7,0.8,0.9]}]

gs = GridSearchCV(param_grid=param, cv=3, estimator=xgb.XGBClassifier(random_state = 311), scoring='neg_log_loss', verbose=1,n_jobs=-1)

gs.fit(xtr, ytr)
gs.best_score_, gs.best_estimator_
est_2 = gs.best_estimator_
LogisticRegression()
param = [{'C':[0.01,0.1,1,2,3],

         'l1_ratio':[0.1,0.3,0.6,0.8],

         'max_iter':[10,50,100,200,300,500,1000],

          'warm_start':[True, False]}]

gs = GridSearchCV(param_grid=param, cv=3, estimator=LogisticRegression(random_state=990110), scoring='neg_log_loss', verbose=1,n_jobs=-1)

gs.fit(xtr, ytr)
gs.best_score_, gs.best_estimator_
est_3 = gs.best_estimator_
result = model(StackingClassifier(final_estimator = LogisticRegression(random_state=12113),

                        estimators = [('lg', est_1),

                                      ('xg', est_2),

                                      ('lr', est_3)]),5,train,label,test,511200,True)
model(VotingClassifier(estimators = [('lg', lgb.LGBMClassifier(random_state=2)),

                                      ('xg', xgb.XGBClassifier(random_state=2)),

                                      ('lr',LogisticRegression(random_state=2))], voting = 'soft'),5,train,label,test,22)
temp = pd.DataFrame({0: 1.0, 1:0.0}, index = ind_no)

temp3 = pd.DataFrame({0: 0.0, 1:1.0}, index = ind_yes)
result = pd.DataFrame({0:(result[0][0]+result[1][0]+result[2][0]+result[3][0]+result[4][0])/5,

             1:(result[0][1]+result[1][1]+result[2][1]+result[3][1]+result[4][1])/5}, index= result[0].index)
result = pd.concat([result,temp, temp3], axis = 0).sort_index()
result.shape
sample.shape, result.shape
result.columns= sample.columns
result
result.to_excel('stack_4.xlsx',index=False)