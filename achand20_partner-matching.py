# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split
train = pd.read_csv("../input/speed-dating-experiment/Speed Dating Data.csv", encoding = 'ISO-8859-1')
train.head()
#Finding Fields with NULL values

train.isnull().sum()
date = pd.concat([train.iloc[:, 0],train.iloc[:, 2],train.iloc[:,11:28],train.iloc[:,30:36],train.iloc[:,39:43],train.iloc[:,45:68],train.iloc[:,69:75],

                  train.iloc[:,81:92],train.iloc[:,97:102],train.iloc[:,104:108]], axis=1)
#Removing null rows now that the nulls are in the hundreds and not the thousands.

date2 = date.dropna()
f, axes = plt.subplots(1,6,figsize=(15,6))

f.subplots_adjust(hspace=0.4,wspace=0.7)

attributes = ['attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1','shar1_1']

for i in range(6):

    sns.barplot(y=attributes[i], x= "gender", data=date2 , ax=axes[i])
f, axes = plt.subplots(1,2,figsize=(10,5))

sns.barplot(y="like_o", x= "gender",data=date2,ax=axes[0])

sns.barplot(y="dec_o", x= "gender",data=date2,ax=axes[1])

f, axes = plt.subplots(figsize=(15,5))

sns.barplot(y="dec_o", x= "age",data=date2,ax=axes)
fig, axes = plt.subplots(1,5,figsize=(20,5))

x_attributes = ['attr3_1','sinc3_1','intel3_1','fun3_1','amb3_1']

y_attributes = ['attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1']

for i in range(5):

    sns.regplot(x=x_attributes[i], y=y_attributes[i],data=date2, ax=axes[i])

    
corrmat = date2.corr()

plt.subplots(figsize=(20,40))

sns.heatmap(corrmat, vmax=0.9, square=True)

#print(corrmat)
fig, axes = plt.subplots(3,3,figsize=(20,10))

attributes = ['pf_o_att','pf_o_sin','dec','pf_o_fun','like_o','dec','met','dec_o','met_o']



for i in range(3):

    sns.regplot(y="match", x=attributes[i],data=date2, ax=axes[0][i])

for i in range(3):

    sns.regplot(y="match", x=attributes[i+3],data=date2, ax=axes[1][i])

for i in range(3):

    sns.regplot(y="match", x=attributes[i+6],data=date2, ax=axes[2][i])

    
#We drop the fields that have no correlation with the 'MATCH' field.

date3 = date2.drop(['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 

                    'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 

                   'shopping', 'yoga'], axis=1)
#To find percentage of people who got matched.

pd.crosstab(index=date3['match'],columns="count")
no_love_count = len(date3[(date3['dec_o']==0) & (date3['dec']==1)]) 

+ len(date3[(date3['dec_o']==1) & (date3['dec']==0)])

perc_broken_heart = no_love_count / len(date3.index)

perc_broken_heart*100
#Taking only the important fields into consideration.

Y=date3['match']

X=date3[['like','dec','met','met_o']]

#X.drop(['match'],axis=1,inplace=True)

from sklearn.model_selection import GridSearchCV

def hyper_parameter_tuning(parameters,model,c_v):

    grid_search = GridSearchCV(model,

                               parameters,

                               cv = c_v,

                               n_jobs = 10,

                               verbose = True)

    grid_search.fit(X,Y)

    #print("All Scores =",grid_search.cv_results_)

    print("Best Score =",grid_search.best_score_)

    print("Best Params =",grid_search.best_params_)

    return(grid_search.best_score_,grid_search.best_params_)
#Hyper-Parameter-Tuning for Logistic Regression.

hyper_parameter_tuning({'C':[0.001,0.01,1,10],

                        'max_iter':[100,200,5000],

                        'random_state':[0,1,2,3]},LogisticRegression(),5)
#Hyper-Parameter-Tuning for XGBClassifier.

hyper_parameter_tuning({'learning_rate':[0.01,0.1], 

                        'n_estimators':[140,200], 

                        'max_depth':[4,5,7],

                        'min_child_weight':[2,3,4], 

                        'gamma':[0.2], 

                        'subsample':[0.6,0.8], 

                        'colsample_bytree':[0.7,1.0],

                        'objective':['binary:logistic'], 

                        'seed':[27]},XGBClassifier(),5)
#Hyper-Parameter-Tuning for RandomForestClassifier.

hyper_parameter_tuning({'bootstrap': [True, False],

                        'max_depth': [40, 60, None],

                        'max_features': ['auto', 'sqrt'],

                        'min_samples_leaf': [1, 2],

                        'min_samples_split': [2, 5],

                        'n_estimators': [200, 600]},RandomForestClassifier(),5)
Xgboost = XGBClassifier(learning_rate =0.01, 

                        colsample_bytree=1, 

                        gamma=0.2, 

                        max_depth=4, 

                        min_child_weight=2, 

                        n_estimators=140, 

                        objective='binary:logistic',

                        seed=27,

                        subsample=0.8 )