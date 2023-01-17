# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

full=pd.concat([data,test])

print(full.shape)

full.head()
test.head()
data.isnull().sum()

test.isnull().sum()
# A string coorelation can be observed 

fig = plt.figure(figsize=(11,6))

ax = sns.countplot(data['Sex'],hue=data['Survived'])

ax.set_title('By Sex Survived',fontsize = 20)

fig = plt.figure(figsize=(11,6))

ax = sns.countplot(data['Fare'],hue=data['Survived'])

ax.set_title('By Fare Survived',fontsize = 20)
fare_ranges = np.array([x*20 for x in range(30)])

data['Fare'] =fare_ranges.searchsorted(data.Fare)

data.head()
fig = plt.figure(figsize=(11,6))

ax = sns.countplot(data['Fare'],hue=data['Survived'])

ax.set_title('By Fare Survived',fontsize = 20)
full['Has_Age'] = full['Age'].isnull().map(lambda x : 0 if x == True else 1)

full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

full.head()

full['Title'] = full['Title'].replace(['Capt', 'Col', 'Countess', 'Don',

                                               'Dr', 'Dona', 'Jonkheer', 

                                                'Major','Rev','Sir'],'Unknown') 

full['Title'] = full['Title'].replace(['Mlle', 'Ms','Mme'],'Miss')

full['Title'] = full['Title'].replace(['Lady'],'Mrs')

g = sns.factorplot(y='Age',x='Title',kind='box',hue='Pclass', data=full, 

               size=6,aspect=1.5)

plt.subplots_adjust(top=0.93)

g.fig.suptitle('Age vs Pclass cross Title', fontsize = 20)

full.drop(columns=['Cabin','PassengerId','Ticket'],inplace=True)

full['FamilySize']=full['Parch']+full['SibSp']+1

full.head()

data=full[full['Survived'].isnull()==False]

print(data[['FamilySize', 'Survived']].groupby(data['FamilySize'], as_index=False).mean())

fx, axes = plt.subplots(1, 2, figsize=(15,5))

axes[0].set_title('Family Size counts')

axes[1].set_title('Survival Rate vs Family Size')

fig1_family = sns.countplot(x=data.FamilySize, ax=axes[0], palette='cool')

fig2_family = sns.barplot(x=data.FamilySize, y=data.Survived, ax=axes[1], palette='cool')

full['isAlone'] = full['FamilySize'].map(lambda x: 1 if x<2 or x>7   else 0)

data=full[full['Survived'].isnull()==False]

fx, axes = plt.subplots(1, 2, figsize=(15, 6))

fig1_alone = sns.countplot(data=data, x='isAlone', ax=axes[0])

fig2_alone = sns.barplot(data=data, x='isAlone', y='Survived', ax=axes[1])

## no need for sibsp and parch now

full.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)



missing_mask = (full['Has_Age'] == 0)

pd.crosstab(full[missing_mask]['Pclass'],full[missing_mask]['Title'])
#lets see the median of each combo (pclass/title)

# we use median to escape the outliers influence

full.pivot_table(values='Age', index=['Pclass'], columns=['Title'],aggfunc=np.median)

full['Title'] = full['Title'].map({"Mr":0, "Unknown" : 1, "Master" : 2, "Miss" : 3, "Mrs" : 4 })

Pclass_title_pred = full.pivot_table(values='Age', index=['Pclass'], columns=['Title'],aggfunc=np.median).values

# by default lets copy the old age col in the pcall_title col

full['P_Ti_Age'] = full['Age']

# filling Missing age with Pclass & Title

for i in range(0,5):

    # 0,1,2,3,4

    for j in range(1,4):

        # 1,2,3

            full.loc[(full.Age.isnull()) & (full.Pclass == j) & (full.Title == i),'P_Ti_Age'] = Pclass_title_pred[j-1, i]

full['P_Ti_Age'] = full['P_Ti_Age'].astype('int')
full.head()

age_ranges = np.array([16,100])

full['Age'] =age_ranges.searchsorted(full.Age)



full['P_Ti_Age'] = age_ranges.searchsorted(full.P_Ti_Age)

full.head()

## transforming the embarked and sex feature to numeric 

full=full[False==full['Embarked'].isnull()]

full['Embarked']=full['Embarked'].apply(lambda x : 0 if x=='S' else 1 if x=='C' else 2)

full['Sex']=full['Sex'].apply(lambda x : 0 if x=='male' else 1 )

data_without_old_age=full.drop(columns=['Name','Age'])

data_without_old_age.head()
# split the data again to train and test 

to_predict=data_without_old_age[data_without_old_age['Survived'].isnull()]

data=data_without_old_age[False==data_without_old_age['Survived'].isnull()]

## At This stage, I have finished the feature engineering part, Shoosed xgboost to go with and I have only to define my validation 

##strategy to go with 
import warnings

from sklearn.model_selection import StratifiedKFold ## slightly better than normal cross val

warnings.filterwarnings('ignore')



estim=np.array([x for x in (5,25,50,100)])

scores=[]

for n in range(4):

    xg_gbm = xgb.XGBClassifier(max_depth=7, n_estimators=estim[n], learning_rate=0.5,colsample_bytree=0.6)

    (x_tr,y_tr)=(data.drop('Survived',axis=1).values, data['Survived'].values)

    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=123)

    cv_scores=[]

    for tr_ind,val_ind in skf.split(x_tr, y_tr):

        X_tr, X_val=data.iloc[tr_ind],data.iloc[val_ind]

        xg_gbm.fit(data.iloc[tr_ind].drop('Survived',axis=1).values,data.iloc[tr_ind].Survived.values)

        cv_scores.append(accuracy_score(xg_gbm.predict(data.iloc[val_ind].drop('Survived',axis=1).values),data.iloc[val_ind].Survived.values))



    scores.append((np.asarray(cv_scores)).mean())

    #print((np.asarray(cv_scores)).mean())

plt.plot(estim,scores)

## after arround 40 estimator, the model started to overfit,

## also keep in mind that there is a relation between nb_estimator and learning rate
test['Survived']=xg_gbm.predict(to_predict.drop('Survived',axis=1).values)

result_df=test[['PassengerId','Survived']]

result_df['Survived']=result_df['Survived'].astype('int')

result_df.to_csv('Random_f_output_family_sz_.csv',header=True,index=False)

result_df.tail()
