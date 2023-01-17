import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.info()
test.info()
mean_survival = train.Survived.mean()

mean_survival
all_data = pd.concat([train,test],sort=False,ignore_index=True)

all_data.info()
train.groupby('Pclass').Survived.agg(['mean','size'])
sns.countplot(train.Pclass,hue=train.Survived)
sns.countplot(train.Sex,hue=train.Survived)
train.groupby('Sex')['Survived'].mean()
#train,test合わせた年齢の分布

plt.hist(all_data.Age,bins=50,range=(0,100))
plt.hist(train.Age[train.Survived==0],alpha=0.5,label=0,bins=50,range=(0,100))

plt.hist(train.Age[train.Survived==1],alpha=0.5,label=1,bins=50,range=(0,100))

plt.legend()

plt.vlines([16,34],0,35,colors='r',linestyles='dashed')
plt.hist([train.SibSp[train.Survived==0],train.SibSp[train.Survived==1]],label=[0,1])

plt.legend()
plt.hist([train.Parch[train.Survived==0],train.Parch[train.Survived==1]],label=[0,1])

plt.legend()
train['family_size'] = train.SibSp+train.Parch+1

all_data['family_size'] = all_data.SibSp+all_data.Parch+1
plt.hist([train.family_size[train.Survived==0],train.family_size[train.Survived==1]],label=[0,1])

plt.legend()
train.groupby('family_size').Survived.mean()
plt.hist(train.Fare[train.Survived==0],alpha=0.5,label=0,bins=20)

plt.hist(train.Fare[train.Survived==1],alpha=0.5,label=1,bins=20)

plt.legend()
#Fareの欠損値をtrainデータの中央値で埋める

all_data.Fare[all_data.Fare.isnull()] = train.Fare.median()
train.Cabin.unique()
cabin_first = train.Cabin.map(lambda x:str(x)[0])
train.Survived.groupby(cabin_first).agg(['mean','size'])
all_data['Cabin_class'] = all_data.Cabin.map(lambda x:str(x)[0])

np.sort(all_data.Cabin_class.unique())
train.groupby('Embarked').Survived.agg(['mean','size'])
#欠損値を最頻値で埋める

all_data.Embarked.fillna(train.Embarked.mode()[0],inplace=True)
train.Ticket.sort_values().unique()
train.groupby(train.Ticket.map(lambda x: str(x)[0])).Survived.agg(['mean','count'])
ticket_num = train.Ticket.str.extract('^(\d)|^\D.*\s(\d)\d*$').fillna(0).astype(int).sum(axis=1)
train.Survived.groupby(ticket_num).agg(['mean','count'])
test.groupby(test.Ticket.map(lambda x: str(x)[0])).size()
all_data['Ticket_first_letter'] = all_data.Ticket.map(lambda x: str(x)[0])

all_data['Ticket_first_num'] = all_data.Ticket.str.extract('^(\d)|^\D.*\s(\d)\d*$').fillna(0).astype(int).sum(axis=1)
plt.scatter(all_data.Pclass,all_data.Ticket_first_num,alpha=0.01)

np.corrcoef(all_data.Pclass,all_data.Ticket_first_num)
train.Name.head(10)
#family_nameとhonorificを抽出

names = all_data.Name.str.extract('^(.*), (\w+)')

all_data['family_name'] = names.loc[:,0]

all_data['honorific'] = names.loc[:,1]
all_data.loc[:,['family_name','honorific']].head()
#trainデータにおいてhonorificごとの生存率

all_data.iloc[:891,:].groupby('honorific').Survived.agg(['mean','size'])
all_data.iloc[891:,:].groupby('honorific').size()
name_map={

    "Capt":        "Officer",

    "Col":         "Officer",

    "Major":       "Officer",

    "Jonkheer":    "Royalty",

    "Don":         "Royalty",

    "Sir" :        "Royalty",

    "Dr":          "Officer",

    "Rev":         "Officer",

    "the":         "Royalty",  #the Coutess

    "Dona":        "Royalty",

    "Mme":         "Mrs",

    "Mlle":        "Miss",

    "Ms":          "Mrs",

    "Mr" :         "Mr",

    "Mrs" :        "Mrs",

    "Miss" :       "Miss",

    "Master" :     "Master",

    "Lady" :       "Royalty"}

all_data.honorific.replace(name_map,inplace=True)
all_data.honorific.unique()
all_data.groupby(['honorific','Sex']).Survived.agg(['mean','size'])
all_data.groupby(['honorific','Sex']).Age.mean()
def newage (cols):

    title=cols[0]

    Sex=cols[1]

    Age=cols[2]

    if pd.isnull(Age):

        if title=='Master' and Sex=="male":

            return 5.48

        elif title=='Miss' and Sex=='female':

            return 21.80

        elif title=='Mr' and Sex=='male': 

            return 32.25

        elif title=='Mrs' and Sex=='female':

            return 36.87

        elif title=='officer' and Sex=='female':

            return 49

        elif title=='Officer' and Sex=='male':

            return 46.14

        elif title=='Royalty' and Sex=='female':

            return 40

        else:

            return 42.33

    else:

        return Age 
all_data.Age = all_data[['honorific','Sex','Age']].apply(newage,axis=1)
all_data.family_name.value_counts().head(10)
all_data.query('family_name=="Andersson"')
def family_mean(series):

    others_survival = pd.Series([series.drop(i).mean() for i in series.index])

    

    return others_survival
all_data.Survived.fillna(mean_survival,inplace=True)



all_data['family_survival'] = all_data.query('family_size>1').groupby('family_name').Survived.transform(family_mean)

all_data['family_survival'].fillna(mean_survival,inplace=True)
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier

import lightgbm as lgb

import optuna
all_data_one_hot = pd.get_dummies(all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','family_size','Cabin_class',

                                            'Ticket_first_letter','family_survival']],

                                  columns=['Ticket_first_letter','Cabin_class','Sex','Pclass','Embarked'])
train = all_data_one_hot.iloc[:891]

test = all_data_one_hot.iloc[891:]



train.info()

test.info()
train.head()
def objective(trial):

    

    params = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'verbosity': -1,

        'boosting_type': 'gbdt',

        'learning_rate':trial.suggest_loguniform('learning_rate',0.01,0.05),

        'num_leaves': trial.suggest_int('num_leaves', 2, 256),

        'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),

        'max_depth':trial.suggest_int('max_depth',3,8)

    }



    kf = KFold(n_splits=5,shuffle=True,random_state=1)



    accuracy_scores = []



    for train_index,valid_index in kf.split(train):

        X_train, X_valid = train.drop(columns='Survived').iloc[train_index],train.drop(columns='Survived').iloc[valid_index]

        y_train, y_valid = train.Survived.iloc[train_index],train.Survived.iloc[valid_index]



        #学習

        lgb_train = lgb.Dataset(X_train,y_train)

        gbm = lgb.train(params,lgb_train)



        #予測

        y_pred = np.round(gbm.predict(X_valid)).astype(int)

        accuracy_scores.append(accuracy_score(y_valid,y_pred))



    return np.mean(accuracy_scores)

study = optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=1))

optuna.logging.disable_default_handler()

study.optimize(objective, n_trials=100)
print('Number of finished trials: {}'.format(len(study.trials)))



print('Best trial:')

trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')

for key, value in trial.params.items():

    print('    {}: {}'.format(key, value))
params = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'verbosity': -1,

        'boosting_type': 'gbdt'

         }



params.update(trial.params)
kf = KFold(n_splits=5,shuffle=True,random_state=6)



accuracy_scores = []



for train_index,valid_index in kf.split(train):

    X_train, X_valid = train.drop(columns='Survived').iloc[train_index],train.drop(columns='Survived').iloc[valid_index]

    y_train, y_valid = train.Survived.iloc[train_index],train.Survived.iloc[valid_index]



    #学習

    lgb_train = lgb.Dataset(X_train,y_train)

    gbm = lgb.train(params,lgb_train,num_boost_round=100)



    #予測

    y_pred = np.round(gbm.predict(X_valid)).astype(int)

    accuracy_scores.append(accuracy_score(y_valid,y_pred))
print(accuracy_scores)

print('CV平均:',np.mean(accuracy_scores))
#重要な特徴量を表示

lgb.plot_importance(gbm,figsize=(12,10))
all_train_data = lgb.Dataset(train.drop(columns=['Survived']),label=train.Survived)
gbm_final = lgb.train(train_set=all_train_data,

                      params = params,

                      num_boost_round=100

                      )
y_pred = np.round(gbm_final.predict(test.drop(columns=['Survived']))).astype(int)
submission_df = pd.read_csv('../input/titanic/gender_submission.csv')
submission_df['Survived'] = y_pred
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=+9), 'JST')

ts = datetime.now(JST).strftime('%y%m%d%H%M')



submission_df.to_csv((ts+'lgb.csv'),index=False)
lgb.plot_importance(gbm_final, figsize=(12, 6))

plt.show()
kf = KFold(n_splits=5,shuffle=True,random_state=6)



mean_accuracy = {}

thresh_holds = np.arange(0.2,0.6,0.01)



for thresh_hold in thresh_holds:

    accuracy_scores = []

    for train_index,valid_index in kf.split(train):

        X_train, X_valid = train.drop(columns='Survived').iloc[train_index],train.drop(columns='Survived').iloc[valid_index]

        y_train, y_valid = train.Survived.iloc[train_index],train.Survived.iloc[valid_index]



    #学習

        lgb_train = lgb.Dataset(X_train,y_train)

        gbm = lgb.train(params,lgb_train,num_boost_round=100)



    #予測

        y_pred = np.where(gbm.predict(X_valid)<thresh_hold,0,1)

        accuracy_scores.append(accuracy_score(y_valid,y_pred))

    

    mean_accuracy[thresh_hold] = np.mean(accuracy_scores)
mean_accuracy
thresh_holds