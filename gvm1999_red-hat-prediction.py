# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



def act_data_treatment(dsname):

    dataset = dsname

    

    for col in list(dataset.columns):

        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:

            if dataset[col].dtype == 'object':

                dataset[col].fillna('type 0', inplace=True)

                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)

            elif dataset[col].dtype == 'bool':

                dataset[col] = dataset[col].astype(np.int8)

    

    return dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_act = pd.read_csv('/kaggle/input/predicting-red-hat-business-value/act_train.csv')

df_people = pd.read_csv('/kaggle/input/predicting-red-hat-business-value/people.csv')



df_people['date'] = pd.to_datetime(df_people['date'], format="%Y-%m-%d")

df_people['month_people'] = df_people.date.dt.month

df_people['year_people'] = df_people.date.dt.year

df_people = df_people.drop('date',axis=1)



df_act.drop(['activity_id'],axis=1)

df_act['date'] = pd.to_datetime(df_act['date'], format="%Y-%m-%d")

df_act['month_act'] = df_act.date.dt.month

df_act['year_act'] = df_act.date.dt.year

df_act = df_act.drop('date',axis=1)



df_act = act_data_treatment(df_act)

df_people = act_data_treatment(df_people)
df_train = pd.merge(df_act,df_people,how='inner',on='people_id')

df_train = df_train.drop(['activity_id','people_id'], axis=1)
outcome1= df_act[df_act['outcome']==1].shape[0]

outcome0= df_act[df_act['outcome']==0].shape[0]



f,ax = plt.subplots(figsize =(5,5))

ax.tick_params(axis="x", labelsize=15)

ax.tick_params(axis="y", labelsize=30)



ax.pie([outcome1,outcome0], labels=['1','0'],explode=(0, 0.1),shadow=True,colors=['r','black'],autopct='%1.1f%%')

plt.title("Total of Outcomes",fontsize=20)

plt.show()
gb = df_act.loc[df_act['outcome']==1].groupby('people_id')['activity_id'].count().sort_values(ascending=False)[:20]

gb = gb.sort_values()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=20)

ax.tick_params(axis="y", labelsize=30)



ax.barh(list(gb.index), list(gb.values),height = 0.9, color = ['black','r'])

ax.set_facecolor('whitesmoke')

ax.patch.set_alpha(0.9)

plt.title("People Count with Outcome 1",fontsize=30)

plt.ylabel('Gran Prix',fontsize = 20)

plt.xlabel('Total',fontsize = 20,)

plt.show()



gb = df_act.loc[df_act['outcome']==0].groupby('people_id')['activity_id'].count().sort_values(ascending=False)[:20]

gb = gb.sort_values()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=20)

ax.tick_params(axis="y", labelsize=30)



ax.barh(list(gb.index), list(gb.values),height = 0.9, color = ['black','r'])

ax.set_facecolor('whitesmoke')

ax.patch.set_alpha(0.9)

plt.title("People Count with Outcome 0",fontsize=30)

plt.ylabel('Gran Prix',fontsize = 20)

plt.xlabel('Total',fontsize = 20,)

plt.show()



gb = df_act.loc[df_act['outcome']==0].groupby('activity_category')['activity_id'].count().sort_values(ascending=False)[:20]

gb = gb.sort_values()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=20)

ax.tick_params(axis="y", labelsize=30)



ax.barh(list(gb.index), list(gb.values),height = 0.9, color = ['black','r'])

ax.set_facecolor('whitesmoke')

ax.patch.set_alpha(0.9)

plt.title("Activity Count with Outcome 0",fontsize=30)

plt.ylabel('Gran Prix',fontsize = 20)

plt.xlabel('Total',fontsize = 20,)

plt.show()



gb = df_act.loc[df_act['outcome']==1].groupby('activity_category')['activity_id'].count().sort_values(ascending=False)[:20]

gb = gb.sort_values()



f,ax = plt.subplots(figsize =(25,15))

ax.tick_params(axis="x", labelsize=20)

ax.tick_params(axis="y", labelsize=30)



ax.barh(list(gb.index), list(gb.values),height = 0.9, color = ['black','r'])

ax.set_facecolor('whitesmoke')

ax.patch.set_alpha(0.9)

plt.title("Activity Count with Outcome 1",fontsize=30)

plt.ylabel('Gran Prix',fontsize = 20)

plt.xlabel('Total',fontsize = 20,)

plt.show()
y = df_act['outcome']

df_train = df_train.drop(['outcome'],axis=1)



columns = [column for column in df_train.columns if column != 'char_38']
import lightgbm as lgbm

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size = 0.25, random_state = 0)



train_set = lgbm.Dataset(X_train, y_train, silent=False,categorical_feature=list(columns))

test_set = lgbm.Dataset(X_test, y_test, silent=False,categorical_feature=list(columns))



params = {}



params['learning_rate'] = 0.03

params['boosting_type'] = 'gbdt'

params['task'] = 'train'

params['verbose']= 10

params['is_training_metric']= True

params['num_iterations'] = 10100

params['max_bin'] = 30000

params['objective'] = 'binary'

params['metric'] = ['binary_logloss','auc','roc']

params['min_gain_to_split']= 0.25

#params['early_stopping_round']= 10



clf = lgbm.train(params, train_set = train_set, num_boost_round=15000,verbose_eval=10, valid_sets=test_set)

df_teste = pd.read_csv('/kaggle/input/predicting-red-hat-business-value/act_test.csv')

keys = df_teste['activity_id']



df_teste = df_teste.drop(['activity_id'],axis=1)



df_teste['date'] = pd.to_datetime(df_teste['date'], format="%Y-%m-%d")

df_teste['month_act'] = df_teste.date.dt.month

df_teste['year_act'] = df_teste.date.dt.year

df_teste = df_teste.drop('date',axis=1)



df_teste = act_data_treatment(df_teste)



df_teste = pd.merge(df_teste,df_people,how='inner',on='people_id')

df_teste = df_teste.drop(['people_id'], axis=1)



values = clf.predict(df_teste, iteration=clf.best_iteration)



submission = pd.DataFrame({

        "activity_id": keys,

        "outcome": values

})
submission.to_csv('submission.csv',index=False)