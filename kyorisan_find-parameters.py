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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import optuna



%matplotlib inline

plt.style.use('ggplot')
df = pd.read_csv('/kaggle/input/input/crowdfunding.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df[['category','main_category']].nunique()
df['main_category'].value_counts()
succes_rate = round(df['state'].value_counts() / len(df['state']) * 100,2)

succes_rate
print('before:',df.shape)

df = df[(df['state'] =='failed') | (df['state'] == 'successful')]

df['state'] = df['state'].map({ 'failed' : 0,'successful' : 1})

print('after:',df.shape)
plt.figure(figsize=(12,8))

plt.hist(df['usd_goal_real'],bins=100)

plt.show()
df['usd_goal_real'] = np.log1p(df['usd_goal_real'])

plt.figure(figsize=(12,8))

plt.hist(df['usd_goal_real'],bins=100)

plt.show()
df['deadline'] = pd.to_datetime(df['deadline'],format = '%Y-%m-%d %H:%M:%S')

df['launched'] = pd.to_datetime(df['launched'],format = '%Y-%m-%d %H:%M:%S')



df['dulation'] = (df['deadline'] - df['launched']).dt.days



df['quarter'] = df['launched'].dt.quarter

df['month'] = df['launched'].dt.month

df['year'] = df['launched'].dt.year

df['dayoweek'] = df['launched'].dt.dayofweek



df.head()
df = df.drop(columns=['ID','deadline','goal','launched','pledged','backers','usd pledged','usd_pledged_real'])

df.head()
df['name_len'] = df['name'].str.len()

df['num_word'] = df['name'].apply(lambda x: len(str(x).split(' ')))

df.drop(columns=['name'],inplace =True)

df.head()
df.isnull().sum()
df['name_len'] = df['name_len'].fillna(0)
df.isnull().sum()
df = pd.get_dummies(df,['category','main_category','currency','country'])

df.head()
X =df.drop(columns='state')

y = df['state']



print(X.shape)

print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
def EvaluateFunc(X_train, X_test, y_train, y_test, score_metric):

    def _evaluate_func(model_obj):

        """

        Crossvalidation_add

        """

        model_obj.fit(X_train, y_train)

        y_pred = model_obj.predict(X_test)

        error = score_metric(y_test, y_pred)

        return error

    return _evaluate_func





def Objective(evaluate_func, trial_models, trial_condition):

    """

    - int: integer

    - uni: a uniform float sampling

    - log: a uniform float sampling on log scale

    - dis: a discretized uniform float sampling

    - cat: category; ('auto', 'mode1', 'mode2', )

    

    """

    model_names = list(trial_models)

    method_names = {

        'int': 'suggest_int',

        'uni': 'suggest_uniform',

        'log': 'suggest_loguniform',

        'dis': 'suggest_discrete_uniform',

        'cat': 'suggest_categorical',

    }

    model_params = {

        model_name: {key: (method_names.get(test[0]), ('{}_{}'.format(model_name, key), *test[1:])) if type(test) is tuple else test

            for key, test in trial_condition.get(model_name).items()}

                for model_name in model_names

    }

    

    def _objective(trial):



        # ハイパーパラメータ生成

        model_name = trial.suggest_categorical('classifier', model_names)

        params = {

            key: getattr(trial, test[0])(*test[1]) if type(test) is tuple else test

                for key, test in model_params.get(model_name).items()

        }

        model_obj = trial_models.get(model_name)(**params)



        #評価

        error = evaluate_func(model_obj)



        return error

    return _objective
trial_models = {

    'rfc': RandomForestClassifier,                            

    'xgbc': xgb.XGBClassifier,

}



# 比較対象の各モデルに設定するハイパーパラメータ名と値の型と範囲を指定

#    - int: integer. ex: ('int', 最小値, 最大値)

#    - uni: a uniform float sampling. ex: ('uni', 最小値, 最大値)

#    - log: a uniform float sampling on log scale. ex: ('log', 最小値, 最大値)

#    - dis: a discretized uniform float sampling. ex: ('dis', 最小値, 最大値, 間隔)

#    - cat: category. ex: ('cat', (文字列A, 文字列B, 文字列C, ))



trial_condition = {

    'rfc': {

        'n_jobs':-1,

        'n_estimators': ('int', 1, 1000),

        'warm_start':True,

        'max_depth': ('int', 1, 20),

        'min_samples_leaf':('int', 1,10),

        'max_features':'sqrt',

        'verbose':1

    },

    'xgbc': {

        'n_jobs':-1,

        'n_estimators': ('int', 1, 1000),

        'max_depth': ('int',3,10),

        'tree_method':('cat', ('auto','exact','hist')),

        'reg_lambda': ('log', 1e-3, 1e3),

        'reg_alpha': ('log', 1e-3, 1e3),

        'min_child_weight': ('int',1,20),

        'gamma' : ('dis',0.1,1,0.1),

        'eta' : ('log', 0.001,0.3),

        'subsample': ('dis', 0.1,0.5,0.05),

        'objective':'binary:logistic',

        'eval_metric':'logloss',

        'predictor':'cpu_predictor',

        'verbose':1

    },

    

}



# 最適化する指標の指定

score_metric = accuracy_score

direction = 'maximize'
evaluate = EvaluateFunc(X_train,X_test,y_train,y_test,score_metric)

objective = Objective(evaluate,trial_models,trial_condition)





# 最適化

study = optuna.create_study(direction=direction)

study.optimize(objective, n_trials=10)

print(study.best_value)

print(study.best_params)
