import time

import math

import seaborn as sns

import pandas as pd

import numpy as np

import scipy as sci

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from hyperopt import hp, tpe, Trials, STATUS_OK

from hyperopt import fmin



from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.model_selection import train_test_split



from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight') 

%matplotlib inline
ibm_df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

description = pd.DataFrame(index=['observations(rows)', 'percent missing', 'dtype', 'range'])

numerical = []

categorical = []

for col in ibm_df.columns:

    obs = ibm_df[col].size

    p_nan = round(ibm_df[col].isna().sum()/obs, 2)

    num_nan = f'{p_nan}% ({ibm_df[col].isna().sum()}/{obs})'

    dtype = 'categorical' if ibm_df[col].dtype == object else 'numerical'

    numerical.append(col) if dtype == 'numerical' else categorical.append(col)

    rng = f'{len(ibm_df[col].unique())} labels' if dtype == 'categorical' else f'{ibm_df[col].min()}-{ibm_df[col].max()}'

    description[col] = [obs, num_nan, dtype, rng]



numerical.remove('EmployeeCount')

numerical.remove('StandardHours')

pd.set_option('display.max_columns', 100)

display(description)

display(ibm_df.head())
p_col = 2

fig, ax = plt.subplots(12, p_col, figsize=(10, 30))



for idx, feature in enumerate(numerical): 

    col = idx%p_col

    row = math.floor(idx/p_col)

    ibm_df.boxplot(column=feature, by='Attrition', ax = ax[row][col])

    

plt.tight_layout()
count_yes = ibm_df.Attrition[ibm_df.Attrition == 'Yes'].size

count_no = ibm_df.Attrition[ibm_df.Attrition == 'No'].size



plt.bar(['Leave', 'Stay'], [count_yes, count_no])

plt.title('IBM Attrition Label Imbalance')

plt.xlabel('Whether IBM Employee Left')

plt.ylabel('Count of Employees')

plt.show()
features = ['MonthlyIncome', 'Attrition', 'JobLevel', 'TotalWorkingYears', 'YearsAtCompany', 'YearsWithCurrManager']

pairplot = sns.pairplot(ibm_df[features], diag_kind='kde', hue='Attrition')

plt.show()
trace1 = go.Heatmap(

    z = ibm_df[numerical].astype(float).corr().values,

    x = ibm_df[numerical].columns.values,

    y = ibm_df[numerical].columns.values,

    colorscale = 'Portland', 

    reversescale = False, 

    opacity = 1.0)

        

data = [trace1]

layout = go.Layout(

    title = 'Correlation Among IBM Employee Attrition Numerical Features',

    xaxis = dict(ticks = '', nticks = 36),

    yaxis = dict(ticks = ''),

    width = 700, height = 700

)



fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
def org_results(trials, hyperparams, model_name):

    fit_idx = -1

    for idx, fit  in enumerate(trials):

        hyp = fit['misc']['vals']

        xgb_hyp = {key:[val] for key, val in hyperparams.items()}

        if hyp == xgb_hyp:

            fit_idx = idx

            break

            

    train_time = str(trials[-1]['refresh_time'] - trials[0]['book_time'])

    acc = round(trials[fit_idx]['result']['accuracy'], 3)

    train_auc = round(trials[fit_idx]['result']['train auc'], 3)

    test_auc = round(trials[fit_idx]['result']['test auc'], 3)



    results = {

        'model': model_name,

        'parameter search time': train_time,

        'accuracy': acc,

        'test auc score': test_auc,

        'training auc score': train_auc,

        'parameters': hyperparams

    }

    return results
xgb_data = ibm_df.copy()

xgb_dummy = pd.get_dummies(xgb_data[categorical], drop_first=True)

xgb_data = pd.concat([xgb_dummy, xgb_data], axis=1)

xgb_data.drop(columns = categorical, inplace=True)

xgb_data.rename(columns={'Attrition_Yes': 'Attrition'}, inplace=True)



y_df = xgb_data['Attrition'].reset_index(drop=True)

x_df = xgb_data.drop(columns='Attrition')

train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20)



def xgb_objective(space, early_stopping_rounds=50):

    

    model = XGBClassifier(

        learning_rate = space['learning_rate'], 

        n_estimators = int(space['n_estimators']), 

        max_depth = int(space['max_depth']), 

        min_child_weight = space['m_child_weight'], 

        gamma = space['gamma'], 

        subsample = space['subsample'], 

        colsample_bytree = space['colsample_bytree'],

        objective = 'binary:logistic'

    )



    model.fit(train_x, train_y, 

              eval_set = [(train_x, train_y), (test_x, test_y)],

              eval_metric = 'auc',

              early_stopping_rounds = early_stopping_rounds,

              verbose = False)

     

    predictions = model.predict(test_x)

    test_preds = model.predict_proba(test_x)[:,1]

    train_preds = model.predict_proba(train_x)[:,1]

    

    xgb_booster = model.get_booster()

    train_auc = roc_auc_score(train_y, train_preds)

    test_auc = roc_auc_score(test_y, test_preds)

    accuracy = accuracy_score(test_y, predictions) 



    return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,

            'test auc': test_auc, 'train auc': train_auc

           }



space = {

    'n_estimators': hp.quniform('n_estimators', 50, 1000, 25),

    'max_depth': hp.quniform('max_depth', 1, 12, 1),

    'm_child_weight': hp.quniform('m_child_weight', 1, 6, 1),

    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),

    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),

    'learning_rate': hp.loguniform('learning_rate', np.log(.001), np.log(.3)),

    'colsample_bytree': hp.quniform('colsample_bytree', .5, 1, .1)

}



trials = Trials()

xgb_hyperparams = fmin(fn = xgb_objective, 

                 max_evals = 150, 

                 trials = trials,

                 algo = tpe.suggest,

                 space = space

                 )



xgb_results = org_results(trials.trials, xgb_hyperparams, 'XGBoost')

display(xgb_results)
lgb_data = ibm_df.copy()

lgb_dummy = pd.get_dummies(lgb_data[categorical], drop_first=True)

lgb_data = pd.concat([lgb_dummy, lgb_data], axis=1)

lgb_data.drop(columns = categorical, inplace=True)

lgb_data.rename(columns={'Attrition_Yes': 'Attrition'}, inplace=True)



y_df = lgb_data['Attrition'].reset_index(drop=True)

x_df = lgb_data.drop(columns='Attrition')

train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20)



def lgb_objective(space, early_stopping_rounds=50):

    

    lgbm = LGBMClassifier(

        learning_rate = space['learning_rate'],

        n_estimators= int(space['n_estimators']), 

        max_depth = int(space['max_depth']),

        num_leaves = int(space['num_leaves']),

        colsample_bytree = space['colsample_bytree'],

        feature_fraction = space['feature_fraction'],

        reg_lambda = space['reg_lambda'],

        reg_alpha = space['reg_alpha'],

        min_split_gain = space['min_split_gain']

    )

    

    lgbm.fit(train_x, train_y, 

            eval_set = [(train_x, train_y), (test_x, test_y)],

            early_stopping_rounds = early_stopping_rounds,

            eval_metric = 'auc',

            verbose = False)

    

    predictions = lgbm.predict(test_x)

    test_preds = lgbm.predict_proba(test_x)[:,1]

    train_preds = lgbm.predict_proba(train_x)[:,1]

    

    train_auc = roc_auc_score(train_y, train_preds)

    test_auc = roc_auc_score(test_y, test_preds)

    accuracy = accuracy_score(test_y, predictions)  



    return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,

            'test auc': test_auc, 'train auc': train_auc

           }



trials = Trials()

space = {

    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),

    'n_estimators': hp.quniform('n_estimators', 50, 1200, 25),

    'max_depth': hp.quniform('max_depth', 1, 15, 1),

    'num_leaves': hp.quniform('num_leaves', 10, 150, 1),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0), 

    'feature_fraction': hp.uniform('feature_fraction', .3, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    'min_split_gain': hp.uniform('min_split_gain', 0.0001, 0.1)

}



lgb_hyperparams = fmin(fn = lgb_objective, 

                 max_evals = 150, 

                 trials = trials,

                 algo = tpe.suggest,

                 space = space

                 )



lgb_results = org_results(trials.trials, lgb_hyperparams, 'LightGBM')

display(lgb_results)
cbo_data = ibm_df.copy()



for cat in categorical:

    cbo_data[cat] = cbo_data[cat].astype('category').cat.codes



y_df = cbo_data['Attrition'].reset_index(drop=True)

x_df = cbo_data.drop(columns='Attrition')



cboost_cat = categorical[1:]

train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20)

cat_dims = [train_x.columns.get_loc(name) for name in cboost_cat]     

    

def cat_objective(space, early_stopping_rounds=30):

    

    cboost = CatBoostClassifier(

    eval_metric  = 'AUC', 

    learning_rate = space['learning_rate'],

    iterations = space['iterations'],

    depth = space['depth'],

    l2_leaf_reg = space['l2_leaf_reg'],

    border_count = space['border_count']

    )

    

    cboost.fit(train_x, train_y, 

              eval_set = [(train_x, train_y), (test_x, test_y)],

              early_stopping_rounds = early_stopping_rounds,

              cat_features = cat_dims, 

              verbose = False)

    

    predictions = cboost.predict(test_x)

    test_preds = cboost.predict_proba(test_x)[:,1]

    train_preds = cboost.predict_proba(train_x)[:,1]    



    train_auc = roc_auc_score(train_y, train_preds)

    test_auc = roc_auc_score(test_y, test_preds)

    accuracy = accuracy_score(test_y, predictions)

    

    return {'status': STATUS_OK, 'loss': 1-test_auc, 'accuracy': accuracy,

            'test auc': test_auc, 'train auc': train_auc}

    

trials = Trials()

space = {

    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),

    'iterations': hp.quniform('iterations', 25, 1000, 25),

    'depth': hp.quniform('depth', 1, 16, 1),

    'border_count': hp.quniform('border_count', 30, 220, 5), 

    'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 10, 1)

}



cboost_hyperparams = fmin(fn = cat_objective, 

                 max_evals = 150, 

                 trials = trials,

                 algo = tpe.suggest,

                 space = space

                 )



cbo_results = org_results(trials.trials, cboost_hyperparams, 'CatBoost')

display(cbo_results)
final_results = pd.DataFrame([xgb_results, lgb_results, cbo_results])

display(final_results)