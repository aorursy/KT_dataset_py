import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sn
import matplotlib.pyplot as plt
import lightgbm as lgbm
import gc
import shap
from hyperopt import hp, tpe, Trials, fmin
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
def mae_score(truth, predictions):
    return mean_absolute_error(truth, predictions)

def score(y_test, y_pred):
    weights = X_test.isholiday.apply(lambda x: 5 if 1 else 1)
    return np.round(np.sum(weights*abs(y_test-y_pred))/(np.sum(weights)), 4)

def wmae_score(truth, predictions):
    weights = X_train.isholiday.apply(lambda x: 5 if 1 else 1)
    return np.round(np.sum(weights*abs(truth-predictions))/(np.sum(weights)), 4)
stores = pd.read_csv('../input/wallmart/stores.csv', sep=',')
stores.T.head()
base_treino = pd.read_csv('../input/wallmart/train.csv', sep=',')
base_treino.shape
base_treino.head()
base_treino[base_treino['IsHoliday'] == True]
base_feats = pd.read_csv('../input/wallmart/features.csv', sep=',')
base_feats.head()
base_feats.shape
base_feats.Store.value_counts()
# stores = pd.read_csv('stores.csv', sep=',')
# base_treino = pd.read_csv('train.csv', sep=',')
# base_feats = pd.read_csv('features.csv', sep=',')
base_treino = pd.merge(base_treino, stores, how='left', on = ['Store'])
base_treino.head()
base_treino = pd.merge(base_treino, base_feats.drop(['IsHoliday'], axis=1), how='left', on = ['Store','Date'])
base_treino.head()
# base_treino['IsHoliday_x'].equals(base_treino['IsHoliday_y'])
base_treino.dtypes
base_treino['Date'] = pd.to_datetime(base_treino['Date'])
base_treino.dtypes
base_treino['day'] = base_treino['Date'].dt.day
base_treino['month'] = base_treino['Date'].dt.month
base_treino['year'] = base_treino['Date'].dt.year
base_treino.head()
base_treino["Store_char"] = base_treino["Store"].astype(str) 
base_treino["Dept_char"] = base_treino["Dept"].astype(str) 
base_treino["Date"] = base_treino["Date"].astype(str)
base_treino['key'] = base_treino[['Store_char', 'Dept_char', 'Date']].agg('_'.join, axis=1)
base_treino.drop(['Store_char','Dept_char','Date'], axis=1, inplace=True)
holiday = {False: 0, True: 1}
base_treino['isholiday'] = base_treino['IsHoliday'].map(holiday)
base_treino.drop(['IsHoliday'], axis=1, inplace=True)
tipo = {'A':1, 'B':2, 'C':3}
base_treino['type_num'] = base_treino['Type'].map(tipo)

# le = preprocessing.LabelEncoder()
# base_treino['type_num'] = le.fit_transform(base_treino.Type)
base_treino.Type.value_counts()
base_treino.type_num.value_counts()
base_treino.drop(['Type'], axis=1, inplace=True)
list(base_treino.columns)
base_treino['rate_cpi_size'] = base_treino['Size'] / base_treino['CPI']
base_treino.head()
base_treino['holiday_type'] = np.nan
# Super Bowl 2-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
base_treino.loc[(base_treino['day'] == 2) & (base_treino['month'] == 2) & (base_treino['year'] == 2010), 'holiday_type'] = 1
base_treino.loc[(base_treino['day'] == 11) & (base_treino['month'] == 2) & (base_treino['year'] == 2011), 'holiday_type'] = 1
base_treino.loc[(base_treino['day'] == 10) & (base_treino['month'] == 2) & (base_treino['year'] == 2012), 'holiday_type'] = 1
# Labor Day 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
base_treino.loc[(base_treino['day'] == 10) & (base_treino['month'] == 9) & (base_treino['year'] == 2010), 'holiday_type'] = 2
base_treino.loc[(base_treino['day'] == 9) & (base_treino['month'] == 9) & (base_treino['year'] == 2011), 'holiday_type'] = 2
base_treino.loc[(base_treino['day'] == 7) & (base_treino['month'] == 9) & (base_treino['year'] == 2012), 'holiday_type'] = 2
# Thanksgiving 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
base_treino.loc[(base_treino['day'] == 26) & (base_treino['month'] == 11) & (base_treino['year'] == 2010), 'holiday_type'] = 3
base_treino.loc[(base_treino['day'] == 25) & (base_treino['month'] == 11) & (base_treino['year'] == 2011), 'holiday_type'] = 3
base_treino.loc[(base_treino['day'] == 23) & (base_treino['month'] == 11) & (base_treino['year'] == 2012), 'holiday_type'] = 3
# Christmas 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13
base_treino.loc[(base_treino['day'] == 31) & (base_treino['month'] == 12) & (base_treino['year'] == 2010), 'holiday_type'] = 4
base_treino.loc[(base_treino['day'] == 30) & (base_treino['month'] == 12) & (base_treino['year'] == 2011), 'holiday_type'] = 4
base_treino.loc[(base_treino['day'] == 28) & (base_treino['month'] == 12) & (base_treino['year'] == 2012), 'holiday_type'] = 4
base_treino.month.value_counts()
base_treino['holiday_type'].value_counts()
base_treino.shape
base_treino.head().T
base_treino.describe()
percent_missing = base_treino.isnull().sum() * 100 / len(base_treino)
missing_value_df = pd.DataFrame({'column_name': base_treino.columns,
                                 'percent_missing': percent_missing})
missing_value_df
corr_matrix = base_treino.drop('key', axis=1).corr()
fig, ax = plt.subplots(figsize=(14,8)) 
sn.heatmap(corr_matrix, annot=True, ax=ax)
plt.show()
cols_to_drop = [
    'holiday_type',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

base_treino.drop(cols_to_drop, axis=1, inplace=True)
base_treino.head()
X_train, X_test, y_train, y_test = train_test_split(base_treino.fillna(0).drop(['key','Weekly_Sales'], axis=1),
                                                    base_treino.Weekly_Sales,
                                                    test_size = 0.3,
                                                    random_state = 42)
best_mae = 0
wmae_scorer = make_scorer(wmae_score, greater_is_better=True)
def objective(params):
#     params['n_jobs'] = 10
    params['boosting_type'] = 'gbdt'
    params['num_leaves'] = int(params['num_leaves'])
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = 8
    params['subsample'] = 0.8
    params['feature_fraction'] = 0.8
    params['random_state'] = 42
    
    best_params['lambda_l1'] = 0.1
    best_params['lambda_l2'] = 0.1
    
    reg = lgbm.LGBMRegressor(**params)
    
    global best_mae
    
    scores = cross_val_score(reg, X_train, y_train, scoring=wmae_scorer, cv=KFold(5))
    score_avg = scores.mean()
    score_std = scores.std()
    
    print('WMAE {:.5f} std{:.5f} params {}'.format(score_avg, score_std, params))
    
    best_err = score_avg
    gc.collect()
    return (score_avg)
space = {
    'num_leaves': hp.quniform('num_leaves', 2, 20, 2),
    'n_estimators': hp.quniform('n_estimators', 20, 200, 20),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
}

print('Hyperopt')
best_params = {}
best_params = fmin(fn = objective,
                   space = space,
                   algo = tpe.suggest,
                   max_evals = 12)

best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['n_estimators'] = int(best_params['n_estimators'])

best_params['boosting_type'] = 'gbdt'
best_params['max_depth'] = 8
best_params['subsample'] = 0.8
best_params['feature_fraction'] = 0.8
best_params['random_state'] = 42
best_params['lambda_l1'] = 0.1
best_params['lambda_l2'] = 0.1

reg = lgbm.LGBMRegressor(**best_params)
print('Fim de treinamento')
reg.fit(X_train, y_train)
score(y_test, reg.predict(X_test.fillna(0)))
explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
base_test = pd.read_csv('../input/wallmart/test.csv', sep=',')
base_test.shape
(115064/421570)*100
base_test.head()
base_test[base_test['IsHoliday'] == True]
8928/29661
def dataprep(base):
    stores = pd.read_csv('../input/wallmart/stores.csv', sep=',')
    base_feats = pd.read_csv('../input/wallmart/features.csv', sep=',')

    base_teste = pd.merge(base, stores, how='left', on = ['Store'])

    base_teste = pd.merge(base_teste, base_feats.drop(['IsHoliday'], axis=1), how='left', on = ['Store','Date'])

    base_teste['Date'] = pd.to_datetime(base_teste['Date'])
   
    base_teste['day'] = base_teste['Date'].dt.day
    base_teste['month'] = base_teste['Date'].dt.month
    base_teste['year'] = base_teste['Date'].dt.year

    base_teste["Store_char"] = base_teste["Store"].astype(str) 
    base_teste["Dept_char"] = base_teste["Dept"].astype(str) 
    base_teste["Date"] = base_teste["Date"].astype(str)

    base_teste['key'] = base_teste[['Store_char', 'Dept_char', 'Date']].agg('_'.join, axis=1)

    base_teste.drop(['Store_char','Dept_char','Date'], axis=1, inplace=True)

    holiday = {False: 0, True: 1}
    base_teste['isholiday'] = base_teste['IsHoliday'].map(holiday)
    base_teste.drop(['IsHoliday'], axis=1, inplace=True)

    tipo = {'A':1, 'B':2, 'C':3}
    base_teste['type_num'] = base_teste['Type'].map(tipo)
    base_teste.drop(['Type'], axis=1, inplace=True)

    base_teste['rate_cpi_size'] = base_teste['Size'] / base_teste['CPI']

    base_teste['holiday_type'] = np.nan

    # Super Bowl 2-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
    base_teste.loc[(base_teste['day'] == 10) & (base_teste['month'] == 2) & (base_teste['year'] == 2012), 'holiday_type'] = 1
    base_teste.loc[(base_teste['day'] == 8) & (base_teste['month'] == 2) & (base_teste['year'] == 2013), 'holiday_type'] = 1

    # Labor Day 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
    base_teste.loc[(base_teste['day'] == 7) & (base_teste['month'] == 9) & (base_teste['year'] == 2012), 'holiday_type'] = 2
    base_teste.loc[(base_teste['day'] == 6) & (base_teste['month'] == 9) & (base_teste['year'] == 2013), 'holiday_type'] = 2

    # Thanksgiving 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
    base_teste.loc[(base_teste['day'] == 23) & (base_teste['month'] == 11) & (base_teste['year'] == 2012), 'holiday_type'] = 3
    base_teste.loc[(base_teste['day'] == 29) & (base_teste['month'] == 11) & (base_teste['year'] == 2013), 'holiday_type'] = 3
    
    # Christmas 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13
    base_teste.loc[(base_teste['day'] == 28) & (base_teste['month'] == 12) & (base_teste['year'] == 2012), 'holiday_type'] = 4
    base_teste.loc[(base_teste['day'] == 27) & (base_teste['month'] == 12) & (base_teste['year'] == 2013), 'holiday_type'] = 4 
 
    cols_to_drop = [
        'holiday_type',
        'MarkDown1',
        'MarkDown2',
        'MarkDown3',
        'MarkDown4',
        'MarkDown5'
    ]
    base_teste.drop(cols_to_drop, axis=1, inplace=True)

    return base_teste
backtest = dataprep(base_test)
backtest.shape
backtest.head()
ID = []
ID = backtest.key

ws = []
ws = reg.predict(backtest.fillna(0).drop('key', axis=1))

df_test = pd.DataFrame({'Id': ID, 'Weekly_Sales': ws})
# df_test.to_csv('submission_7.csv', sep=',', index=False)
# df_test.shape
