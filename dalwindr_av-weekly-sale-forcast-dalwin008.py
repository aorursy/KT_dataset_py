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
        print("hi",os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, RepeatedKFold,StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set()
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import statsmodels.api as sm
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import warnings
# import the_module_that_warns

warnings.filterwarnings("ignore")

from fbprophet import Prophet


## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD,Adadelta,Adam,RMSprop 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
def poly(feat,_df,degree=0):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder
    from sklearn.linear_model import SGDRegressor,LinearRegression,Ridge,Lasso,ElasticNet

    y = _df[_df[label_col]>0][label_col]
    X = _df.loc[_df[label_col]>0, feat] 
    #print(X.shape,y.shape)
   
    # Initializatin of regression models
    regr = LinearRegression()
    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X)
    linear_r2 = r2_score(y, regr.predict(X))
    
    # create polynomial features
    quadratic = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    cubic = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
    fourth = PolynomialFeatures(degree=4, interaction_only=False, include_bias=False)
    fifth = PolynomialFeatures(degree=5, interaction_only=False, include_bias=False)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)
    X_fourth = fourth.fit_transform(X)
    X_fifth = fifth.fit_transform(X)

    if degree==3:
        res = X_cubic
        Degreedpoly=cubic
        target_feature_names = [feat.replace(' ','_') for feat in Degreedpoly.get_feature_names(X.columns)]
        output_df = pd.DataFrame(res, columns = target_feature_names,  index=X.index).iloc[:, len(X.columns):]
        return output_df
    # quadratic fit
    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X))
    quadratic_r2 = r2_score(y, y_quad_fit)
    
    # cubic fit
    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X))
    cubic_r2 = r2_score(y, y_cubic_fit)

    # Fourth fit
    regr = regr.fit(X_fourth, y)
    y_fourth_fit = regr.predict(fourth.fit_transform(X))
    four_r2 = r2_score(y, y_fourth_fit)

    # Fifth fit
    regr = regr.fit(X_fifth, y)
    y_fifth_fit = regr.predict(fifth.fit_transform(X))
    five_r2 = r2_score(y, y_fifth_fit)
    

    
    if len(feat)==1:
        fig = plt.figure(figsize=(30,10))
        # Plot lowest Polynomials
        fig1 = fig.add_subplot(121)
        plt.scatter(X[feat], y, label='training points', color='lightgray')
        plt.plot(X[feat], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2, color='blue', lw=0.5, linestyle=':')
        plt.plot(X[feat], y_quad_fit, label='quadratic (d=2), $R^2=%.3f$' % quadratic_r2, color='red', lw=0.5, linestyle='-')
        plt.plot(X[feat], y_cubic_fit, label='cubic (d=3), $R^2=%.3f$' % cubic_r2,  color='green', lw=0.5, linestyle='--')

        plt.xlabel(feat)
        plt.ylabel('Sale Price')
        plt.legend(loc='upper left')

        # Plot higest Polynomials
        fig2 = fig.add_subplot(122)
        plt.scatter(X[feat], y, label='training points', color='lightgray')
        plt.plot(X[feat], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2, color='blue', lw=2, linestyle=':')
        plt.plot(X[feat], y_fifth_fit, label='Fifth (d=5), $R^2=%.3f$' % five_r2, color='yellow', lw=2, linestyle='-')
        plt.plot(X[feat], y_fifth_fit, label='Fourth (d=4), $R^2=%.3f$' % four_r2, color='red', lw=2, linestyle=':')

        plt.xlabel(feat)
        plt.ylabel('Sale Price')
        plt.legend(loc='upper left')
    else:
        # Plot initialisation
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, s=40)
        return
        # make lines of the regressors:
        plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2, 
                 color='blue', lw=2, linestyle=':')
        plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_quad_fit, label='quadratic (d=2), $R^2=%.3f$' % quadratic_r2, 
                 color='red', lw=0.5, linestyle='-')
        plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_cubic_fit, label='cubic (d=3), $R^2=%.3f$' % cubic_r2, 
                 color='green', lw=0.5, linestyle='--')
        # label the axes
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_zlabel('Sales Price')
        ax.set_title("Poly up to 3 degree")
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def evaluateColumn_Polynomial(_feature,_label_col,_df,_degree=0,_numCol=3):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder
    from sklearn.linear_model import SGDRegressor,LinearRegression,Ridge,Lasso,ElasticNet
   
    # Initializatin of regression models
    regr = lgb1#LinearRegression()
    def extractData(feat):
        #print("ddd")
        y = _df[_df[_label_col]>0][_label_col]
        X = _df.loc[_df[_label_col]>0, [feat]]
        return (X,y)
    
    def getFeatureR2(_model,feat):
        X,y=extractData(feat)
        _model = _model.fit(X, y)
        y_lin_fit = _model.predict(X)
        #print(r2_score(y, y_lin_fit))
        return r2_score(y, y_lin_fit)


    def getPolyDF(_model,_degree,_feat):
        X,y=extractData(_feat)
        _polyType = PolynomialFeatures(degree=_degree, interaction_only=False, include_bias=False)
        _fittedPoly = _polyType.fit_transform(_df[[feat]])
        target_feature_names = [feature.replace(' ','_') for feature in _polyType.get_feature_names(feat)]
        output_df = pd.DataFrame(_fittedPoly, columns = target_feature_names,  index=_df.index).iloc[:,0:]
        
        _fitted_model = _model.fit(output_df, y)
        _polyPrediction = _model.predict(output_df)
        poly_r2 = r2_score(y, _polyPrediction)
        linear_r2 =getFeatureR2(regr,feat).round(4)
        return ( linear_r2.round(4),poly_r2.round(4),output_df.shape[1])
    _cols=['colname','col_R2_val1','col_poly_R2_val2','diff(val2,val1)','colCount']
    FeaturePoly_r2DF=pd.DataFrame(columns=_cols)

    for feat in _feature:
            r2VAL1,r2VAL2,cnt= getPolyDF(regr,_degree,feat)
            #print(feat,r2VAL1,r2VAL2,r2VAL2-r2VAL1)
            _ser1={'colname':feat,'col_R2_val1':r2VAL1,'col_poly_R2_val2':r2VAL2,'diff(val2,val1)':r2VAL2-r2VAL1,'colCount':cnt}
            FeaturePoly_r2DF=FeaturePoly_r2DF.append(_ser1,ignore_index=True)

    return FeaturePoly_r2DF.sort_values('diff(val2,val1)',ascending=False)

train = pd.read_csv('/kaggle/input/train_0irEZ2H.csv')
test = pd.read_csv('/kaggle/input/test_nfaJ3J5.csv')
sub = pd.read_csv('/kaggle/input/sample_submission_pzljTaX.csv')
label_col='units_sold'
train.shape,test.shape
train['total_price'].fillna(train['total_price'].median(),inplace=True)
pd.DataFrame({'trainnull': train.isna().sum() ,'testnull':test.isna().sum(),'type':train.dtypes})
# polydf = poly(['store_id', 'sku_id', 'total_price', 'base_price', 'month', 'day',
#        'year', 'quarter', 'week_of_year', 'profit_margin',
#        'sku_id_raking_by_unitsold', 'store_id_ranking_by_unitsold'],train,3)

def split_data(train_data,test_data):
    #week_of_year_formated=pd.to_datetime(train['week'])#.week_of_year
    #train_data = train_data.drop(["sku_id","store_id"],1)
    #test_data = test_data.drop(["sku_id","store_id"],1)
    train_data['week'] = pd.to_datetime(train_data['week'])
    test_data['week'] = pd.to_datetime(test_data['week'])
    
    # https://pandas.pydata.org/pandas-docs/stable/reference/series.html#datetime-properties
    train_data['month'] = train_data['week'].dt.month
    train_data['day'] = train_data['week'].dt.dayofweek
    train_data['year'] = train_data['week'].dt.year
    train_data['quarter'] = train_data['week'].dt.quarter
    train_data['week_of_year'] = train_data['week'].dt.weekofyear
    train_data['profit_margin'] =  train_data['base_price'] - train_data['total_price']
    train_data['relative_diff_base'] = train_data['profit_margin']/train_data['base_price']
    train_data['relative_diff_total'] = train_data['profit_margin']/train_data['total_price']
    sku_id_ranking=train.groupby('sku_id')[label_col].mean().sort_values()#.reset_index().reset_index()[['sku_id','index']]
    train_data['sku_id_raking_by_unitsold'] = train_data['sku_id'].apply(lambda x:sku_id_ranking[x])
    store_id_ranking=train.groupby('store_id')[label_col].mean().sort_values()
    train_data['store_id_ranking_by_unitsold'] = train_data['store_id'].apply(lambda x:store_id_ranking[x])
    
    sku_id_ranking_base=train.groupby('sku_id')['base_price'].mean().sort_values()#.reset_index().reset_index()[['sku_id','index']]
    train_data['sku_id_raking_by_base_price'] = train_data['sku_id'].apply(lambda x:sku_id_ranking_base[x])
    store_id_ranking_base=train.groupby('store_id')['base_price'].mean().sort_values()
    train_data['store_id_ranking_by_base_price'] = train_data['store_id'].apply(lambda x:store_id_ranking_base[x])
    
    
    #ss=train_data.groupby(['store_id','sku_id'])['base_price'].mean().sort_values()
    #ss=ss.to_dict()
    #train_data['store_id_sku_id_ranking_by_base_price']= train_data[['store_id','sku_id']].apply(lambda y: ss[(y.store_id,y.sku_id)],1)
    
    #tot=train_data.groupby(['store_id','sku_id'])['total_price'].mean().sort_values()
    #tot=tot.to_dict()
    #train_data['store_id_sku_id_ranking_by_total_price']= train_data[['store_id','sku_id']].apply(lambda y: tot[(y.store_id,y.sku_id)],1)
    
    st= train_data[['store_id','sku_id']].drop_duplicates().groupby(['store_id']).count()
    train_data['#sku_id_perStore_id']  =   train_data['store_id'].apply(lambda y: st.loc[y].values[0] )
    sk= train[['store_id','sku_id']].drop_duplicates().groupby(['sku_id']).count()
    train_data['#store_id_perSku_id']  =  train_data['sku_id'].apply(lambda y: sk.loc[y].values[0] )

    test_data['month'] = test_data['week'].dt.month
    test_data['day'] = test_data['week'].dt.dayofweek
    test_data['year'] = test_data['week'].dt.year
    test_data['quarter'] = test_data['week'].dt.quarter
    test_data['week_of_year'] = test_data['week'].dt.weekofyear
    test_data['profit_margin'] =  test_data['base_price'] - test_data['total_price']
    test_data['relative_diff_base'] = test_data['profit_margin']/test_data['base_price']
    test_data['relative_diff_total'] = test_data['profit_margin']/test_data['total_price']
    test_data['sku_id_raking_by_unitsold'] = test_data['sku_id'].apply(lambda x:sku_id_ranking[x])
    test_data['store_id_ranking_by_unitsold'] = test_data['store_id'].apply(lambda x:store_id_ranking[x])
    #test_data['store_id_sku_id_ranking_by_base_price']= test_data[['store_id','sku_id']].apply(lambda y: ss[(y.store_id,y.sku_id)],1)
    #test_data['store_id_sku_id_ranking_by_total_price']= test_data[['store_id','sku_id']].apply(lambda y: tot[(y.store_id,y.sku_id)],1)
    
    test_data['sku_id_raking_by_base_price'] = test_data['sku_id'].apply(lambda x:sku_id_ranking_base[x])
    test_data['store_id_ranking_by_base_price'] = test_data['store_id'].apply(lambda x:store_id_ranking_base[x])
    
    
    test_data['#sku_id_perStore_id']  =   test_data['store_id'].apply(lambda y: st.loc[y].values[0] )
    test_data['#store_id_perSku_id']  =  test_data['sku_id'].apply(lambda y: sk.loc[y].values[0] )

    col = [i for i in test_data.columns if i not in ['week','record_ID']]
    y = train_data[label_col]
    #train_x, test_x, train_y, test_y = train_test_split(train_data[col],train_data[y], test_size=0.2, random_state=2018)
    return (train_data[col], y,test_data[col])

train_x,  train_y,test_data = split_data(train,test)
train_x.shape, train_y.shape, test_data.shape
#ss=train.groupby(['store_id','sku_id'])['base_price'].mean()
#sku_id_ranking=
#train.groupby(['week'])['week'].max()
#.sort_values()#.reset_index().reset_index()[['sku_id','index']]
#test['sku_id'].apply(lambda x:sku_id_ranking[x])

# Best param Evaulated
lgb1= lgb.LGBMRegressor(**{'bagging_fraction': 0.73,'n_estimators':3500,
 'boosting': 'gbdt',
 'feature_fraction': 0.67,
 'lambda_l1': 0.01416357346505337,
 'lambda_l2': 2.5960957064519636,
 'learning_rate': 0.180623531498291,
 'max_bin': 241,
 'max_depth': 20,
 'metric': 'MAE',
 'min_data_in_bin': 131,
 'min_data_in_leaf': 108,
 'min_gain_to_split': 0.13,
 'num_leaves': 1512,
 'objective': 'gamma',
 'subsample': 0.580232207898679})
regressionCrossVal(lgb1,train_x,train_y)

# pred=lgb1.predict(test_data)
# pred.shape,sub.shape
# sub['units_sold']=pred
# sub.to_csv("AV_WEEKLY_SALES.csv",index=False)
# Prediction make 
pred=lgb1.predict(test_data)
pred.shape,sub.shape
sub['units_sold']=pred
sub.to_csv("AV_WEEKLY_SALES.csv",index=False)
#     params = {
#         'nthread': 10,
#          'max_depth': 5,
#         'task': 'train',
#         'boosting_type': 'gbdt',
#         'objective': 'regression_l1',
#         'metric': 'mape', # this is abs(a-e)/max(1,a)

#         'num_leaves': 64,
#         'learning_rate': 0.2,
#        'feature_fraction': 0.9,
#        'bagging_fraction': 0.8,
#         'bagging_freq': 5,
#         'lambda_l1': 3.097758978478437,
#         'lambda_l2': 2.9482537987198496,
#         'verbose': 1,
#         'min_child_weight': 6.996211413900573,
#         'min_split_gain': 0.037310344962162616,
#         }
    



def regressionCrossVal(_model,_X,_Y,stop=5):
    fold = KFold(n_splits=5,  random_state=2020) # for Regression problem
    # fold = StratifiedKFold(n_splits=5,  random_state=2020) # for Classification

    i = 1
    test_res=pd.DataFrame()
    cv_score_train=[]
    cv_score_val=[]
    for train_index, test_index in fold.split(_X, _Y):
        x_train, x_val = _X.iloc[train_index], _X.iloc[test_index]
        y_train, y_val = _Y.iloc[train_index], _Y.iloc[test_index]
        # x_train.shape,x_val.shape,y_train.shape,y_val.shape
        
        _model.fit(x_train,y_train)
        y_pred = _model.predict(x_train)
        train_rmse= np.sqrt(mean_squared_log_error(y_train,y_pred))
        
        y_pred_val = _model.predict(x_val)
        val_rmse = np.sqrt(mean_squared_log_error(y_val,y_pred_val))
        print("iter", i ,"train_MSE = ", train_rmse*100,"validation_MSE = ", val_rmse*100)
        cv_score_train.append(train_rmse)
        cv_score_val.append(val_rmse)
        if i == stop:
            break
        i = i + 1


param_grid = {
    'num_leaves': list(range(8, 92, 4)),
    'min_data_in_leaf': [10, 20, 40, 60, 100],
    'max_depth': [3, 4, 5, 6, 8, 12, 16, -1],
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'bagging_freq': [3, 4, 5, 6, 7],
    'bagging_fraction': np.linspace(0.6, 0.95, 10),
    'reg_alpha': np.linspace(0.1, 0.95, 10),
    'reg_lambda': np.linspace(0.1, 0.95, 10)
}
#regressionCrossVal(lgb1,train_x,train_y)
fixed_params = {
    'objective': 'huber',
    'boosting': 'gbdt',
    'verbosity': -1,
    'random_seed': 19,
    'n_estimators': 50000,
    'metric': 'mae',
    'bagging_seed': 11
}
#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 1000 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM
EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 
EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric

#XGBOOST PARAMETERS
XGB_MAX_LEAVES = 2**12 #maximum number of leaves when using histogram splitting
XGB_MAX_DEPTH = 25 #maximum tree depth for XGBoost
EVAL_METRIC_XGB_REG = 'mae' #XGBoost regression metric
EVAL_METRIC_XGB_CLASS = 'auc' #XGBoost classification metric

#CATBOOST PARAMETERS
CB_MAX_DEPTH = 8 #maximum tree depth in CatBoost
OBJECTIVE_CB_REG = 'MAE' #CatBoost regression metric
OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric

#OPTIONAL OUTPUT
BEST_SCORE = 0

def quick_hyperopt(data, labels, package='lgbm', num_evals=NUM_EVALS, diagnostic=False):
    
    #==========
    #LightGBM
    #==========
    
    if package=='lgbm':
        
        print('Running {} rounds of LightGBM parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth',
                         'num_leaves',
                          'max_bin',
                         'min_data_in_leaf',
                         'min_data_in_bin']
        
        def objective(space_params):
            
            #cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
            
            #extract nested conditional parameters
            if space_params['boosting']['boosting'] == 'goss':
                top_rate = space_params['boosting'].get('top_rate')
                other_rate = space_params['boosting'].get('other_rate')
                #0 <= top_rate + other_rate <= 1
                top_rate = max(top_rate, 0)
                top_rate = min(top_rate, 0.5)
                other_rate = max(other_rate, 0)
                other_rate = min(other_rate, 0.5)
                space_params['top_rate'] = top_rate
                space_params['other_rate'] = other_rate
            
            subsample = space_params['boosting'].get('subsample', 1.0)
            space_params['boosting'] = space_params['boosting']['boosting']
            space_params['subsample'] = subsample
            
            #for classification, set stratified=True and metrics=EVAL_METRIC_LGBM_CLASS
            cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,
                                early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)
            
            best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = 1 - cv_results['auc-mean'][-1]
            #if necessary, replace 'auc-mean' with '[your-preferred-metric]-mean'
            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = lgb.Dataset(data, labels)
                
        #integer and string parameters, used with hp.choice()
        boosting_list = [{'boosting': 'gbdt',
                          'subsample': hp.uniform('subsample', 0.5, 1)},
                         
                         {'boosting': 'goss',
                          'subsample': 1.0,
                         'top_rate': hp.uniform('top_rate', 0, 0.5),
                         'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'
        
        metric_list = ['MAE', 'RMSE'] 
        
        #for classification comment out the line above and uncomment the line below
        #metric_list = ['auc'] #modify as required for other classification metrics
        objective_list_reg = ['huber', 'gamma', 'fair', 'tweedie']
        objective_list_class = ['binary', 'cross_entropy']
        #for classification set objective_list = objective_list_class
        objective_list = objective_list_reg

        space ={'boosting' : hp.choice('boosting', boosting_list),
                'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),
                'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),
                'max_bin': hp.quniform('max_bin', 32, 255, 1),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),
                'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),
                'min_gain_to_split' : hp.quniform('min_gain_to_split', 0.1, 5, 0.01),
                'lambda_l1' : hp.uniform('lambda_l1', 0, 5),
                'lambda_l2' : hp.uniform('lambda_l2', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'metric' : hp.choice('metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),
                'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01)
            }
        
        #optional: activate GPU for LightGBM
        #follow compilation steps here:
        #https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/
        #then uncomment lines below:
        #space['device'] = 'gpu'
        #space['gpu_platform_id'] = 0,
        #space['gpu_device_id'] =  0

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
                
        #fmin() will return the index of values chosen from the lists/arrays in 'space'
        #to obtain actual values, index values are used to subset the original lists/arrays
        best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice
        best['metric'] = metric_list[best['metric']]
        best['objective'] = objective_list[best['objective']]
                
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    #==========
    #XGBoost
    #==========
    
    if package=='xgb':
        
        print('Running {} rounds of XGBoost parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth']
        
        def objective(space_params):
            
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            #extract multiple nested tree_method conditional parameters
            #libera te tutemet ex inferis
            if space_params['tree_method']['tree_method'] == 'hist':
                max_bin = space_params['tree_method'].get('max_bin')
                space_params['max_bin'] = int(max_bin)
                if space_params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'depthwise':
                    grow_policy = space_params['tree_method'].get('grow_policy').get('grow_policy').get('grow_policy')
                    space_params['grow_policy'] = grow_policy
                    space_params['tree_method'] = 'hist'
                else:
                    max_leaves = space_params['tree_method']['grow_policy']['grow_policy'].get('max_leaves')
                    space_params['grow_policy'] = 'lossguide'
                    space_params['max_leaves'] = int(max_leaves)
                    space_params['tree_method'] = 'hist'
            else:
                space_params['tree_method'] = space_params['tree_method'].get('tree_method')
                
            #for classification replace EVAL_METRIC_XGB_REG with EVAL_METRIC_XGB_CLASS
            cv_results = xgb.cv(space_params, train, nfold=N_FOLDS, metrics=[EVAL_METRIC_XGB_REG],
                             early_stopping_rounds=100, stratified=False, seed=42)
            
            best_loss = cv_results['test-mae-mean'].iloc[-1] #or 'test-rmse-mean' if using RMSE
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = 1 - cv_results['test-auc-mean'].iloc[-1]
            #if necessary, replace 'test-auc-mean' with 'test-[your-preferred-metric]-mean'
            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = xgb.DMatrix(data, labels)
        
        #integer and string parameters, used with hp.choice()
        boosting_list = ['gbtree', 'gblinear'] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['MAE', 'RMSE'] 
        #for classification comment out the line above and uncomment the line below
        #metric_list = ['auc']
        #modify as required for other classification metrics classification
        
        tree_method = [{'tree_method' : 'exact'},
               {'tree_method' : 'approx'},
               {'tree_method' : 'hist',
                'max_bin': hp.quniform('max_bin', 2**3, 2**7, 1),
                'grow_policy' : {'grow_policy': {'grow_policy':'depthwise'},
                                'grow_policy' : {'grow_policy':'lossguide',
                                                  'max_leaves': hp.quniform('max_leaves', 32, XGB_MAX_LEAVES, 1)}}}]
        
        #if using GPU, replace 'exact' with 'gpu_exact' and 'hist' with
        #'gpu_hist' in the nested dictionary above
        
        objective_list_reg = ['reg:linear', 'reg:gamma', 'reg:tweedie']
        objective_list_class = ['reg:logistic', 'binary:logistic']
        #for classification change line below to 'objective_list = objective_list_class'
        objective_list = objective_list_reg
        
        space ={'boosting' : hp.choice('boosting', boosting_list),
                'tree_method' : hp.choice('tree_method', tree_method),
                'max_depth': hp.quniform('max_depth', 2, XGB_MAX_DEPTH, 1),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 5),
                'reg_lambda' : hp.uniform('reg_lambda', 0, 5),
                'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
                'gamma' : hp.uniform('gamma', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'eval_metric' : hp.choice('eval_metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'colsample_bynode' : hp.quniform('colsample_bynode', 0.1, 1, 0.01),
                'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
                'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
                'nthread' : -1
            }
        
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
        
        best['tree_method'] = tree_method[best['tree_method']]['tree_method']
        best['boosting'] = boosting_list[best['boosting']]
        best['eval_metric'] = metric_list[best['eval_metric']]
        best['objective'] = objective_list[best['objective']]
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        if 'max_bin' in best:
            best['max_bin'] = int(best['max_bin'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    #==========
    #CatBoost
    #==========
    
    if package=='cb':
        
        print('Running {} rounds of CatBoost parameter optimisation:'.format(num_evals))
        
        #clear memory 
        gc.collect()
            
        integer_params = ['depth',
                          #'one_hot_max_size', #for categorical data
                          'min_data_in_leaf',
                          'max_bin']
        
        def objective(space_params):
                        
            #cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            #extract nested conditional parameters
            if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
                bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')
                space_params['bagging_temperature'] = bagging_temp
                
            if space_params['grow_policy']['grow_policy'] == 'LossGuide':
                max_leaves = space_params['grow_policy'].get('max_leaves')
                space_params['max_leaves'] = int(max_leaves)
                
            space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']
            space_params['grow_policy'] = space_params['grow_policy']['grow_policy']
                           
            #random_strength cannot be < 0
            space_params['random_strength'] = max(space_params['random_strength'], 0)
            #fold_len_multiplier cannot be < 1
            space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)
                       
            #for classification set stratified=True
            cv_results = cb.cv(train, space_params, fold_count=N_FOLDS, 
                             early_stopping_rounds=25, stratified=False, partition_random_seed=42)
           
            best_loss = cv_results['test-MAE-mean'].iloc[-1] #'test-RMSE-mean' for RMSE
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = cv_results['test-Logloss-mean'].iloc[-1]
            #if necessary, replace 'test-Logloss-mean' with 'test-[your-preferred-metric]-mean'
            
            return{'loss':best_loss, 'status': STATUS_OK}
        
        train = cb.Pool(data, labels.astype('float32'))
        
        #integer and string parameters, used with hp.choice()
        bootstrap_type = [{'bootstrap_type':'Poisson'}, 
                           {'bootstrap_type':'Bayesian',
                            'bagging_temperature' : hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
                          {'bootstrap_type':'Bernoulli'}] 
        LEB = ['No', 'AnyImprovement', 'Armijo'] #remove 'Armijo' if not using GPU
        #score_function = ['Correlation', 'L2', 'NewtonCorrelation', 'NewtonL2']
        grow_policy = [{'grow_policy':'SymmetricTree'},
                       {'grow_policy':'Depthwise'},
                       {'grow_policy':'Lossguide',
                        'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]
        eval_metric_list_reg = ['MAE', 'RMSE', 'Poisson']
        eval_metric_list_class = ['Logloss', 'AUC', 'F1']
        #for classification change line below to 'eval_metric_list = eval_metric_list_class'
        eval_metric_list = eval_metric_list_reg
                
        space ={'depth': hp.quniform('depth', 2, CB_MAX_DEPTH, 1),
                'max_bin' : hp.quniform('max_bin', 1, 32, 1), #if using CPU just set this to 254
                'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0, 5),
                'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),
                'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),
                #'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1), #uncomment if using categorical features
                'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),
                'learning_rate' : hp.uniform('learning_rate', 0.05, 0.25),
                'eval_metric' : hp.choice('eval_metric', eval_metric_list),
                'objective' : OBJECTIVE_CB_REG,
                #'score_function' : hp.choice('score_function', score_function), #crashes kernel - reason unknown
                'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),
                'grow_policy': hp.choice('grow_policy', grow_policy),
                #'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),# CPU only
                'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
                'od_type' : 'Iter',
                'od_wait' : 25,
                'task_type' : 'GPU',
                'verbose' : 0
            }
        
        #optional: run CatBoost without GPU
        #uncomment line below
        #space['task_type'] = 'CPU'
            
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
        
        #unpack nested dicts first
        best['bootstrap_type'] = bootstrap_type[best['bootstrap_type']]['bootstrap_type']
        best['grow_policy'] = grow_policy[best['grow_policy']]['grow_policy']
        best['eval_metric'] = eval_metric_list[best['eval_metric']]
        
        #best['score_function'] = score_function[best['score_function']] 
        #best['leaf_estimation_method'] = LEM[best['leaf_estimation_method']] #CPU only
        best['leaf_estimation_backtracking'] = LEB[best['leaf_estimation_backtracking']]        
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    else:
        print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "cb" for CatBoost.')  
import gc
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
lgbm_params = quick_hyperopt(train_x, train_y, 'lgbm', 50)
poly_col=['profit_margin^2']#,'year_quarter_profit_margin']#,'year_week_of_year_profit_margin']#,'month_year_profit_margin']#]
polydf_train = poly(['store_id', 'sku_id', 'total_price', 'base_price', 'month', 'day',
       'year', 'quarter', 'week_of_year', 'profit_margin',
       'sku_id_raking_by_unitsold', 'store_id_ranking_by_unitsold'],train,3)
polydf_train[poly_col]
train_x_final= pd.concat([train_x,polydf_train[poly_col]],1)
test_temp = test_data.copy()
test_temp[label_col]=pred
polydf_test = poly(['store_id', 'sku_id', 'total_price', 'base_price', 'month', 'day',
       'year', 'quarter', 'week_of_year', 'profit_margin',
       'sku_id_raking_by_unitsold', 'store_id_ranking_by_unitsold'],test_temp,3)
polydf_test[poly_col]
test_data_final= pd.concat([test_data,polydf_test[poly_col]],1)
lgb1= lgb.LGBMRegressor(**{'bagging_fraction': 0.73,
 'boosting': 'gbdt',
 'feature_fraction': 0.67,
 'lambda_l1': 0.01416357346505337,
 'lambda_l2': 2.5960957064519636,
 'learning_rate': 0.180623531498291,
 'max_bin': 241,
 'max_depth': 20,
 'metric': 'MAE',
 'min_data_in_bin': 131,
 'min_data_in_leaf': 108,
 'min_gain_to_split': 0.13,
 'num_leaves': 1512,
 'objective': 'gamma',
 'subsample': 0.580232207898679})
regressionCrossVal(lgb1,train_x_final,train_y)