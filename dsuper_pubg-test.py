import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random 
random.seed(42)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def reload():
    gc.collect()
    df = pd.read_csv('../input/train_V2.csv')
    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    #df=pd.concat([df, pd.get_dummies(df['matchType'])],axis=1)
    return df

def reload_test():
    gc.collect()
    df_test=pd.read_csv('../input/test_V2.csv')
    return df_test
df=reload()
def train_test_split(df, test_size=0.1):
    match_ids = df['matchId'].unique().tolist()
    train_size = int(len(match_ids) * (1 - test_size))
    train_match_ids = random.sample(match_ids, train_size)

    train = df[df['matchId'].isin(train_match_ids)]
    test = df[-df['matchId'].isin(train_match_ids)]
    
    return train, test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#using simple model to test the feature selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

def run_experiment(preprocess):
    df = reload()
    df.drop(columns=['matchType'], inplace=True)
    
    df = preprocess(df)

    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    train, val = train_test_split(df, 0.1)
    '''
    #standarlize
    scaler = preprocessing.StandardScaler().fit(train[cols_to_fit])
    train[cols_to_fit]=scaler.transform(train[cols_to_fit])
    val[cols_to_fit]=scaler.transform(val[cols_to_fit])
    ''' 
    
    model = LinearRegression()
    model.fit(train[cols_to_fit], train[target])
    
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)


def run_experiments(preprocesses):
    results = []
    for preprocess in preprocesses:
        start = time.time()
        score = run_experiment(preprocess)
        execution_time = time.time() - start
        results.append({
            'name': preprocess.__name__,
            'error': score,
            'execution time': f'{round(execution_time, 2)}s'
        })
        gc.collect()
        
    return pd.DataFrame(results, columns=['name', 'error', 'execution time']).sort_values(by='error')
def original(df):
    return df

def best6(df):
    target = 'winPlacePerc'
    save_col='matchId'
    cols_to_choose = ['killPlace', 'walkDistance', 'numGroups', 'maxPlace','kills','matchDuration',save_col,target]
    return df[cols_to_choose]

def best7(df):
    target = 'winPlacePerc'
    save_col='matchId'
    cols_to_choose = ['killPlace', 'walkDistance', 'numGroups', 'maxPlace','kills','matchDuration','rideDistance',save_col,target]
    return df[cols_to_choose]

def best8(df):
    target = 'winPlacePerc'
    save_col='matchId'
    cols_to_choose = ['killPlace', 'walkDistance', 'numGroups', 'maxPlace','kills','matchDuration','rideDistance','DBNOs',save_col,target]
    return df[cols_to_choose]

def best9(df):
    target = 'winPlacePerc'
    save_col='matchId'
    cols_to_choose = ['killPlace', 'walkDistance', 'numGroups', 'maxPlace','kills','matchDuration','rideDistance','DBNOs','boosts',save_col,target]
    return df[cols_to_choose]

def best10(df):
    target = 'winPlacePerc'
    save_col='matchId'
    cols_to_choose = ['killPlace', 'walkDistance', 'numGroups', 'maxPlace','kills','matchDuration','rideDistance','DBNOs','boosts','weaponsAcquired',save_col,target]
    return df[cols_to_choose]

def best11(df):
    target = 'winPlacePerc'
    save_col='matchId'
    cols_to_choose = ['killPlace', 'walkDistance', 'numGroups', 'maxPlace','kills','matchDuration','rideDistance','DBNOs','boosts','weaponsAcquired','winPoints',save_col,target]
    return df[cols_to_choose]

def drop1(df):
    df.drop(columns=['vehicleDestroys'], inplace=True)
    return df

def drop2(df):
    df.drop(columns=['vehicleDestroys'], inplace=True)
    df.drop(columns=['roadKills'], inplace=True)
    return df

def drop3(df):
    df.drop(columns=['vehicleDestroys'], inplace=True)
    df.drop(columns=['roadKills'], inplace=True)
    df.drop(columns=['headshotKills'], inplace=True)
    return df

def drop4(df):
    df.drop(columns=['vehicleDestroys'], inplace=True)
    df.drop(columns=['roadKills'], inplace=True)
    df.drop(columns=['headshotKills'], inplace=True)
    df.drop(columns=['teamKills'], inplace=True)
    return df
#complex model
#1.random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
def run_Random_Forest(preprocess,df_pre):
    #get dummy but should drop the 'matchType' still
    df=df_pre[:]
    df.drop(columns=['matchType'], inplace=True)
    
    df =preprocess(df)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    #downsample
    nouse, use = train_test_split(df, 0.1)
    
    train, val = train_test_split(use, 0.1)
    
    model=RandomForestRegressor(n_estimators=200,max_depth=10)
    model.fit(train[cols_to_fit], train[target]) 
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)

def run_Random_Forests(preprocesses,df_pre):
    results = []
    for preprocess in preprocesses:
        start = time.time()
        score = run_Random_Forest(preprocess,df_pre)
        execution_time = time.time() - start
        results.append({
            'name': preprocess.__name__,
            'score': score,
            'execution time': f'{round(execution_time, 2)}s'
        })
        gc.collect()
        
    return pd.DataFrame(results, columns=['name', 'score', 'execution time']).sort_values(by='score')

def run_Random_Forest_CV(preprocess,n_est,m_depth,df_pre):
    #get dummy but should drop the 'matchType' still
    df=df_pre[:]
    df.drop(columns=['matchType'], inplace=True)
    
    df =preprocess(df)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    #downsample
    nouse, use = train_test_split(df, 0.1)
    
    train, val = train_test_split(use, 0.1)
    
    model=RandomForestRegressor(n_estimators=n_est,max_depth=m_depth)
    model.fit(train[cols_to_fit], train[target]) 
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)

def run_Random_Forests_CV(n_est,m_depth,df_pre):
    results = []
    for n in n_est:
        for m in m_depth:
            start = time.time()
            score = run_Random_Forest_CV(drop2,n,m,df_pre) #decide to use drop2
            execution_time = time.time() - start
            results.append({
                'n_estimators': n,
                'max_depth': m,
                'error': score,
                'execution time': f'{round(execution_time, 2)}s'
            })
            gc.collect()
        
    return pd.DataFrame(results, columns=['n_estimators', 'max_depth', 'error', 'execution time']).sort_values(by='error')

def run_Linear_lasso_CV(preprocess,alpha_use,df_pre):
    #get dummy but should drop the 'matchType' still
    df=df_pre[:]
    df=pd.concat([df, pd.get_dummies(df['matchType'])],axis=1)
    df.drop(columns=['matchType'], inplace=True)
    
    df =preprocess(df)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    #downsample
    nouse, use = train_test_split(df, 0.1)
    
    train, val = train_test_split(use, 0.1)
    
    model=linear_model.Lasso(alpha=alpha_use)
    model.fit(train[cols_to_fit], train[target]) 
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)

def run_linear_lassos_CV(alphas,df_pre):
    results = []
    for alpha_use in alphas:
        print(alpha_use)
        start = time.time()
        score = run_Linear_lasso_CV(drop2,alpha_use,df_pre) #decide to use drop2
        execution_time = time.time() - start
        results.append({
            'alpha': alpha_use,
            'error': score,
            'execution time': f'{round(execution_time, 2)}s'
        })
        gc.collect()
        
    return pd.DataFrame(results, columns=['alpha', 'error', 'execution time']).sort_values(by='error')
from sklearn.svm import SVR
def run_SVR_CV(preprocess,kernel_use,df_pre):
    #get dummy but should drop the 'matchType' still
    df=df_pre[:]
    #df=pd.concat([df, pd.get_dummies(df['matchType'])],axis=1)
    df.drop(columns=['matchType'], inplace=True)
    
    df =preprocess(df)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    #downsample
    nouse, use = train_test_split(df, 0.01)
    
    train, val = train_test_split(use, 0.1)
    
    model=SVR(kernel=kernel_use,C=1e3,gamma=0.1)
    model.fit(train[cols_to_fit], train[target]) 
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)

def run_SVRs_CV(kernels,df_pre):
    results = []
    for kernel_use in kernels:
        print(kernel_use)
        start = time.time()
        score = run_SVR_CV(drop2,kernel_use,df_pre) #decide to use drop2
        execution_time = time.time() - start
        results.append({
            'kernel': kernel_use,
            'error': score,
            'execution time': f'{round(execution_time, 2)}s'
        })
        gc.collect()
        
    return pd.DataFrame(results, columns=['kernel', 'error', 'execution time']).sort_values(by='error')
df_pre=reload()
from sklearn.neural_network import MLPRegressor
def run_NN_CV(preprocess,h_l_sizes,iter_use,df_pre):
    #get dummy but should drop the 'matchType' still
    df=df_pre[:]
    #df=pd.concat([df, pd.get_dummies(df['matchType'])],axis=1)
    df.drop(columns=['matchType'], inplace=True)
    
    df =preprocess(df)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    #downsample
    nouse, use = train_test_split(df, 0.01)
    
    train, val = train_test_split(use, 0.1)
    
    model=MLPRegressor(hidden_layer_sizes=h_l_sizes, max_iter=iter_use)
    model.fit(train[cols_to_fit], train[target]) 
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)

def run_NNs_CV(h_l_sizes_s,iters,df_pre):
    results = []
    for h_l_sizes in h_l_sizes_s:
        for iter_use in iters: 
            start = time.time()
            score = run_NN_CV(drop2,h_l_sizes,iter_use,df_pre) #decide to use drop2
            execution_time = time.time() - start
            results.append({
                'hidden_layer_sizes': h_l_sizes,
                'iter': iter_use,
                'error': score,
                'execution time': f'{round(execution_time, 2)}s'
            })
            gc.collect()
        
    return pd.DataFrame(results, columns=['hidden_layer_sizes', 'iter', 'error', 'execution time']).sort_values(by='error')
df_test_pre=reload_test()
#get the submission 
df=df_pre[:]
df_test=df_test_pre[:]

#df=pd.concat([df, pd.get_dummies(df['matchType'])],axis=1)
#df_test=pd.concat([df_test, pd.get_dummies(df_test['matchType'])],axis=1)
df=drop2(df)
df_test=drop2(df_test)

df.drop(columns=['matchType'], inplace=True)
df_test.drop(columns=['matchType'], inplace=True)
    
target = 'winPlacePerc'
cols_to_drop = ['Id', 'groupId', 'matchId', target]
cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
cols_to_drop_test = ['Id', 'groupId', 'matchId']
cols_to_fit_test = [col for col in df_test.columns if col not in cols_to_drop_test]
    #downsample
nouse, use = train_test_split(df, 0.01)    
train, val = train_test_split(use, 0.1)
start = time.time()   
model=RandomForestRegressor(n_estimators=300,max_depth=20) #best model
model.fit(train[cols_to_fit], train[target]) 
#validation error(test)
y_true = val[target]
y_pred = model.predict(val[cols_to_fit])
val_error=mean_absolute_error(y_true, y_pred)
y_pred_test = model.predict(df_test[cols_to_fit_test])
execution_time = time.time() - start
results=[]
results.append({
    'error': val_error,
    'execution time': f'{round(execution_time, 2)}s'
})
print(pd.DataFrame(results, columns=['error', 'execution time']).sort_values(by='error'))
sample=df_test[['Id']]
sample['winPlacePerc']=y_pred_test
sample.to_csv('submission1.csv',index=False)
