%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.structured import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from IPython.core.debugger import set_trace
from sklearn.model_selection import KFold

import os
print(os.listdir("../input"))
PATH = "../input/"
df_train = pd.read_csv(f'{PATH}train.csv', parse_dates=['Open Date'])
df_test = pd.read_csv(f'{PATH}test.csv', parse_dates=['Open Date'])

df_joined = pd.concat([df_train.drop('revenue', axis=1), df_test], axis=0)
def score(model, X_train, y_train, X_valid = [], y_valid = []):
    #set_trace()
    rms = sqrt(mean_squared_error(np.square(np.exp(y_train)), np.square(np.exp(model.predict(X_train)))))
    score = [rms, model.score(X_train, y_train)]
    
    if len(X_valid) != 0 and len(y_valid) != 0:
        score.append(sqrt(mean_squared_error(np.square(np.exp(y_valid)), np.square(np.exp(model.predict(X_valid))))))
        
    if model.oob_score:
        score.append(model.oob_score_)
    
    return score


n_train = df_train.shape[0]

def prcs(df, fe=[]):
    add_datepart(df, 'Open Date')
    
    if 'city' in fe:
        df = df.drop('City', axis=1)
    # Quitamos el outlier (16)
    if 'outlier' in fe:
        df = df.drop(index=16, axis=0)
    
    if 'MB' in fe:
        # No hay apenas tipo 'MB'
        df['Type'] = df['Type'].replace('MB', 'DT')
    
    if 'city_group' in fe:
        df = df.drop('City Group', axis=1)
    
    if 'dummies' in fe:
        #Get dummies
        p_cols = [ f'P{n}' for n in range(1,38)]
            
        df = pd.get_dummies(df, columns=p_cols)
        if 'city_group' not in fe:
            df = pd.get_dummies(df, columns=['City Group'], drop_first=True)
        df = pd.get_dummies(df, columns=['Type'])
    
    #Train cats
    train_cats(df)

    X, _, _ = proc_df(df, None)
    drop_cols = ['Open Year', 'Open Month', 'Open Week', 'Open Day', 'Open Dayofweek',
       'Open Dayofyear', 'Open Is_month_end', 'Open Is_month_start',
       'Open Is_quarter_end', 'Open Is_quarter_start', 'Open Is_year_end',
       'Open Is_year_start']
    
    X = X.drop(drop_cols, axis=1)
    # La columna Id no aporta nada
    if 'id' in fe:
        X = X.drop('Id', axis=1)
    
    if 'scale_open' in fe:
        X['Open Elapsed'] = (X['Open Elapsed']/1000).apply(np.log)
    
    X_train = X[:n_train]
    X_test = X[n_train:]
    
    return X_train, X_test

def train_cv(X, y):
    models = []
    scores = []
    
    kf = KFold(n_splits=4, random_state=12, shuffle=False)
    for train_index, val_index in kf.split(X):
        X_train_ = X.iloc[train_index]
        y_train_ = y.iloc[train_index]
        X_val_ = X.iloc[val_index]
        y_val_ = y.iloc[val_index]
        m = RandomForestRegressor(n_jobs=-1, n_estimators=100, max_features=0.5, oob_score=True)
        m.fit(X_train_, y_train_)
        models.append(m)
        scores.append(score(m, X_train_, y_train_, X_val_, y_val_))
        
    return models, np.array(scores).mean(axis=0)

def predict(models, X):
    f = 1 / len(models)
    pred = 0
    for m in models:
        pred += f * m.predict(X)
    
    return pred
X_train, X_test = prcs(df_joined.copy())
y_train = df_train['revenue'].copy().apply(np.log)
m = RandomForestRegressor(n_jobs=-1, n_estimators=150, oob_score=True, max_features=0.5)
m.fit(X_train, y_train)
score(m,X_train, y_train)
df_preds = pd.DataFrame(columns=['Prediction'],index=X_test.index, data=np.exp(predict(m, X_test)))
df_preds.to_csv('submission0.csv', index=True, index_label='Id')
df_preds.head()
models, scores = train_cv(X_train, y_train)
print(scores)
df_preds = pd.DataFrame(columns=['Prediction'],index=X_test.index, data=np.exp(predict(models, X_test)))
df_preds.to_csv('submission1.csv', index=True, index_label='Id')
df_preds.head()
X_train, X_test = prcs(df_joined.copy(), fe=['id'])

# Doble transformación para que la distribución sea Normal
y_train = df_train['revenue'].copy().apply(np.sqrt).apply(np.log)
models, scores = train_cv(X_train, y_train)
print(scores)
df_preds = pd.DataFrame(columns=['Prediction'],index=X_test.index, data=np.square(np.exp(predict(models, X_test))))
df_preds.to_csv('submission2.csv', index=True, index_label='Id')
df_preds.head()
X_train, X_test = prcs(df_joined.copy(), fe=['id', 'dummies'])

# Doble transformación para que la distribución sea Normal
y_train = df_train['revenue'].copy().apply(np.sqrt).apply(np.log)
models, scores = train_cv(X_train, y_train)
print(scores)
df_preds = pd.DataFrame(columns=['Prediction'],index=X_test.index, data=np.square(np.exp(predict(models, X_test))))
df_preds.to_csv('submission3.csv', index=True, index_label='Id')
df_preds.head()
X_train, X_test = prcs(df_joined.copy(), fe=['id', 'dummies', 'city'])

# Doble transformación para que la distribución sea Normal
y_train = df_train['revenue'].copy().apply(np.sqrt).apply(np.log)
models, scores = train_cv(X_train, y_train)
print(scores)
df_preds = pd.DataFrame(columns=['Prediction'],index=X_test.index, data=np.square(np.exp(predict(models, X_test))))
df_preds.to_csv('submission4.csv', index=True, index_label='Id')
df_preds.head()
X_train, X_test = prcs(df_joined.copy(), fe=['id', 'dummies', 'city', 'city_group'])

# Doble transformación para que la distribución sea Normal
y_train = df_train['revenue'].copy().apply(np.sqrt).apply(np.log)
models, scores = train_cv(X_train, y_train)
print(scores)
df_preds = pd.DataFrame(columns=['Prediction'],index=X_test.index, data=np.square(np.exp(predict(models, X_test))))
df_preds.to_csv('submission5.csv', index=True, index_label='Id')
df_preds.head()