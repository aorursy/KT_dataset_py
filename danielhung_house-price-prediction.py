import pandas as pd
import numpy as np
import re

# Plotly
from plotly.offline import init_notebook_mode, iplot
import cufflinks as cf

init_notebook_mode()
cf.go_offline()
train = '../input/train.csv'
test = '../input/test.csv'
def findCorrelation(df, threshold=0.9):
    cor = df.corr()
    cor.loc[:,:] = np.tril(cor.values, k=-1)
    cor = cor.stack()
    return cor[abs(cor) > threshold].sort_values(ascending=False)
pd.read_csv(train).SalePrice.iplot(kind = 'hist',  xTitle = 'Sales Price')
def feature_extraction(df, threshold=0.4):
    df_train = pd.read_csv(train)
    df_train.fillna(df_train.mean(), inplace=True)
    df.fillna(df.mean(), inplace=True)
    
    feature = list(findCorrelation(df_train, threshold)['SalePrice'].index)
    df_x = df[feature].applymap(lambda x: 0 if x == 0 else np.log(x))
    
    col_cat = []
    for x in df.columns:
        if df[x].dtype == 'object':
            df = df.join(pd.get_dummies(df[x], prefix=x, drop_first=True))
            df_train = df_train.join(pd.get_dummies(df_train[x], prefix=x, drop_first=True))
            for x in list(pd.get_dummies(df[x], prefix=x, drop_first=True).columns):
                col_cat.append(x)
                
    col_cat.append('SalePrice')
    col_feature = list(findCorrelation(df_train[col_cat], threshold)['SalePrice'].index)

    return df_x.join(df[col_feature])
feature_extraction(pd.read_csv(train)).head()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

def rms (y_actual, y_predicted):
    return np.sqrt(mean_squared_error(np.log(y_actual), np.log(y_predicted)))

rmsele_scorer = make_scorer(rms, greater_is_better=False)
df = pd.read_csv(train)
yhat = [df['SalePrice'].mean()] * len(df)
y = df.SalePrice
baseline = rms(y, yhat)
baseline
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(train)
Xhat = feature_extraction(df)
y = df.SalePrice

rfg = RandomForestRegressor(random_state = 42)
param_grid = { 
    'n_estimators': np.arange(10,50,10),
    'max_depth': np.arange(10,50,10),
    "min_samples_leaf" : np.arange(2,10,1),
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion' : ['mae']
}
CV_rfg = GridSearchCV(estimator = rfg, param_grid = param_grid, cv = 5 , scoring=rmsele_scorer)
CV_rfg.fit(Xhat, y)

print (CV_rfg.best_params_)
print (abs(CV_rfg.best_score_))