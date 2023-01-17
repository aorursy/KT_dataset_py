import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler 

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor
import time
import gc
from scipy.stats import uniform
import calendar
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', 100)
train = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")
train.head()
store = pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv")
store.head()
train.isnull().sum()
store.isnull().sum()
#store.shape
#store.dropna(inplace=True)
#store.shape
store['StoreType'].value_counts()
store['Assortment'].value_counts()
store['StoreType']= store['StoreType'].map({'a':1, 'b' : 2, 'c': 3, 'd' : 4})
store['Assortment'] = store['Assortment'].map({'a':1, 'b' : 2, 'c': 3})
store.head()
data = pd.merge(train, store,on = 'Store', how='left')
data.head()
data.shape
data['StateHoliday'].value_counts()
#Tried with reducing the dataset by taking only the records have Sales > 0.
#still system is getting longer time for procesing the model

#data = data[data['Sales'] > 0]

data.dropna(inplace = True)
data.shape
# credits to kaggle link on specifying how to handle date and month values
# https://www.kaggle.com/rohinigarg/random-forest-and-xgboost-parameter-tuning

def checkpromomonth(row):
 if (row['MonthName'] in row['PromoInterval']):
    return 1
 else:
    return 0

def ProcessData(data):
    data["CompetitionDistance"].fillna(data["CompetitionDistance"].mean(), inplace = True)
    
    data['StateHoliday']= data['StateHoliday'].map({'0':0, 0: 0,'a':1, 'b' : 2, 'c': 3})
    
    data['Date']=pd.to_datetime(data['Date'])
    data['Year']=data['Date'].dt.year
    data['MonthNumber']=data['Date'].dt.month
    data['MonthName']=data['MonthNumber'].apply(lambda x: calendar.month_abbr[x])
    data['Day']=data['Date'].dt.day
    data['WeekNumber']=data['Date'].dt.weekofyear

    data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (data['MonthNumber'] - data['CompetitionOpenSinceMonth'])
    data['CompetitionOpen'] = data['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)

    data['Promo2Open'] = 12 * (data['Year'] - data['Promo2SinceYear']) + (data['WeekNumber'] - data['Promo2SinceWeek']) / float(4)
    data['Promo2Open'] = data['Promo2Open'].apply(lambda x: x if x > 0 else 0)

    data['PromoInterval']=data['PromoInterval'].astype(str)
    
    data['IsPromoMonth'] =  data.apply(lambda row: checkpromomonth(row),axis=1)

    data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis = 1,  inplace = True)
    data.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis = 1,  inplace = True)
    data.drop(['Date', 'MonthName','PromoInterval'], axis = 1,  inplace = True)
ProcessData(data)
data.head()
data.isnull().sum()
data.shape
data.min().min()           
data.max().max() 
data = data.astype('int32')
data.info()
y = data['Sales']
data.drop(['Sales','Customers'], axis = 1,  inplace = True)
data.nunique()
num_columns = data.columns[data.nunique() > 12]
cat_columns = data.columns[data.nunique() <= 12]
num_columns
cat_columns
plt.figure(figsize=(15,10))
sns.distributions._has_statsmodels=False
for i in range(len(num_columns)):
    plt.subplot(2,3,i+1)
    sns.distplot(data[num_columns[i]])
    
plt.tight_layout()
ct=ColumnTransformer([
    ('rs',RobustScaler(),num_columns),
    ('ohe',OneHotEncoder(),cat_columns),
    ],
    remainder="passthrough"
    )
ct.fit_transform(data)
X=data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
steps_xg = [('sts', StandardScaler() ),
            ('pca', PCA()),
            ('xg',  XGBRegressor(objective='reg:squarederror',silent = False, n_jobs=3, reg_lambda=1,gamma=0))
            ]

pipe_xg = Pipeline(steps_xg)

pipe_xg.get_params()
#credit : https://www.kaggle.com/tushartilwankar/sklearn-rf
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def RMSPE(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe 
#from sklearn.metrics import make_scorer, r2_score, mean_squared_error
#Randomized Search
parameters = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,300),
              'xg__max_depth':      range(3,10),
              'pca__n_components' : range(10,17)}

rs = RandomizedSearchCV(pipe_xg,
                        param_distributions=parameters,
                        #scoring=make_scorer(mean_squared_error, squared=False),
                        #scoring= RMSPE,
                        n_iter=15,    
                        verbose = 1,
                        #refit = RMSPE,
                        n_jobs = 3,
                        cv = 3              
                        )
start = time.time()
rs.fit(X_train, y_train)
end = time.time()
(end - start)/60 
rs.best_estimator_.named_steps["xg"].feature_importances_
rs.best_estimator_.named_steps["xg"].feature_importances_.shape
# Model with parameters of random search
model_rs = XGBRegressor(objective='reg:squarederror',silent = False, n_jobs=3, reg_lambda=1,gamma=0,
                    learning_rate = rs.best_params_['xg__learning_rate'],
                    max_depth = rs.best_params_['xg__max_depth'],
                    n_estimators=rs.best_params_['xg__max_depth']
                    )


model_rs.fit(X_train, y_train)
y_pred_rs = model_rs.predict(X_test)
RMSPE(y_test,y_pred_rs)

rs.best_score_
import math
accuracy_rs =  math.sqrt(sum((y_test - y_pred_rs)**2)/y_test.count())
print("Accuracy with Random search XGB model:",accuracy_rs*100)
X_test_df = X_test.reset_index()
y_test_df = y_test.reset_index()
y_pred_df  = pd.DataFrame(y_pred_rs)

final = X_test_df
#final
final = final.merge(y_test_df, left_index=True, right_index=True)
final = final.merge(y_pred_df, left_index=True, right_index=True)
final
test = pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv")
test.head()

test.isnull().sum()
test.shape
test.Open.fillna(0, inplace= True)
test.isnull().sum()
store.head()
data = pd.merge(test, store,on = 'Store', how='left')
data.head()

data.shape
ProcessData(data)

data.head()
submission = data['Id']
data=data.drop('Id',axis=1)
data.head()
data.info()
data.min().min()           
data.max().max()
data = data.astype('int32')
data.nunique() 
num_columns = data.columns[data.nunique() > 12]
cat_columns = data.columns[data.nunique() <= 12]
num_columns
cat_columns
plt.figure(figsize=(15,10))
sns.distributions._has_statsmodels=False
for i in range(len(num_columns)):
    plt.subplot(2,3,i+1)
    graph = sns.distplot(data[num_columns[i]])
    
plt.tight_layout()


ct=ColumnTransformer([
    ('rs',RobustScaler(),num_columns),
    ('ohe',OneHotEncoder(),cat_columns),
    ],
    remainder="passthrough"
    )
ct.fit_transform(data)

y_pred_rs = model_rs.predict(data)
y_pred_rs
final = submission.reset_index()
y_pred_df  = pd.DataFrame(y_pred_rs)

final = final.merge(y_pred_df, left_index=True, right_index=True)
final
final.to_csv('submission.csv')
