%matplotlib inline
'''
Based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
'''

import pandas as pd
import numpy as np
import xgboost as xgb
import uuid
from sklearn import cross_validation
from datetime import date, timedelta
from sklearn.cross_validation import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from scipy.stats import pearsonr
import numpy as np
from sklearn.cluster import AgglomerativeClustering



def pearson_affinity(M):
    return 1 - np.array([[pearsonr(a,b)[0] for a in M] for b in M])

def factor(series):
    #input should be a pandas series object
    dic = {}
    for i,val in enumerate(series.value_counts().index):
        dic[val] = i
    return [ dic[val] for val in series.values ]  




# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

def XGBoost(X_train,X_valid,params,verbose=False):
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
    dvalid = xgb.DMatrix(X_valid[features], np.log(X_valid["Sales"] + 1))
    num_trees = params['num_trees']
    
    watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
    #watchlist = [(dvalid, 'eval'),(dtrain, 'train')]

    gbm = xgb.train(params,
                    dtrain,
                    num_trees,
                    evals=watchlist,
                    early_stopping_rounds=10,
                    feval=rmspe_xg,
                    verbose_eval=verbose,
                    )
    
    train_probs = gbm.predict(xgb.DMatrix(X_train[features]),ntree_limit=gbm.best_iteration)

    train_error = rmspe(np.exp(train_probs) - 1, X_train['Sales'].values)
    
    valid_probs = gbm.predict(xgb.DMatrix(X_valid[features]),ntree_limit=gbm.best_iteration)
    indices = valid_probs < 0
    valid_probs[indices] = 0
    valid_error = rmspe(np.exp(valid_probs) - 1, X_valid['Sales'].values)
    return gbm, valid_error, train_error



# Gather some features
def build_features(features, data, dates):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
        
    # Use some properties directly
    features.extend(['Store', 'CustomerCluster', 'SalesCluster','CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2SinceWeek', 'Promo2SinceYear'])
    
    #log of CompetitionDistance
    features.append('logDist')
    data['logDist'] = np.log(1+data.CompetitionDistance)

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    #features.append('StateHoliday')
    StateHolidayDict= { 0:0 , '0' : 0 , 'a': 1 , 'b' : 2 , 'c' : 3}
    data['StateHoliday'] =  [ StateHolidayDict[i]  for i in data.StateHoliday.values ]

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['Date'] = pd.to_datetime(data.Date)
    data['Date'] = pd.DatetimeIndex(data.Date)
    data = data.join(dates,on='Date')

    #features.append('StoreType')
    StoreTypeDict  = { 'a' : 0 ,'b' : 1 , 'c' : 2 , 'd':3 }
    data['StoreType']  = [ StoreTypeDict[i]  for i in data.StoreType.values ]

    #features.append('Assortment')
    AssortmentDict = { 'a' : 0 ,'b' : 1 , 'c' : 2 }
    data['Assortment'] = [ AssortmentDict[i] for i in data.Assortment.values]
    
    features.append('AssortStoreType')
    data['AssortStoreType'] = data['Assortment'] + 10*data['StoreType']
    
    return data


print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
store = pd.read_csv("../input/store.csv")

print("Generate Dates Table")
dates = pd.DataFrame(pd.date_range(train.Date.min(),test.Date.max()),columns=['Date']).set_index('Date')
dates['day']   = dates.index.day.astype(int)
dates['month'] = dates.index.month.astype(int)
dates['year']  = dates.index.year.astype(int)

print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print('Cluster stores by sales correlation.')
sales_pivot  = pd.pivot_table(train,values='Sales',index='Date', columns=['Store'],aggfunc='mean').dropna()
print(sales_pivot.head())
sales_corr   = sales_pivot.corr()
cluster = AgglomerativeClustering(n_clusters=50, linkage='average',affinity=pearson_affinity).fit(sales_corr)
store['SalesCluster'] = cluster.labels_

print('Cluster stores by customer correlation.')
cust_pivot  = pd.pivot_table(train,values='Customers',index='Date', columns=['Store'],aggfunc='mean').dropna()
print(cust_pivot.head())
cust_corr   = sales_pivot.corr()
cluster = AgglomerativeClustering(n_clusters=50, linkage='average',affinity=pearson_affinity).fit(cust_corr)
store['CustomerCluster'] = cluster.labels_


print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("Augment features")
train = build_features(features, train, dates)
test  = build_features([], test, dates)
print(features)

print('Reduce data for optimization.')
#train = train[train.Store < 200] 
train.index = range(len(train))
train.head()

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]

print("Consider only days with sales for training.")
train = train[train["Sales"] != 0]

#Cross validation
train.sort(['Date','Store'],inplace=True)
index = range(len(train))
train.index = index
print(train.head())
trainSize = 0.99
trainIds, validIds =  train.index[:len(train)*trainSize] , train.index[len(train)*trainSize:] 
#validIds, testIds     =  train_test_split(validTestIds,train_size=0.50)
testIds = []

trainIds = list(trainIds)

print("Size Train:", len(trainIds))
print("Size Valid:", len(validIds))
print("Size  Test:", len(testIds))
assert len(trainIds)+len(testIds)+len(validIds) == len(train)

plt.scatter(trainIds,[1]*len(trainIds),marker='+',color='b',label='Train')
plt.scatter(validIds,[1]*len(validIds),marker='+',color='r',label='Valid')
plt.scatter( testIds,[1]*len(testIds) ,marker='+',color='g',label='Test')
plt.legend(loc=2)

# Parameter optimization

params = {'base_score': 0, 
          'alpha': 0, 
          'booster': 'gbtree', 
          'colsample_bytree': 0.8, 
          'silent': 1,
          'subsample': 0.9,
          'eta': 0.2, 
          'num_trees': 10000, 
          'objective': 'reg:linear', 
          'max_depth': 12, 
          'lambda': 1,
          'nthread':None}



gbm,error_valid,error_train = XGBoost(train.loc[trainIds],train.loc[validIds],params,verbose=True)

import operator
importance = gbm.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = 100. * df['fscore'] / df['fscore'].max()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 20))

# Make Prediction

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(xgb.DMatrix(test[features]),ntree_limit=gbm.best_iteration)
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
submission_id = uuid.uuid4()
print(submission_id)
submission.to_csv("xgboost_script_submission_%s.csv" %(submission_id), index=False)

