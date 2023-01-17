'''
Based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
Score : 0.
'''


import pandas as pd
import numpy as np
from sklearn import cross_validation
import xgboost as xgb
from datetime import date, timedelta


#set random seed for reproducibility
np.random.seed(14566433)

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

def factor(series):
    dic = {}
    for i,val in enumerate(series.value_counts().index):
        dic[val] = i
    return [ dic[val] for val in series.values ]   


# Gather some features
def build_features(features, data, dates):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
        
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    #features.append('SchoolHoliday')
    #data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    
    features.append('StateHoliday')
    data['StateHoliday'] = factor(data.StateHoliday)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['Date'] = pd.to_datetime(data.Date)
    data['Date'] = pd.DatetimeIndex(data.Date)
    data = data.join(dates,on='Date')

    features.append('StoreType')
    data['StoreType'] = factor(data.StoreType) 

    features.append('Assortment')
    data['Assortment'] = factor(data.Assortment) 
    
    #add data from yesterday and tomorrow
    features.extend(['OpenY','dayY',
                     'OpenT','dayT'])
                    
    yesterday = data.copy(deep=True)
    yesterday['Date'] = yesterday.Date + timedelta(days=1)
    yesterday = yesterday.set_index(['Store','Date'])

    tomorrow = data.copy(deep=True)
    tomorrow['Date'] = tomorrow.Date - timedelta(days=1)
    tomorrow = tomorrow.set_index(['Store','Date'])
    data = data.join(yesterday[['Open','day']], on=['Store','Date'],rsuffix='Y')
    data = data.join(tomorrow[['Open','day']], on=['Store','Date'],rsuffix='T')
   
    # remove new NaNs
    # maybe assume smarter values later
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1    
    
    return data


print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
store = pd.read_csv("../input/store.csv")

print("Generate Dates Table")
dates = pd.DataFrame(pd.date_range(train.Date.min(),test.Date.max()),columns=['Date']).set_index('Date')
dates['day']   = dates.index.day.astype(int)
dates['month'] = dates.index.month.astype(int)
dates['year']  = dates.index.year.astype(int)

print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("Augment features")
train = build_features(features, train, dates)
test  = build_features([], test, dates)
print(features)

params = {"objective": "reg:linear",
          "eta": 0.1,
          "max_depth": 10,
          "subsample": 0.85,
          "colsample_bytree": 0.75,
          "silent": 1
          }
num_trees = 2000

print("Train a XGBoost model")
val_size = 100000
train = train.sort(['Date'])
print(train.tail(1)['Date'])


X_train, X_test = cross_validation.train_test_split(train, test_size=0.01)
#X_train, X_test = train.head(len(train) - val_size), train.tail(val_size)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)

print("Validating")
train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
print('error', error)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test[features]))
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
submission.to_csv("xgboost_kscript_submission.csv", index=False)