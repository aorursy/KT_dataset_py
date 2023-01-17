import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt
from sklearn import preprocessing

train_file = '../input/train.csv'
test_file = '../input/test.csv'
store_file = '../input/store.csv'
output_file = 'predictions.csv'

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )
store = pd.read_csv(store_file)

train = pd.merge(train,store,on='Store')
test = pd.merge(test,store,on='Store')



train['Date'] = pd.to_datetime(train['Date'], coerce=True)
test['Date'] = pd.to_datetime(test['Date'], coerce=True)

train['year'] = pd.DatetimeIndex(train['Date']).year
train['month'] = pd.DatetimeIndex(train['Date']).month

test['year'] = pd.DatetimeIndex(test['Date']).year
test['month'] = pd.DatetimeIndex(test['Date']).month

train['logSale'] = np.log1p(train.Sales)




le = preprocessing.LabelEncoder()
le.fit(['a','b','c','d'])
train['labled_StoreType'] = le.transform(train.StoreType)
test['labled_StoreType'] = le.transform(test.StoreType)

le.fit(['a','b','c'])
train['labled_Assortment'] = le.transform(train.Assortment)
test['labled_Assortment'] = le.transform(test.Assortment)

train_holiday_in_str = train.StateHoliday.astype(str)
test_holiday_in_str = test.StateHoliday.astype(str)
le.fit(['0','a','b','c'])
train['labled_StateHoliday'] = le.transform(train_holiday_in_str)
test['labled_StateHoliday'] = le.transform(test_holiday_in_str)

 
train.dtypes
train.drop(['Date','Sales','logSale','Customers', 'PromoInterval', 'StateHoliday', 'StoreType', 'Assortment'], axis=1, inplace=True);
train.drop(['CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear'], axis=1 , inplace=True);

train.dtypes
test.drop(['Id','Date', 'PromoInterval', 'StateHoliday', 'StoreType', 'Assortment'], axis=1, inplace = True);
test.drop(['CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear'], axis=1, inplace = True);


test.dtypes
test.Open.fillna(1, inplace=True);
rf = RandomForestClassifier(n_estimators=100, max_depth=30)
rf.fit(train,output)
test.drop(['Id'],axis=1,inplace=True)
