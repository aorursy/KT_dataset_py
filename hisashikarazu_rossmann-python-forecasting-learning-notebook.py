import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
%matplotlib inline
from fbprophet import Prophet
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 99
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox
from xgboost import XGBRegressor

# importing train data to learn
train = pd.read_csv("../input/train.csv", 
                    parse_dates = True, low_memory = False, index_col = 'Date')

# additional store data
store = pd.read_csv("../input/store.csv", 
                    low_memory = False)
test=pd.read_csv("../input/test.csv",parse_dates = True, low_memory = False, index_col = 'Date')
# time series as indexes
train.index
#look at stores
store.drop('PromoInterval',axis=1,inplace=True)
dummies1=pd.get_dummies(store['StoreType'],prefix='StoreType', drop_first=True)
dummies2=pd.get_dummies(store['Assortment'],prefix='Assortment', drop_first=True)
store=pd.concat([store,dummies1,dummies2],axis=1)
store.drop(['StoreType','Assortment'],axis=1,inplace=True)
store.replace(np.nan, 0,inplace=True)
store.head()
#trial=store[store['CompetitionOpenSinceYear']!=0.0]
#trial['compdate']=pd.to_datetime(dict(year=trial.CompetitionOpenSinceYear, month=trial.CompetitionOpenSinceMonth, day=1))
#store['compdate']=trial['compdate']
#trial.compdate.replace('Nat',0,inplace=True)
#store.head()
test.dtypes
test['Open']=test['Open'].fillna(0).astype(int)
test['Sales']=0
test['Customers']=0
test=test[['Id','Store','DayOfWeek','Sales','Customers','Open','Promo','StateHoliday','SchoolHoliday']]
test.head()

train['Id']=-1
train=train[['Id','Store','DayOfWeek','Sales','Customers','Open','Promo','StateHoliday','SchoolHoliday']]
train.head()
df1=pd.concat([train,test])
df1.head()
df1=df1.sort_index(ascending=True)
df1.head(15)
# data extraction
df1['Year'] = df1.index.year
df1['Month'] = df1.index.month
df1['Day'] = df1.index.day
df1['WeekOfYear'] = df1.index.weekofyear
dummies=pd.get_dummies(df1['StateHoliday'],prefix='HolidayType', drop_first=True)
df1=pd.concat([df1,dummies],axis=1)
df1.drop(['StateHoliday','Customers'],axis=1,inplace=True)
df1.head()

def generate_base_forecasts (df):
    newdf=[]
    for s in df['Store'].unique():
        work=df1[df1['Store']==s]
        work['MA90'] = work[work['Open']==1]['Sales'].rolling(window=90).mean()
        work['MA120'] = work[work['Open']==1]['Sales'].rolling(window=120).mean()
        work['Lag60']=work[work['Open']==1]['Sales'].shift(60)
        work['Lag90']=work[work['Open']==1]['Sales'].shift(90)
        work['Sqrt']=np.sqrt(work[work['Open']==1]['MA120'])
        work['Log']=np.log(work[work['Open']==1]['MA120'])
        #work['box']=boxcox(work[work['Open']==1]['MA120'])
        work.replace(np.nan, 0,inplace=True)
        newdf.append(work)
    return pd.concat(newdf)

df2=generate_base_forecasts(df1)      
df2.head()
df2[['Sales', 'MA90', 'MA120', 'Lag60', 'Lag90']]=np.log1p(df2[['Sales', 'MA90', 'MA120', 'Lag60', 'Lag90']])
df2.head()
Readydf=df2.merge(store, on='Store').set_index(df2.index)
my_imputer = SimpleImputer(strategy='median')
#my_scaler=StandardScaler()
cols=['Sales', 'MA90', 'MA120', 'Lag60', 'Lag90']
Readydf[cols]=my_imputer.fit_transform(Readydf[cols])
#Readydf[cols]=my_scaler.fit_transform(Readydf[cols])
Readydf.head(20)
train2=Readydf.loc['2013-01-01':'2015-07-31']
test2=Readydf.loc['2015-07-31':'2015-09-17']
y = train2.Sales
X = train2.drop(['Sales'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)


from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
predictions
X2 = test2.drop(['Sales'], axis=1)
X2.head()
X2['Sales']=np.expm1(my_model.predict(X2.as_matrix()).clip(0))

#X2[cols]=my_scaler.inverse_transform(X2[cols])
X2.head()
testE=pd.read_csv("../input/test.csv",parse_dates = True, low_memory = False, index_col = 'Date')

results=testE.merge(X2,on='Id')
results.head()
submit = results[['Id','Sales']]
submit.to_csv("submission_xgb.csv", index=False)
submit.info()

submit.tail(100)
#def prophesize (df):
   # newdf=[]
    #for s in df['Store'].unique():
        #temp=df[df['Store']==s]
        #proph_train = temp['2013-01-01':'2015-07-31'].reset_index().iloc[:,[0,4]]
        #proph_train.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)
        #m = Prophet(daily_seasonality=True)
        #m.fit(proph_train[['ds', 'y']])
        #future = m.make_future_dataframe(periods=len(temp['2015-07-31':'2015-09-17']),freq='D',include_history=False)
        #fcst = m.predict(future)
        #temp['yhat']=0
        #temp['yhat']['2015-07-31':'2015-09-17']=fcst['yhat'].values
        #newdf.append(temp)
    #return pd.concat(newdf)
#prophet=prophesize(df2)
#np.expm1