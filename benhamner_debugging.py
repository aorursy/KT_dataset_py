%matplotlib inline
# Debugging cast42's script from https://www.kaggle.com/c/rossmann-store-sales/forums/t/17437/basic-time-series-exploration-notebook

import pandas as pd
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
from seaborn import set_style
set_style("darkgrid")
import seaborn as sns
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])

    # Label encode some features
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    # Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Okt', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return data
print("Load the training, test and store data using pandas")
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(int),
         'PromoInterval': np.dtype(str)}
train = pd.read_csv("../input/train.csv", parse_dates=[2], dtype=types)
test = pd.read_csv("../input/test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("../input/store.csv")
print("Assume store open, if not provided")
test.fillna(1, inplace=True)

# print("Consider only open stores for training. Closed stores wont count into the score.")
# train = train[train["Open"] != 0]
# print("Use only Sales bigger then zero")
# train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
train = build_features(features, train)
test = build_features([], test)
print(features)

print('training data processed')
print ('From',test.Date.min(),'to', test.Date.max())
print ('That is', test.Date.max()-test.Date.min(), 'days')
test.Store.nunique()

rS = 979 # rS =  random.choice(train.Store.unique())
print ('Random store number =', rS)
train.Year.unique()
train[(train.Store==rS) & (train.Year==2013)].Sales.plot(label='2013', figsize=(18,4))
plt.title('Store {}'.format(rS))
plt.show()
train.set_index('Date', inplace=True)

st = train[train.Store==rS] # Select store rS
st['2013']['Sales'].plot(label='2013', figsize=(18,4), title='Store {}'.format(rS))
plt.show()
train[train.DayOfWeek==6].Sales.sum()
salesOnSundayPerStore = train[(train.Open) & (train.DayOfWeek==6)].groupby('Store')['Sales']
salesOnSundayPerStore.count().sort_values().plot(kind='barh')
plt.title('Number of sunday open per store')
plt.show()
train[(train.Store==85) & (train.DayOfWeek==6)].Sales.plot(figsize=(18,4))
plt.title('Sales of store 85 on sundays')
plt.show()
# fig, axes = plt.subplots(3, 1, figsize=(18, 4));
def plotStore(rS):
    st = train[train.Store==rS]
    storerS13 = st[st.Year==2013].Sales.reset_index(drop=True)
    storerS14 = st[st.Year==2014].Sales.reset_index(drop=True)
    storerS15 = st[st.Year==2015].Sales.reset_index(drop=True)

    df_plot = pd.concat([storerS13, storerS14, storerS15], axis=1)
    df_plot.columns = ['2013', '2014', '2015']
    df_plot.index = pd.date_range('1/1/2015', periods=365, freq='D')
    df_plot.plot(subplots=True,figsize=(18, 6), title='Sales at store {}'.format(rS))
    plt.show()
plotStore(rS)
rS =  85 #random.choice(train.Store.unique())
plotStore(rS)
plotStore(769)

trainStore = train[train.Store == rS]
prevy1 = trainStore.ix['2014-08-02':'2014-09-18']['Sales'].reset_index(drop=True)
prevy2 = trainStore.ix['2013-08-03':'2013-09-19']['Sales'].reset_index(drop=True)
meanSales = np.mean(np.vstack((prevy1, prevy2)), axis=0)
df_plot = pd.DataFrame(meanSales, index = pd.date_range('8/1/2015', periods=48, freq='D'))
df_plot.columns = ['Prediction']
df_plot.plot(title='Prediction for store {}'.format(rS));
rS = 1013  # rS =  random.choice(train.Store.unique())
storerS13 = train[(train.Store==rS) & (train.Year==2013)].Sales.reset_index(drop=True)
storerS14 = train[(train.Store==rS) & (train.Year==2014)].Sales.reset_index(drop=True)
storerS15 = train[(train.Store==rS) & (train.Year==2015)].Sales.reset_index(drop=True)

df_plot = pd.concat([storerS13, storerS14, storerS15], axis=1)
df_plot.columns = ['2013', '2014', '2015']
df_plot.index = pd.date_range('1/1/2015', periods=365, freq='D')
df_plot['pred'] = pd.DataFrame(meanSales, index = pd.date_range('8/1/2015', periods=48, freq='D'))
df_plot.plot(subplots=True,figsize=(18, 6), title='Sales at store {}'.format(rS))
plt.show()
