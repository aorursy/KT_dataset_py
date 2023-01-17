%matplotlib inline
import pandas as pd
import numpy as np
import datetime
import random
!pip freeze | grep pandas
import matplotlib.pyplot as plt
%matplotlib inline
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
rS = 979
train[(train.Store==rS) & (train.Year==2013)].Sales.plot(label='2013', figsize=(16,4))
plt.title('Store {}'.format(rS))
plt.show()
train.set_index('Date', inplace=True)
st = train[train.Store==rS] # Select store rS
st['2013']['Sales'].plot(label='2013', figsize=(17,4), title='Store {}'.format(rS))
plt.show()
train[train.DayOfWeek==6].Sales.sum()
salesOnSundayPerStore = train[(train.Open) & (train.DayOfWeek==6)].groupby('Store')['Sales']
salesOnSundayPerStore.count().sort_values().plot(kind='barh')
plt.title('Number of sunday open per store')
plt.show()
train[(train.Store==85) & (train.DayOfWeek==6)].Sales.plot(figsize=(17,4))
plt.title('Sales of store 85 on sundays')
plt.show()
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
plotStore(979)
rS = 1013  # rS =  random.choice(train.Store.unique())
plotStore(1013)
rS =  85 #random.choice(train.Store.unique())
plotStore(rS)
plotStore(769)
rS = 1013
trainStore = train[train.Store == rS]
prevy1 = trainStore.ix['2014-08-02':'2014-09-18']['Sales'].reset_index(drop=True)
prevy2 = trainStore.ix['2013-08-03':'2013-09-19']['Sales'].reset_index(drop=True)
meanSales = np.mean(np.vstack((prevy1, prevy2)), axis=0)
df_plot = pd.DataFrame(meanSales, index = pd.date_range('8/1/2015', periods=48, freq='D'))
df_plot.columns = ['Prediction']
df_plot.plot(title='Prediction for store {}'.format(rS));
# Adapt above code so it runs on Kaggle
rS = 1013
periodym1 = train.ix['2014-08-02':'2014-09-18']
periodym2 = train.ix['2013-08-03':'2013-09-19']
prevy1 = periodym1[periodym1.Store == rS]['Sales'].reset_index(drop=True)
prevy2 = periodym2[periodym2.Store == rS]['Sales'].reset_index(drop=True)
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
def plotTrainPred(rS, pred, title=None):
    trainStore = train[train.Store==rS]
    plotIndex = pd.date_range('1/1/2015', periods=270, freq='D')
    df_plot = pd.DataFrame(trainStore['2015']['Sales'], index = plotIndex)
    df_plot.columns = ['2015']
    predIndex = pd.date_range('8/1/2015', periods=48, freq='D')
    df_plot['pred'] = pd.DataFrame(pred, index = predIndex)
    df_plot['2015'].plot(label='train')
    if title:
        df_plot['pred'].plot(label='pred', figsize=(17, 5), title=title)
    else:
        df_plot['pred'].plot(label='pred', figsize=(17, 5), title='Sales at store {} in 2015'.format(rS))
    plt.legend();

plotTrainPred(1013, meanSales)
rS = 344
periodym1 = train.ix['2014-08-02':'2014-09-18']
periodym2 = train.ix['2013-08-03':'2013-09-19']
prevy1 = periodym1[periodym1.Store == rS]['Sales'].reset_index(drop=True)
prevy2 = periodym2[periodym2.Store == rS]['Sales'].reset_index(drop=True)
meanSales = np.mean(np.vstack((prevy1, prevy2)), axis=0)

plotTrainPred(344, meanSales)
rs= 876 # rS =  random.choice(train.Store.unique())
periodym1 = train.ix['2014-08-02':'2014-09-18']
periodym2 = train.ix['2013-08-03':'2013-09-19']
prevy1 = periodym1[periodym1.Store == rS]['Sales'].reset_index(drop=True)
prevy2 = periodym2[periodym2.Store == rS]['Sales'].reset_index(drop=True)
meanSales = np.mean(np.vstack((prevy1, prevy2)), axis=0)

plotTrainPred(876, meanSales)
rS = 265 # random.choice(train.Store.unique())
periodym1 = train.ix['2014-08-02':'2014-09-18']
periodym2 = train.ix['2013-08-03':'2013-09-19']
prevy1 = periodym1[periodym1.Store == rS]['Sales'].reset_index(drop=True)
prevy2 = periodym2[periodym2.Store == rS]['Sales'].reset_index(drop=True)
meanSales = np.mean(np.vstack((prevy1, prevy2)), axis=0)

storerS13 = train[(train.Store==rS) & (train.Year==2013)].Sales.reset_index(drop=True)
storerS14 = train[(train.Store==rS) & (train.Year==2014)].Sales.reset_index(drop=True)
storerS15 = train[(train.Store==rS) & (train.Year==2015)].Sales.reset_index(drop=True)

df_plot = pd.concat([storerS13, storerS14, storerS15], axis=1)
df_plot.columns = ['2013', '2014', '2015']
df_plot.index = pd.date_range('1/1/2015', periods=365, freq='D')
df_plot['pred'] = pd.DataFrame(meanSales, index = pd.date_range('8/1/2015', periods=48, freq='D'))
df_plot.plot(subplots=True,figsize=(18, 6), title='Sales at store {}'.format(rS))
plt.show()
fig, ax = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
train2013 = train['2013']
train2013.groupby(train2013.index.day)['Sales'].mean().plot(label='2013', ax=ax[0],
    title='Monthly pattern of sales in 2013')
train2014 = train['2014']
train2014.groupby(train2014.index.day)['Sales'].mean().plot(label='2014', ax=ax[1],
    title='Monthly pattern of sales in 2014')
train2015 = train['2015']
train2015.groupby(train2015.index.day)['Sales'].mean().plot(label='2015', ax=ax[2],
     title='Monthly pattern of sales in 2014')
plt.legend(loc='upper center')
plt.title('Monthly pattern of sales in 2015');
train2013 = train['2013']
train2013.groupby(train2013.index.dayofyear%14)['Sales'].mean().plot(label='2013')
train2014 = train['2014']
train2014.groupby(train2014.index.dayofyear%14)['Sales'].mean().plot(label='2014')
train2015 = train['2015']
train2015.groupby(train2015.index.dayofyear%14)['Sales'].mean().plot(label='2015')
plt.legend(loc='lower left');
plt.title('14 days pattern of sales in 2013/14/15');
train2013 = train['2013']
train2013.groupby((train2013.index.dayofyear+12)%14)['Sales'].mean().plot(label='2013')
train2014 = train['2014']
train2014.groupby((train2014.index.dayofyear+13)%14)['Sales'].mean().plot(label='2014')
train2015 = train['2015']
train2015.groupby(train2015.index.dayofyear%14)['Sales'].mean().plot(label='2015')
plt.legend(loc='lower left');
plt.title('14 days pattern of sales in 2013/14/15');
rS = 660 # random.choice(train.Store.unique())
periodym1 = train.ix['2014-08-09':'2014-09-25']
periodym2 = train.ix['2013-08-03':'2013-09-19']
prevy1 = periodym1[periodym1.Store == rS]['Sales'].reset_index(drop=True)
prevy2 = periodym2[periodym2.Store == rS]['Sales'].reset_index(drop=True)
meanSales = np.mean(np.vstack((prevy1, prevy2)), axis=0)

storerS13 = train[(train.Store==rS) & (train.Year==2013)].Sales.reset_index(drop=True)
storerS14 = train[(train.Store==rS) & (train.Year==2014)].Sales.reset_index(drop=True)
storerS15 = train[(train.Store==rS) & (train.Year==2015)].Sales.reset_index(drop=True)

df_plot = pd.concat([storerS13, storerS14, storerS15], axis=1)
df_plot.columns = ['2013', '2014', '2015']
df_plot.index = pd.date_range('1/1/2015', periods=365, freq='D')
df_plot['pred'] = pd.DataFrame(meanSales, index = pd.date_range('8/1/2015', periods=48, freq='D'))
df_plot.plot(subplots=True,figsize=(18, 6), title='Sales at store {}'.format(rS))
plt.show()
periodym1 = train.ix['2014-08-09':'2014-09-25']
periodym1[periodym1.Store == 660]
train[train.Store==600].index.min()
train[train.Store==600].index.max()
from IPython.display import Image
Image('http://i.imgur.com/GrERfoZ.png')
periodym1 = train.ix['2014-08-02':'2014-09-18']
periodym2 = train.ix['2013-08-03':'2013-09-19']
prevy1 = periodym1[periodym1.Store == rS]['Sales'].reset_index(drop=True)
prevy2 = periodym2[periodym2.Store == rS]['Sales'].reset_index(drop=True)
meanSales = np.mean(np.vstack((prevy1, prevy2)), axis=0)
df_plot = pd.DataFrame(meanSales, index = pd.date_range('8/1/2015', periods=48, freq='D'))
df_plot.columns = ['Prediction']
df_plot.plot(title='Prediction for store {}'.format(rS));
