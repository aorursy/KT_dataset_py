import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt 



import gc

import warnings



warnings.filterwarnings("ignore")
def make_features(df):



    # Date

    df['date'] = df['date'].str.replace(r'[年,月]', '/')

    df['date'] = df['date'].str.replace(r'[日]', '')



    df[['year','month','day']] = df['date'].str.split('/', expand=True).astype(int)





    # Volume

    df['volume'].fillna(-1,inplace=True)

    df['volume'] = df['volume'].astype(int)



    # Make target feature

    df['change'] = df['close'] - df['open']



    df['target'] = np.nan



    df.loc[df[df['change'] > 0].index,'target'] = 1

    df.loc[df[df['change'] <= 0].index,'target'] = 0





    # Shift features - p as previous

    df['p_volume'] = df['volume'].shift(-1)

    df['p_close'] = df['close'].shift(-1)

    df['p_p_close'] = df['close'].shift(-2)

    df['p_high'] = df['high'].shift(-1)

    df['p_low'] = df['low'].shift(-1)

    df['p_open'] = df['open'].shift(-1)



    df = df[['year','month','day','open','close','p_close','p_p_close','p_volume','p_high','p_low','p_open','target']]



    return df

def stat_features(df):

    

    # Make change in rate from previous open to current one

    df['change'] = (df['open'] - df['p_close']) / df['open']



    durations = [2,5,7]



    for days in durations:

        df['change_' + str(days) + '_days_sum'] = df['change'].rolling(days).sum()



    # print(df[['year','change','var_3_days']].head(10))



    return df
def volatility_ratio(df):

    

    # Current True Range = Maximum−Minimum

    # Maximum = Average of current day’s high and yesterday’s close

    df['maximum'] = (df['p_high'] + df['p_p_close']) / 2

    # Minimum= Average of today’s low and yesterday’s close

    df['minimum'] = (df['p_low'] + df['p_p_close']) / 2

    

    df['CTR'] = df['maximum'] - df['minimum']



    period = 10

    

    # HIGH = Average of the high prices of each day over time period x 

    df['high_sum_'+str(period)] = df['high'].rolling(period).sum() / period

    # LOW = Average of the low prices of each day over time period x

    df['low_sum_'+str(period)] = df['low'].rolling(period).sum() / period

    

    # PTR = Previous true range over x number of days

    df['PTR'] = df['high_sum_'+str(period)] - df['low_sum_'+str(period)]

    

    # Volatility Ratio

    df['vr'] = df['CTR'] / df['PTR']

    

    df.drop(['p_p_close','maximum','minimum','CTR','high_sum_'+str(period),'low_sum_'+str(period),'PTR'],axis=1,inplace=True)



    
def simple_moving_averages(df):

    

    periods = [7,30,120]

    

    for period in periods:

        df['SMA_'+str(period)] = df['p_close'].rolling(period).sum() / period

        

        # Trake difference and normalize it

        df['SMA_'+str(period)] = (df['open'] - df['SMA_'+str(period)]) / df['open']

        
def pivot(df):

    

    # PP = (close + High + Low) / 3

    df['PP'] = (df['p_close'] + df['p_high'] + df['p_low']) / 3

    

    # PP + (PP - L)

    df['R1'] = df['PP']*2 - df['p_low']

    # Take difference and normalize it

    df['R1_ratio'] = (df['R1'] - df['open']) / df['open']

    

    # PP + (PP - H)

    df['S1'] = df['PP']*2 - df['p_high']

    # Take difference and normalize it

    df['S1_ratio'] = (df['open']-df['S1']) / df['open']

    

    df.drop(['PP'],axis=1,inplace=True)

    
def rsi(df):

    

    # RSI = A/(A+B)

    # A = average price of positive difference in 14 days

    # B = average price of negative difference in 14 days

    df['diff'] = df['p_close'].diff()



    df['up'] = df['diff'].rolling(14).apply(lambda x: x[x>0].mean())

    df['down'] = df['diff'].rolling(14).apply(lambda x: x[x<0].mean())

    

    df['rsi'] = df['up'] / (df['up'] + df['down'])

    

    df.drop(['up','down'],axis=1,inplace=True)

    
train = pd.read_csv('../input/nikkei/nikkei.csv')



# Change newest date → oldest date to oldest data → newest date of rows

train = train[::-1]

train.reset_index(drop=True,inplace=True)



# print(train['date'])



# Preproecess features

make_features(train)



train = train[train['year'] > 2000]



stat_features(train)

simple_moving_averages(train)

volatility_ratio(train)

pivot(train)



# Get rid of null rows

train = train[14:]



print(train.head())

print(train.columns.values)
year = 2000



train['p_clo_change'] = (train['close']-train['p_close']) / train['p_close'] * 100



temp = train[(train['year'] > year) & (train['year'] != 2008)]



# print(train['ope_change'].value_counts())



s_mean = temp.groupby('month')['p_clo_change'].mean()

s_var = temp.groupby('month')['p_clo_change'].var()



plt.figure(figsize=(10,5))

plt.title('Change in price from ' + str(year) + ' to 2020/02 (%) by day')



sns.barplot(s_mean.index,s_mean)



plt.figure(figsize=(10,5))

sns.barplot(s_var.index,s_var)



print(s_mean)

print(s_var)



plt.show()



del temp; gc.collect()



# Change in price = (close - previous close price) (From 2001 to 2020/02 excluding 2008)



temp = train[(train['year'] > year) & (train['year'] != 2008)]



# print(train['ope_change'].value_counts())



s_mean = temp.groupby('day')['p_clo_change'].mean()

s_var = temp.groupby('day')['p_clo_change'].var()

s_count = temp.groupby('day')['p_clo_change'].count()



plt.figure(figsize=(10,5))

plt.title('Change in price from ' + str(year) + ' to 2020/02 (%) by day')



sns.barplot(s_mean.index,s_mean)



plt.figure(figsize=(10,5))

sns.barplot(s_var.index,s_var)



print(s_mean)

print(s_var)

print(s_count)



plt.show()



del temp; gc.collect()

# Change in price = (close - previous close price) (From 2001 to 2020/02 excluding 2008)



year = 2010



temp = train[(train['year'] > year) & (train['year'] != 2008) & (train['month'].isin([3,6,9,12]))]



# print(train['ope_change'].value_counts())



s_mean = temp.groupby('day')['p_clo_change'].mean()

s_var = temp.groupby('day')['p_clo_change'].var()

s_count = temp.groupby('day')['p_clo_change'].count()



plt.figure(figsize=(10,5))

plt.title('Change in price from ' + str(year) + ' to 2020/02 (%) by day')



sns.barplot(s_mean.index,s_mean)



plt.figure(figsize=(10,5))

sns.barplot(s_var.index,s_var)



print(s_mean)

print(s_var)

print(s_count)



plt.show()



del temp; gc.collect()



year = 2010



train['ope_change'] = (train['close']-train['open']) / train['open'] * 100



temp = train[(train['year'] > year) & (train['year'] != 2008)]





# print(train['ope_change'].value_counts())



s_mean = temp.groupby('month')['ope_change'].mean()

s_var = temp.groupby('month')['ope_change'].var()



plt.figure()

plt.title('Change in price from ' + str(year) + ' to 2020/02 (%)')



sns.barplot(s_mean.index,s_mean)



plt.figure()

sns.barplot(s_var.index,s_var)



plt.show()



print(s_mean)

print(s_var)



del temp; gc.collect()

# R1 = R1 - opening, positive: opening is below R1

# S1 = opening - S1, positive: opening is above S1



fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True)



ax[0].set_title('R1_ratio vs (closing-opening)')

ax[0].scatter(train['R1_ratio'],train['ope_change'])

ax[0].set_xlabel('R1_ratio')

ax[0].set_ylabel('Change in price')



ax[1].set_title('S1_ratio vs (closing-opening)')

ax[1].scatter(train['S1_ratio'],train['ope_change'])

ax[1].set_xlabel('S1_ratio')

ax[1].set_ylabel('Change in price')



plt.show()
fig,ax = plt.subplots(1,3,figsize=(15,5),sharex=True)



smas = ['SMA_7','SMA_30','SMA_120']



for i,sma in enumerate(smas):

    ax[i].set_title(str(sma) + ' vs (closing-opening)')

    ax[i].scatter(train[sma],train['ope_change'])

    ax[i].set_xlabel(str(sma))

    ax[i].set_ylabel('Change in price')



plt.show()
fig,ax = plt.subplots(1,1,figsize=(10,5),sharex=True)



ax.set_title('Volatility ratio vs (closing-opening)')

ax.scatter(train['vr'],train['ope_change'])

ax.set_xlabel('Volatility')

ax.set_ylabel('Change in price')
fig,ax = plt.subplots(1,2,figsize=(15,5),sharex=True)



train['p_high_ratio'] = (train['p_high'] - train['open']) / train['open'] 

train['p_low_ratio'] = (train['open']-train['p_low']) / train['open'] 



smas = ['p_high_ratio','p_low_ratio']



for i,sma in enumerate(smas):

    ax[i].set_title(str(sma) + ' vs (closing-opening)')

    ax[i].scatter(train[sma],train['ope_change'])

    ax[i].set_xlabel(str(sma))

    ax[i].set_ylabel('Change in price')



plt.show()
fig,ax = plt.subplots(1,2,figsize=(15,5),sharex=True)



train['p_close_ratio'] = (train['p_close'] - train['open']) / train['open'] 

train['p_open_ratio'] = (train['p_open'] - train['open']) / train['open'] 



smas = ['p_open_ratio','p_close_ratio']



for i,sma in enumerate(smas):

    ax[i].set_title(str(sma) + ' vs (closing-opening)')

    ax[i].scatter(train[sma],train['ope_change'])

    ax[i].set_xlabel(str(sma))

    ax[i].set_ylabel('Change in price')



plt.show()