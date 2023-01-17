# Load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math



#%matplotlib inline

#import warnings

#warnings.filterwarnings("ignore")

# Load Data

df = pd.read_csv('../input/Data_To_Hourervals_no_filter.csv', header=0, sep = ';')

df['hour'] = df['Date'].str.split(' ', n = 1, expand = True)[1]



df = df.set_index('Date')
#Adding new columns

#Price movement during the period

df['Change'] = df['Close']-df['Open']



#Lets create a compound score that ignores the neutral values

df['Compound_Score_with_filter'] = (df['Sent_Negatives']*df['Count_Negatives']+df['Sent_Positives']*df['Count_Positives'])/(df['Count_Positives']+df['Count_Negatives'])



#Percentual price change during the period

df['log_change'] = np.log(df['Close']) - np.log(df['Close'].shift(1))



#Percentual volume of tweets change during the period

df['log_n'] = np.log(df['n']) - np.log(df['n'].shift(1))



#Drop columns that are not interesting

df = df.drop(columns = ['Low','High', 'Volume (Currency)', 'Open', 'Sent_Negatives', 'Sent_Positives', 'Count_Negatives', 'Count_Positives', 'Count_Neutrals'], axis=0)





#change Volume (BTC) column to float



#df['Volume (BTC)'] = pd.to_numeric(df['Volume (BTC)'])
df.head()
df.tail()
df.info()
values = df.values

groups = [0,1,2,3,4,5,6,7,8]

i =1  

plt.figure(figsize=(16,13))

for group in groups:

    plt.subplot(len(groups),1,i)

    plt.plot(values[:,group])

    plt.title(df.columns[group], y=.5, loc='right')

    i += 1

plt.show()
#As we can see in the graphs volume (btc) is wrong. So I will delete it

df = df.drop(columns = ['Volume (BTC)'], axis=0)

#Lets see the distrinution of our key variables

plt.figure(figsize=(10,7))

plt.hist(df['Compound_Score'], bins=30, range = (-1,1))

plt.title('Compound Score Histogram')

plt.show()



plt.figure(figsize=(10,7))

plt.hist(df['n'], bins=30, range = (0,5000))

plt.title('Tweets Volume Histogram')

plt.show()





plt.figure(figsize=(10,7))

plt.hist(df['log_change'], bins=30, range = (-0.1,0.1))

plt.title('Bitcoin Price Change Histogram')

plt.show()
df.describe()
#Correlation of all values

cor = df.corr()

cor
sns.set(style="white")

f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)

ax =sns.heatmap(cor, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .7})

plt.show()
plt.figure(figsize=(16,13))

plt.plot(df.index, df['Close'], 'red')

plt.plot(df.index, df['n'], 'g')

plt.title('BTC Close Price(hr) vs volume tweets')

plt.xticks(rotation='vertical')

plt.ylabel('Price ($)');

plt.show();

df.index = pd.to_datetime(df.index)

df['hour']=df.index.to_series().apply(lambda x: x.strftime("%X"))

hour_df=df.groupby('hour').agg(lambda x: x.mean())

hour_df['hour'] = hour_df.index

hour_df
#sns Hourly Heatmap

fig, ax = plt.subplots(figsize=(16, 9))

result = hour_df.pivot(index='hour', columns='n', values='Change')

sns.heatmap(result,ax = ax, annot=True, fmt="g", cmap='viridis')

plt.title('Volume of tweets x BTC change avg(Hr)')

plt.show()
df['Day'] = df.index.day_name()

day_df=df.groupby('Day').agg(lambda x: x.mean())

day_df['Day'] = day_df.index

day_df = day_df.reindex(index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday', 'Sunday'])

day_df
fig, ax = plt.subplots(figsize=(16, 9))

result = day_df.pivot(index='Day', columns='n', values='Change')

sns.heatmap(result,ax = ax, annot=True, fmt="g", cmap='viridis')

plt.title('Volume of tweets x BTC change avg(Day of the week))')

plt.show()
#Correlation. Basically I want to see how the volume of today affects the value of the future. 

#Apparently it affects positively until 100 days later

Autocorr = []



for a in range(12362):

    corr_df = df[['Close']]

    corr_df[['n']] = df[['n']].shift(a)

    cor = corr_df.corr()['Close'][1]

    if str(cor) == 'nan':

        break

    Autocorr.append(cor)



plt.figure(figsize=(16,13))

plt.plot(range(len(Autocorr)), Autocorr, 'g')

plt.title('Autocorrelation of volume and price')

plt.xticks(rotation='vertical')

plt.hlines(0.05, 0, 13000, linestyles = 'dashed', colors = 'r')

plt.hlines(0, 0, 13000, linestyles = 'solid', colors = 'blue')

plt.hlines(-0.05, 0, 13000, linestyles = 'dashed', colors = 'r')

plt.ylabel('Tweets');

plt.show();


print('The maximum correlation is at', max(Autocorr), 'and that happens after ', Autocorr.index(max(Autocorr)), 'hours')

#Correlation of values that apparently are not correlated. 

#It is important to test this because maybe the volume of tweets of today does not affect the price of today, but it does tomorrow. 



Autocorr = []



for a in range(12362):

    corr_df = df[['Close']]

    corr_df[['Compound_Score']] = df[['Compound_Score']].shift(a)

    cor = corr_df.corr()['Close'][1]

    if str(cor) == 'nan':

        break

    Autocorr.append(cor)



plt.figure(figsize=(16,13))

plt.plot(range(len(Autocorr)), Autocorr, 'g')

plt.title('Autocorrelation of volume and price')

plt.xticks(rotation='vertical')

plt.hlines(0.05, 0, 13000, linestyles = 'dashed', colors = 'r')

plt.hlines(0, 0, 13000, linestyles = 'solid', colors = 'blue')

plt.hlines(-0.05, 0, 13000, linestyles = 'dashed', colors = 'r')

plt.ylabel('Tweets');

plt.show();


