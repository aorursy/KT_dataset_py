# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import datetime

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.

os.chdir('../input/spanish-stocks-historical-data')

print(os.listdir())
os.listdir()
filenames = [x for x in os.listdir()]

print(filenames)
li = []

for filename in filenames:

    stock_name = filename.replace('.csv', '')

    df = pd.read_csv(filename, index_col=None, header=0)

    df['Name'] = stock_name

    df['Date'] = pd.to_datetime(df['Date'])

    df = df[df['Date'] >= '2016-01-01']

    df['Return'] = df['Close'].pct_change()

    df.dropna(inplace = True)

    li.append(df)



stock_df = pd.concat(li, axis=0, ignore_index=True)

stock_df.head()
stock_df[['Name','Close']].groupby(['Name']).count()
stock_df.isnull().sum()
stock_df.head()
def get_stock_data(stock_name):

    return stock_df[stock_df['Name'] == stock_name]
stock_df.Name.unique()
acs = get_stock_data('acs')

acs.head()
stock_pivot = stock_df.pivot('Date','Name','Close').reset_index()

stock_pivot.head()
plt.figure(figsize = (15,10))

sns.heatmap(stock_pivot.corr())
stock_df.Name.unique()
stock_df.head()
energy_stocks = ['naturgy-energy','repsol', 'red-elctrica','iberdrola','acciona','siemens-gamesa','abengoa']

media_stocks = ['telefnica','mediaset', 'atresmedia','indra']

construction_stocks = ['colonial', 'fcc','ferrovial','acs','sacyr']

banking_insurance_stocks = ['bankinter', 'caixabank','mapre','banco-sabadell','mapfre','santander','bme','bbva']

production_stocks = ['acerinox','enags','inditex','grifols']



stock_df['Group'] = 0
stock_df.loc[stock_df['Name'].isin(energy_stocks), 'Group'] = 'Energy'

stock_df.loc[stock_df['Name'].isin(media_stocks), 'Group'] = 'Media'

stock_df.loc[stock_df['Name'].isin(construction_stocks), 'Group'] = 'Construction'

stock_df.loc[stock_df['Name'].isin(banking_insurance_stocks), 'Group'] = 'Banking'

stock_df.loc[stock_df['Name'].isin(production_stocks), 'Group'] = 'Production'
stock_df.Group.unique()
stock_df.head()
energy_df = stock_df[stock_df['Group'] == 'Energy'][['Date','Close','Return', 'Volume']].groupby('Date').mean()

production_df = stock_df[stock_df['Group'] == 'Production'][['Date','Close','Return','Volume']].groupby('Date').mean()

construction_df = stock_df[stock_df['Group'] == 'Construction'][['Date','Close','Return','Volume']].groupby('Date').mean()

media_df = stock_df[stock_df['Group'] == 'Media'][['Date','Close','Return','Volume']].groupby('Date').mean()

banking_df = stock_df[stock_df['Group'] == 'Banking'][['Date','Close','Return','Volume']].groupby('Date').mean()



plt.figure(figsize = (15,6))

top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)

bottom = plt.subplot2grid((4,4), (3,0), rowspan=3, colspan=4)



top.plot(energy_df.index,energy_df.Close)

top.plot(production_df.index,production_df.Close)

top.plot(construction_df.index,construction_df.Close)

top.plot(media_df.index,media_df.Close)

top.plot(banking_df.index,banking_df.Close)

top.legend(['Energy', 'Production', 'Construction', 'Media', 'Banking'])



bottom.plot(energy_df.index,energy_df.Volume)

bottom.plot(production_df.index,production_df.Volume)

bottom.plot(construction_df.index,construction_df.Volume)

bottom.plot(media_df.index,media_df.Volume)

bottom.plot(banking_df.index,banking_df.Volume)
plt.figure(figsize = (15,12))

plt.subplot(2,3,1)

ax1 = sns.distplot(energy_df['Return'])

ax1.set_title('Energy ')

plt.subplot(2,3,2)

ax2 = sns.distplot(production_df['Return'])

ax2.set_title('Production')

plt.subplot(2,3,3)

ax3 = sns.distplot(construction_df['Return'])

ax3.set_title('Construction')

plt.subplot(2,3,4)

ax4 = sns.distplot(media_df['Return'])

ax4.set_title('Media')

plt.subplot(2,3,5)

ax5 = sns.distplot(banking_df['Return'])

ax5.set_title('Banking')
testing = pd.concat([energy_df.Return,banking_df.Return, production_df.Return, media_df.Return, construction_df.Return],axis = 1)

testing.columns = ['Energy', 'Banking','Production','Media','Construction']



plt.figure(figsize = (15,10))

sns.heatmap(testing.corr())
def plot_ma(stock_name):

    df = stock_df[stock_df['Name'] == stock_name]

    df['Short_MA'] = df['Close'].rolling(50).mean()

    df['Long_MA'] = df['Close'].rolling(200).mean()

    plt.figure(figsize = (15,6))

    plt.xticks(rotation=45)

    plt.plot(df.Date, df.Close)

    plt.plot(df.Date, df.Short_MA)

    plt.plot(df.Date, df.Long_MA)

    plt.title(stock_name)

    plt.legend()
plot_ma('bbva')