# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
list_of_currency_files = check_output(["ls", "../input"]).decode("utf8").split('\n')

input_folder = '../input/'

list_of_currency_files.remove('')
for file in list_of_currency_files:

    if(file != ''):

        print(file[:-4]+" = pd.read_csv('"+input_folder+file+"')")

        try:

            exec(file[:-4]+" = pd.read_csv('"+input_folder+file+"',parse_dates=['Date'])")

        except Exception as exp:

            print(exp)

            exec(file[:-4]+" = pd.read_csv('"+input_folder+file+"')")
import matplotlib.dates as mdates

import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()



bitcoin_price['Date_mpl'] = bitcoin_price['Date'].apply(lambda x: mdates.date2num(x))



print(bitcoin_price.head())

fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(bitcoin_price.Close.values, time=bitcoin_price.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Price in USD', fontsize=12)

plt.title("Closing price distribution of bitcoin", fontsize=15)

plt.show()
percent_change = []

change = []

Sevendays_change = []

price_7days_before = bitcoin_price['Open'][0]

for ind,row in bitcoin_price.iterrows():

    if ind > 7:

        price_7days_before = bitcoin_price['Open'][ind-7]

    change.append(row['Close'] - row['Open'])

    percent_change.append((row['Close'] - row['Open'])/row['Open'])

    Sevendays_change.append((row['Close'] - price_7days_before)/price_7days_before)

bitcoin_price['Change'] = change

bitcoin_price['percent_change'] = percent_change

bitcoin_price['Sevendays_change'] = Sevendays_change

bitcoin_price.head()
#change graph



fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(bitcoin_price.percent_change.values, time=bitcoin_price.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Percent change', fontsize=12)

plt.title("Change distribution of bitcoin", fontsize=15)

plt.show()
#7days change graph



fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(bitcoin_price.Sevendays_change.values, time=bitcoin_price.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Percent change', fontsize=12)

plt.title("7 days Change distribution of bitcoin", fontsize=15)

plt.show()
bitcoin_price_temp = bitcoin_price[bitcoin_price['Date']>'2017-01-01']



fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(bitcoin_price_temp.Sevendays_change.values, time=bitcoin_price_temp.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Percent change', fontsize=12)

plt.title("7 days Change distribution of bitcoin", fontsize=15)

plt.show()



bitcoin_price_temp = bitcoin_price[bitcoin_price['Date']<'2017-01-01'] 

bitcoin_price_temp = bitcoin_price_temp[bitcoin_price_temp['Date']>'2016-01-01']

fig, ax = plt.subplots(figsize=(12,8))

sns.tsplot(bitcoin_price_temp.Sevendays_change.values, time=bitcoin_price_temp.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.xlabel('Date', fontsize=12)

plt.ylabel('Percent change', fontsize=12)

plt.title("7 days Change distribution of bitcoin", fontsize=15)

plt.show()

list_of_currency = []



for val in list(vars()):

    if val.find('price') > 0:

        exec('size ='+val+'.shape[0]')

        if size >= 789:

            list_of_currency.append(val)



print(list_of_currency)

packed_df = bitcoin_price[["Date","Close"]]

packed_df.columns = ["Date","Bitcoin"]

list_of_currency.remove('bitcoin_price')



for currency in list_of_currency:

    exec('currency_df='+currency+'[["Date","Close"]]')

    currency_df.columns = ["Date", currency]

    packed_df = pd.merge(packed_df,currency_df, on="Date")



print(packed_df.shape)



list_of_currency.append('Bitcoin')

temp_df = packed_df[list_of_currency]

corrmat = temp_df.corr(method='spearman')

fig, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Cryptocurrency correlation map", fontsize=15)

plt.show()
print(bitcoin_dataset.describe(),bitcoin_dataset.dtypes)

bitcoin_dataset.head()
#A glimps at out final data

bitcoin = bitcoin_dataset.copy()



#correlation of various variables with bitcoin closing price

x_cols = [col for col in bitcoin.columns if col not in ['Date', 'btc_market_price'] if bitcoin[col].dtype=='float64']



labels = []

values = []

for col in x_cols:

    labels.append(col)

    values.append(np.corrcoef(bitcoin[col].values, bitcoin.btc_market_price.values)[0,1])

corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')



ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,40))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

#autolabel(rects)

plt.show()
from sklearn.neural_network import MLPClassifier

from datetime import datetime, timedelta



bitcoin = bitcoin_dataset.copy()



print(bitcoin.isnull().values.sum())



columns_to_remove = ['btc_market_price','Date']



years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]





#spliting the test and train data (1620)

split = 1400

train_data = bitcoin[ : split].dropna()

test_data = bitcoin.ix[split:1620].dropna()

print(test_data.shape,train_data.shape)

mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='relu', solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)

selected_columns = [elem for elem in bitcoin.columns if elem not in columns_to_remove]

x_train = train_data[selected_columns]

y_train = train_data['btc_market_price']

x_test = test_data[selected_columns]

y_test = test_data['btc_market_price']





mlpc.fit(x_train.values, y_train.values)

prediction = mlpc.predict(x_test)
