# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv")
df['Date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date



df= df.groupby(df['Date']).mean()



print("length of data set:", len(df))

df.head()
df.rename(columns={'Weighted_Price':'cur_price'}, inplace = True)



df[['next_price']] = df[['cur_price']].shift(-1)



df.loc[(df['cur_price']<df['next_price']), 'price_trend'] = 1

df.loc[(df['cur_price']>df['next_price']), 'price_trend'] = 0



df.drop(df.index[-1], axis=0, inplace=True) 



df[['price_trend']] = df[['price_trend']].astype(int)



print("length:", len(df))

df.head()
data_heatmap = df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)','Volume_(Currency)','next_price']]





plt.figure(figsize=(10,10))

sns.heatmap(data_heatmap.corr(), vmin=0.5,annot=True, fmt=".3")

plt.show()
data_base_model = df[['Open','High', 'Low', 'Close', 'price_trend']]



data_base_model.head()


y = data_base_model['price_trend']

x = data_base_model.iloc[:,:4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)



len(x_train),len(x_test), len(y_train), len(y_test)
clf = LogisticRegression(solver='lbfgs')

clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)

score = clf.score(x_test, y_test)

print("Accurancy of model:",score)
matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(matrix, annot=True, fmt='d')

plt.show()
df['Weight'] = df['cur_price']

lasts = ['Open', 'High', 'Low', 'Close','Weight']



df[['last_Open', 'last_High', 'last_Low', 'last_Close', 'last_Weight']] = df[lasts].shift(1)



for price in OHLCW:

    df.loc[(df['last_'+price]<df[price]), price+'_trend'] = 1

    df.loc[(df['last_'+price]>df[price]), price+'_trend'] = 0



df.drop(df.index[0], axis=0, inplace=True)





new_order = ['Volume_(BTC)','Volume_(Currency)']

for price in lasts:

    new_order.extend([price, 'last_'+price, price+'_trend'])

new_order.extend(['cur_price', 'next_price', 'price_trend'])

df = df[new_order]



df[['Open_trend']] = df[['Open_trend']].astype(int)

df[['High_trend']] = df[['High_trend']].astype(int)

df[['Low_trend']] = df[['Low_trend']].astype(int) 

df[['Close_trend']] = df[['Close_trend']].astype(int) 

df[['Weight_trend']] = df[['Weight_trend']].astype(int) 





data_label = df[['Open_trend', 'High_trend', 'Low_trend', 'Close_trend','Weight_trend', 'price_trend']]



data_label
y = data_label['price_trend']

x = data_label.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

clf = LogisticRegression(solver='lbfgs')

clf.fit(x_train, y_train)



y_pred=clf.predict(x_test)

score = clf.score(x_test, y_test)

score
y = data_base_model['price_trend']

x = data_base_model.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)





clf = LogisticRegression(solver='liblinear')

clf.fit(x_train, y_train)



y_pred=clf.predict(x_test)

score = clf.score(x_test, y_test)

score
y = data_base_model['price_trend']

x = data_base_model.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)



clf = LogisticRegression(solver='liblinear', tol=4e-6)

clf.fit(x_train, y_train)



y_pred=clf.predict(x_test)

score = clf.score(x_test, y_test)

score
Bitcoin = data_base_model[['Open', 'Close', 'High', 'Low', 'price_trend']]

Vibration=Bitcoin.High-Bitcoin.Low

Bitcoin['vibration'] = Vibration

Bitcoin.head(5)



predictors = ['Open','Close','High', 'Low']



X_train, X_test, y_train, y_test = model_selection.train_test_split(Bitcoin[predictors], Bitcoin.price_trend,test_size = 0.25, random_state = 1234)