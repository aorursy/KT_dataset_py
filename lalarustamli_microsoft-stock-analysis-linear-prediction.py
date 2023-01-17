# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import math

import numpy as np

from sklearn import preprocessing, svm, model_selection

from sklearn.linear_model import LinearRegression

import datetime

import matplotlib.pyplot as plt

from matplotlib import style
import pandas as pd

df = pd.read_csv("../input/EOD-MSFT.csv")
df=df.iloc[::-1]

df=df[['Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume']]

df['HL_PCT'] = (df['Adj_High']-df['Adj_Low']) / df['Adj_Low'] * 100.0

df['PCT_change'] = (df['Adj_Close']-df['Adj_Open']) / df['Adj_Open'] * 100.0
df = df[['Adj_Close','HL_PCT','PCT_change','Adj_Volume']]
forecast_col = 'Adj_Close'

df.fillna(-9999,inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['Label']=df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['Label'],1))

X = preprocessing.scale(X)

X=X[:-forecast_out]

X_lately=X[-forecast_out:]
X_lately
df.dropna(inplace=True)

y = np.array(df['Label'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
clf = LinearRegression()

clf.fit(X_train, y_train)

accuracy=clf.score(X_test,y_test)
accuracy
forecast_set=clf.predict(X_lately)

print(forecast_set,accuracy,forecast_out)
style.use('ggplot')

df['Forecast']=np.nan

last_date = df.iloc[-1].name 

dti = pd.date_range(last_date, periods=forecast_out+1, freq='D')

index = 1

for i in forecast_set:

    df.loc[dti[index]] = [np.nan for _ in range(len(df.columns)-1)] + [i]

    index +=1
df.Adj_Close.plot(figsize=(25,10))

df.Forecast.plot(figsize=(25,10))

plt.legend(loc=4)

plt.xlabel('Date')

plt.ylabel('Price')

plt.show()