# import required modules

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read data and fetch Time Series 'Charles Dickens' 

train = pd.read_csv('../input/train_1.csv').fillna(0)

df = None

for i, row in train.iterrows():

    if row['Page'] == "Charles_Dickens_en.wikipedia.org_desktop_all-agents":

        df = row

        break

df.drop(['Page'], inplace=True)

df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df.head()
df.tail()
import matplotlib.dates as mdates

fig = plt.figure()

ax = fig.add_subplot(1,1,1)  

plt.plot(df)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=90))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y:%m'))

plt.xlabel('Date')

plt.ylabel('Number of Hits')

plt.title('Wikipedia Hits for Charles Dickens')
df.sort_values()[-5:]
df_train = df['2016-01-01':'2016-06-30'].copy()

df_test = df['2016-07-01':'2016-12-31'].copy()
from statsmodels.tsa.arima_model import ARIMA

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df_train)
# series seems to be already stationary so no need for difference 

arima = ARIMA(np.array(df_train), [5, 0, 0])

result = arima.fit(disp=False)
# use the model to predict second half of 2016

ans = result.predict(start=183, end=365).astype(int)
# better way?

for i in range(len(ans)):

    df_test.iloc[i] = ans[i]
fig = plt.figure()

ax = fig.add_subplot(1,1,1)  

ax.xaxis.set_major_locator(mdates.DayLocator(interval=90))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y:%m'))

plt.plot(df)

plt.plot(df_test)