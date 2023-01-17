# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from dateutil.parser import parse



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

CryptoMarkets = pd.read_csv('../input/crypto-markets.csv')
CryptoMarkets.dtypes
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

cm = pd.read_csv('../input/crypto-markets.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

cm.dtypes
dm = pd.read_csv('../input/crypto-markets.csv')  

dm['date'] = pd.to_datetime(dm['date'])
cmopen=cm.iloc[:50,0]

cmhigh=cm.iloc[:50,1]

cmlow=cm.iloc[:50,2]

cmclose=cm.iloc[:50,3]
plt.plot(cmopen, color='blue', label='open')

plt.plot(cmhigh, color='green', label='high')

plt.plot(cmlow, color='magenta', label='low')

plt.plot(cmclose, color='red', label='close')







plt.legend(loc='upper left')



plt.xticks(rotation=60)



plt.show()

# Select Time Window based on YYYY-mm-dd



view = cmopen['2017-08-14':'2017-07-14']

plt.subplot(2,1,1)

plt.xticks(rotation=45)

plt.title('Open 2017-07-14 to 2017-08-14')

plt.plot(view,color ='blue')





view = cmclose['2017-07-14':'2017-06-14']

plt.subplot(2,1,2)

plt.xticks(rotation=45)

plt.title('Close 2017-06-14 to 2017-07-14')

plt.plot(view,color ='violet')



plt.tight_layout()

plt.show()



# Select Time Window based on YYYY-mm



view = cmhigh['2017-07':'2017-06']

plt.xticks(rotation=0)

plt.title('High 2017-06-14 to 2017-07-14')

plt.plot(view,color ='green')







cm_open_ma=pd.rolling_mean(cmopen,2)

cm_close_ma=pd.rolling_mean(cmclose,2)
# view = cm_open_ma['2017-08-14':'2017-07-14']

plt.subplot(2,1,1)

plt.xticks(rotation=45)

plt.title('Moving Average Open')

plt.plot(cm_open_ma,color ='blue')





# view = cm_close_ma['2017-07-14':'2017-06-14']

plt.subplot(2,1,2)

plt.xticks(rotation=45)

plt.title('Moving Average Close')

plt.plot(cm_close_ma,color ='violet')



plt.tight_layout()

plt.show()
cm_high_ma=pd.rolling_mean(cmhigh,2,2,'D')

cm_low_ma=pd.rolling_mean(cmlow,2,2,'D')



# view = cm_open_ma['2017-08-14':'2017-07-14']

plt.subplot(2,1,1)

plt.xticks(rotation=45)

plt.title('Daily Moving Average High')

plt.plot(cm_high_ma,color ='yellow')





# view = cm_close_ma['2017-07-14':'2017-06-14']

plt.subplot(2,1,1)

plt.xticks(rotation=45)

plt.title('Daily Moving Average Low')

plt.plot(cm_low_ma,color ='black')



plt.tight_layout()

plt.show()
cm_high_ma=pd.rolling_std(cmhigh,2,2,'D')

cm_low_ma=pd.rolling_std(cmlow,2,2,'D')



# view = cm_open_ma['2017-08-14':'2017-07-14']

plt.subplot(2,1,1)

plt.xticks(rotation=45)

plt.title('Daily Moving Standard Deviation High')

plt.plot(cm_high_ma,color ='yellow')





# view = cm_close_ma['2017-07-14':'2017-06-14']

plt.subplot(2,1,1)

plt.xticks(rotation=45)

plt.title('Daily Standard Deviation Average Low')

plt.plot(cm_low_ma,color ='black')



plt.tight_layout()

plt.show()