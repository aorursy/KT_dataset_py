# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data =pd.read_csv("../input/crypto-markets.csv")
#Exploration of Initial Data
print(data.head())
##Explore the columns and the availalbe Data ##
data.columns
df=data[['symbol','date','close']]
df.head()

df['date'] = pd.to_datetime(df.date)
#dg=df.groupby(['symbol',pd.Grouper(key='date',freq='1BM')],sort=False).mean()
#dg=df.groupby(['symbol',pd.Grouper(key='date',freq='A')],sort=False)
dg=df.groupby(['symbol',pd.Grouper(key='date',freq='A')],sort=False).agg(['min','max'])
dg.head()
#new_df= pd.pivot_table(df,index=['date'], columns = 'symbol', values = "close",
#                       aggfunc=len, fill_value=0)

#new_df= pd.pivot_table(df,index=['date','symbol'], values = "close",
#                       aggfunc=np.mean, fill_value=0)

#new_df= pd.pivot_table(df,index=['symbol'], columns=["date"],
#                       values = ["close",],aggfunc=[np.sum])
df['date'] = pd.to_datetime(df.date)
df['date']=df['date'].dt.strftime('%Y')

#df['date']=df['date'].dt.strftime('%Y')
#new_df1= pd.pivot_table(df,index=['symbol','date'],values='close',
#                       aggfunc=[np.mean],fill_value=0)
new_df= pd.pivot_table(df,index=['symbol','date'],values='close',
                       fill_value=0)
print(new_df.head())

#df2=new_df.apply(np.log).pct_change()
df2=new_df.apply(np.log).diff()
#df2.dropna()
df3=df2.sort_index(level=1,ascending=True)
df3.dropna()
print(df.loc[df['symbol']== 'BAT'])
print(new_df.query('symbol == ["BAT"]'))
#print(df2.query('symbol == ["BTC"]'))
#print(df3.query('symbol == ["BTC"]'))
df_Y17= df3.query('date == ["2017"]')
df_Y16= df3.query('date == ["2016"]')
df_Y15= df3.query('date == ["2015"]')
print(df_Y17.nlargest(10, 'close'))
print(df_Y16.nlargest(10, 'close'))
print(df_Y15.nlargest(10, 'close'))

df_test=new_df.apply(np.log).diff()
print(new_df.query('symbol == ["ETH"]'))
print(df_test.query('symbol == ["ETH"]'))
dfplt=new_df.query('symbol == ["BTC"]')
dfplt.plot()
plt.show()