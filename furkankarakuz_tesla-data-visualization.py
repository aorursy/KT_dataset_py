# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")
df.head()
df.tail()
df.describe().T
df.corr()
df.cov()
plt.bar(x="Open",height=df["Open"].mean())
plt.bar(x="Close",height=df["Close"].mean())
plt.show()
plt.bar(x="High",height=df["High"].mean())
plt.bar(x="Low",height=df["Low"].mean())
plt.show()
sns.pairplot(df);
sns.pairplot(df,kind="reg");
sns.regplot(x="Open",y="Close",data=df,color="r")
sns.regplot(x="High",y="Close",data=df,color="b")
sns.regplot(x="Open",y="Close",data=df,color="r")
sns.regplot(x="Low",y="Close",data=df,color="b")
sns.distplot(df["Open"])
sns.distplot(df["Close"])
sns.kdeplot(df["Close"],shade=True)
sns.kdeplot(df["Open"],shade=True)
sns.distplot(df["Low"])
sns.distplot(df["High"])
sns.kdeplot(df["Low"],shade=True)
sns.kdeplot(df["High"],shade=True)
sns.distplot(df["Open"])
sns.distplot(df["Close"])
df.index=df["Date"]
df.index=pd.DatetimeIndex(df.index)
sns.set_context("poster")
sns.set(rc={"figure.figsize":(16,9.)})
sns.set_style("whitegrid")
df["Open"].plot(style="-")
df["Close"].plot(style="--")
plt.legend(["Open","Close"],loc="upper left")
df["Open"].plot(style="-")
df["Close"].plot(style="--")
plt.xlim("2019-05-01","2020-01-01")
plt.legend(["Open","Close"],loc="upper left")
df["Low"].plot()
df["High"].plot()
plt.legend(["Low","High"],loc="upper left")
df["Low"].plot()
df["High"].plot()
plt.xlim("2019-05-01","2020-01-01")
plt.legend(["Low","High"],loc="upper left")
df_close=df["Close"]
df_close.plot()
plt.xlim("2019-05-01","2020-01-01")
df_close_7=df_close.rolling(7).mean()
df_close_10=df_close.rolling(10).mean()
df_close_20=df_close.rolling(20).mean()
df_close_30=df_close.rolling(30).mean()
df_close_7[:10]
df_close_10[:15]
df_close.plot(color="red")
df_close_7.plot(color="blue")
df_close_10.plot(color="green")
df_close_20.plot(color="yellow")
df_close_30.plot(color="black")
plt.xlim("2019-05-01","2020-01-01")
plt.legend(["Close","7_Day","10_Day","20_Day","30_Day"],loc="upper left")
df_close.plot(color="red")
df_close_7.plot(color="blue")
df_close_10.plot(color="green")
df_close_20.plot(color="yellow")
df_close_30.plot(color="black")
plt.xlim("2020-01-01","2020-02-01")
plt.legend(["Close","7_Day","10_Day","20_Day","30_Day"],loc="upper left")
