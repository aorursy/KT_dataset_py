# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
startup_funding = pd.read_csv("../input/startup_funding.csv")
startup_funding.head()
startup_funding.info()
startup_funding.isnull().sum()
startup_funding.drop('Remarks',axis=1,inplace=True)
startup_funding.AmountInUSD = startup_funding.AmountInUSD.apply(lambda x : float(str(x).replace(',','')))
startup_funding.AmountInUSD = startup_funding.AmountInUSD.apply(lambda x : float(x))
#startup_funding.AmountInUSD.astype(np.float)
startup_funding.info()
print("Maximum funding:{}".format(startup_funding.AmountInUSD.max()))
startup_funding[startup_funding.AmountInUSD == 1400000000]
print("Minimun funding:{}".format(startup_funding.AmountInUSD.min()))
startup_funding[startup_funding.AmountInUSD == 16000]
print("Average funding:{}".format(startup_funding.AmountInUSD.sort_values(ascending=False).dropna().mean()))
#Maximum amount of funding received startups
max_fundings = startup_funding.groupby("StartupName").mean()['AmountInUSD']
max_fundings = max_fundings.sort_values(ascending=False)[:10]
plt.figure(figsize=(12,6))
max_fundings.plot(kind='bar')
#Maximum number funding received startup
print("TOTAL STATUP FUNDING:{}".format(len(startup_funding.StartupName.unique())))
print("TOP FUNDING RECEIVER:\n{}".format(startup_funding.StartupName.value_counts()[:10]))
start_name = startup_funding.StartupName.value_counts()[:10]
plt.figure(figsize=(14,7))
fig = sns.barplot(x=start_name.index,y=start_name.values,alpha=0.6)
fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
plt.show(fig)
industry_name = startup_funding.IndustryVertical.value_counts()[:10]
print("TOTAL NUMBER OF INDUSTRY:{}".format(len(startup_funding.IndustryVertical.unique())))
print("TOP INDUSTRY:\n{}".format(industry_name))
plt.figure(figsize=(15,7))
fig = sns.barplot(x=industry_name.index,y=industry_name,palette='hls')
#fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
fig.set_xlabel("INDUSTRY NAME")
fig.set_ylabel("TOTAL NUMBER OF FUNDING")
plt.show(fig)
industy_amount = startup_funding.groupby('IndustryVertical').mean()['AmountInUSD']
industy_amount = industy_amount.sort_values(ascending=False)[:10]
plt.figure(figsize=(12,7))
fig = industy_amount.plot(kind='bar')
fig.set_ylabel("TOTAL AMOUNT OF FUNDING(USD)")
fig.set_xlabel("INDUSTRY")
plt.show(fig)
startup_funding.CityLocation.unique()
plt.figure(figsize=(16,8))
startup_funding.CityLocation.value_counts()[:10].plot(kind='bar')
plt.xticks(rotation=80)
plt.show()

