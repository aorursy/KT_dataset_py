import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_fun=pd.read_csv("../input/fundamentals.csv")

df_fun.head()
df_price=pd.read_csv("../input/prices.csv")

df_price.head()
df_price.symbol.unique()
df_sec=pd.read_csv("../input/securities.csv")

df_sec.head()
df_psa=pd.read_csv("../input/prices-split-adjusted.csv",parse_dates=["date"])

df_psa.head()
type(df_psa.date[0])  #date time index check
df_psa.set_index("date",inplace=True)
df_psa_grp=df_psa.groupby(df_psa.symbol)
df_wltw=df_psa_grp.get_group("WLTW") 

df_acn=df_psa_grp.get_group("ACN") #accenture

df_abt=df_psa_grp.get_group("ABT") #abbot lab
df_wltw.head()
df_wltw.close.resample("M").mean()# monthly resampled data for closing points

df_acn.close.resample("M").mean()

df_abt.close.resample("M").mean()
#monthly closing points of 3 different securities.

df_wltw.close.resample("M").mean().plot(label="wltw")

df_acn.close.resample("M").mean().plot(label="accenture")

df_abt.close.resample("M").mean().plot(label="abbot lab")

plt.legend()
df_acn_2010=df_acn["2010-01-01":"2010-12-31"]#accenture data of year 2010

df_acn_2010.tail()
#accenture quartely results shown in plot

df_acn_2010.close.resample("Q").mean().plot(kind="bar",label="accenture",color="green")

df_acn.head()

df1=df_acn[["open","low","high","volume"]]



df1.corr()
X=df1

y=df_acn.close
from sklearn.feature_selection import SelectKBest

sk=SelectKBest(k=4)

X=sk.fit_transform(X,y)
sk.pvalues_
sk.scores_
#X=df_acn["high"].values.reshape(-1,1) #since we are using single variable

X=df1[["low"]]
#label

y=df_acn["close"]
#train test and split data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)
#lets try to scale the features

from sklearn.preprocessing import MinMaxScaler

mm=MinMaxScaler()

X_train_mm=mm.fit_transform(X_train)
from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor()

#training

#my_fit=gbr.fit(X_train_mm,y_train)

#my_fit



my_fit=gbr.fit(X_train,y_train)

my_fit
y_pred=gbr.predict(X_test)

y_pred
gbr.score(X_test,y_test)