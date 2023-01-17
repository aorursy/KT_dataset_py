import numpy as np

import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from scipy import stats

import statsmodels.api as sm

%matplotlib inline

sns.set()
maincv=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

maincv=maincv.iloc[:,1:8] #delete SNo

maincv.head()
maincv.info()
# Each country

maincv1=maincv.groupby(["Country/Region","ObservationDate"]).sum()

maincv1.head()
plt.figure()

maincv1["Confirmed"].unstack(level=0).plot(figsize=(20, 20))

plt.xticks(rotation=70)

plt.show()
plt.figure()

maincv1["Confirmed"].unstack(level=0).plot(figsize=(20, 20))

plt.xticks(rotation=70)

plt.yscale('log')

plt.show()

#only time-series

maincv2=maincv.groupby("ObservationDate").sum()

maincv2.head()
maincv2.max()
maincv2.plot()

plt.xticks(rotation=70)

plt.show()
maincv2.plot()

plt.xticks(rotation=70)

plt.yscale('log')

plt.show()
plt.figure(figsize=(30, 30))

plt.subplot(2,2,1)

plt.xticks(rotation=70)

plt.plot(maincv2["Confirmed"])

plt.subplot(2,2,2)

plt.xticks(rotation=70)

plt.plot(maincv2["Deaths"])

plt.subplot(2,2,3)

plt.xticks(rotation=70)

plt.plot(maincv2["Recovered"])

plt.show()
#log

plt.figure(figsize=(30, 30))

plt.subplot(2,2,1)

plt.yscale('log')

plt.xticks(rotation=70)

plt.plot(maincv2["Confirmed"])

plt.subplot(2,2,2)

plt.yscale('log')

plt.xticks(rotation=70)

plt.plot(maincv2["Deaths"])

plt.subplot(2,2,3)

plt.yscale('log')

plt.xticks(rotation=70)

plt.plot(maincv2["Recovered"])

plt.show()
#add total

maincv_diff=maincv.copy()

maincv_diff["total"]=maincv["Confirmed"]-maincv["Deaths"]-maincv["Recovered"]

maincv_diff
maincv_diff1=maincv_diff.groupby(["ObservationDate"]).sum()

maincv_diff1.head()
maincv_diff1.plot()

plt.xticks(rotation=70)

plt.show()
plt.figure()

maincv_diff1.plot(figsize=[10,10])

plt.yscale('log')

plt.xticks(rotation=70)

plt.show()
#

maincv_change=maincv_diff1.pct_change().dropna()

maincv_change.head()
maincv_change["total"].plot()

plt.xticks(rotation=70)

plt.show()
maincv3=maincv.groupby(["Province/State"]).sum()

maincv3.head()
maincv3.max()
maincv3.sort_values(by='Confirmed') 
maincv4=maincv.groupby(["Province/State","ObservationDate"]).sum()

maincv4.head()