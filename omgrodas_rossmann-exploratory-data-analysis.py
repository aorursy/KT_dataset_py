from pathlib import Path

import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf

pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 500)
plotly.offline.init_notebook_mode()
cf.go_offline()

%matplotlib inline
%reload_ext autoreload
%autoreload 2
path=Path("../input/rossmann-data-engineering/")
testdf=pd.read_feather(path/"test.feather")
traindf=pd.read_feather(path/"train.feather")
traindf.set_index("Date",inplace=True,drop=False)
testdf.set_index("Date",inplace=True,drop=False)
!ls ../input/
data=traindf.append(testdf,sort=False)
data.groupby(data.index).size().iplot(kind="bar")
#Size of training data
traindf.shape
#Size of test data
testdf.shape
#Start and end date training data
traindf.index.max(),traindf.index.min()
#number of days with training data
traindf.index.max()-traindf.index.min()
#Number of stores
traindf.Store.nunique()
#Start and end date test data
testdf.Date.max(),testdf.Date.min()
#number of days with test data
testdf.Date.max()-testdf.Date.min()
#Number of stores in test dataset
testdf.Store.nunique()
#Distribution of records per store in training dataset
# 934 stores has 942 records
# 180 stores has 758 records
# 1 store has 1941 records
traindf.groupby("Store").size().value_counts()
#Histogram of above
traindf.groupby("Store").size().iplot(kind="histogram")
#Distribution of records per store in test dataset
# 48 stores with 856 recods
testdf.groupby("Store").size().value_counts()
#Number of stores in test set but not in training data
trains=pd.Series(traindf.Store.unique(),name="train")
tests=pd.Series(testdf.Store.unique(),name="test")
len(tests[~tests.isin(trains.values)])
#Find some of the stores with missing data and plot them
traindf.groupby("Store").size().sort_values()[:10]
#Plot one of the stores with missing data
#No spike when data returns. Seems to be just missing data and not a closed store that is opening. When a store opens after beeing closed there is normally a big spike in sales. 
traindf[traindf.Store==710]["Sales"].iplot(kind="bar",rangeslider=True)
traindf[traindf.Store==1]["Sales"].iplot(kind="bar",rangeslider=True)
traindf["Sales"].groupby("Date").agg(["mean"]).iplot(kind="bar",rangeslider=True)
traindf.groupby(["Year","Dayofyear"])["Sales"].agg("mean").to_frame().reset_index().pivot_table(values="Sales",index="Dayofyear",columns="Year").iplot(kind="bar",rangeslider=True)
traindf["Sales"].resample("W").agg(["mean"]).iplot(kind="bar")
traindf.groupby(["Year","Week"])["Sales"].agg("mean").to_frame().reset_index().pivot_table(values="Sales",index="Week",columns="Year").iplot(kind="bar",rangeslider=True)
traindf["Sales"].resample("M").agg(["mean"]).iplot(kind="bar")
#Almost no sales on sundays. Important feature
traindf.groupby("Dayofweek")["Sales"].agg(["mean"]).iplot(kind="bar")
#List the stores with the larges numbers of closed days
traindf[traindf.Open==False].groupby("Store").size().sort_values(ascending=False)[:20]
#Plotting one of the stores with lots of closed days. 
#When a closed stores open there is normally a spike in sales
#There is also sometimes a spike before a store closes
traindf[traindf.Store==103]["Sales"].iplot(kind="bar",rangeslider=True)
from pandas.plotting import autocorrelation_plot
data=traindf.groupby("Day")["Sales"].agg(["mean"])
autocorrelation_plot(data)

datecols=traindf.select_dtypes(include="datetime").columns.tolist()
traindf[datecols]=traindf[datecols].astype("int64")
corr = traindf.corr()
corr.iplot(kind="heatmap",colorscale='spectral')
gcorr=corr.abs()["Sales"].sort_values(ascending=False)
gcorr
gcorr.index.tolist()