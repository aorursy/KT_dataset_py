!pip install statsmodels --upgrade

!pip install prophet
# HYPER PARAMS

max_boosting_rounds = 5500



import time

notebookstart= time.time()



import datetime as datetime

import pandas as pd

import numpy as np

import time  # To time our operations

import fbprophet

from collections import defaultdict





# Viz

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import seaborn as sns

from wordcloud import WordCloud

import statsmodels.api as sm

import scipy.sparse as sparse





# Hide Warnings

Warning = True

if Warning is False:

    import warnings

    warnings.filterwarnings(action='ignore')

    warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    warnings.filterwarnings(action='ignore', category=FutureWarning)



# Modeling..

import lightgbm as lgb

import shap

shap.initjs()

from sklearn import metrics

from sklearn import preprocessing



# Tf-Idf

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from scipy.sparse import hstack, csr_matrix



np.random.seed(2018)



from contextlib import contextmanager

import re

import string

import gc



@contextmanager

def timer(name):

    """

    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin

    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s

    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s

    """

    t0 = time.time()

    yield

    print(f'[{name}] done in {time.time() - t0:.0f} s')
df = pd.read_csv("../input/pitchfork.csv", index_col= 'id')#.sample(50000)

df = df.dropna(subset=['review'])

df['genre'].fillna("No info", inplace=True)

#test = pd.read_csv("../input/test.csv", index_col= 'qid')#.sample(5000)

#testdex = test.index

df['date'] = pd.to_datetime(df['date'])

df['year_datetime'] = df['date'].dt.year

df['month_datetime'] = df['date'].dt.month

df['weekday_datetime'] = df['date'].dt.weekday



from pandas.tseries.offsets import MonthBegin

df['root_month'] = pd.to_datetime(df['date']) - MonthBegin(1)



df = df.sort_values(by='date')
print(df.shape)

df.head()
from matplotlib import pyplot, dates

%matplotlib inline



g = sns.FacetGrid(df[df['year_datetime']!=2019], col="year_datetime",col_wrap=5, height=2.5)

g = (g.map(sns.kdeplot, "score"))

g = g.map_dataframe(plt.plot, [5,5], [0,1], 'b--').set_axis_labels("", "")

g = g.map_dataframe(plt.plot, [0,10], [0,0], 'b--').set_axis_labels("", "")



plt.show()

g.set(yticklabels=[])

selected_genres = df['genre'].value_counts(normalize=True).index[0:15].values



df2 = df.loc[df['genre'].isin(selected_genres),:]



sns.set(rc={'figure.figsize':(12.7,8.27)})

g = sns.boxplot(x="genre", y="score", data=df2).set_title('Score distribution in Pitchfork reviews depending on genre')

plt.xticks(rotation=90)
selected_authors = df['author'].value_counts(normalize=True).index[0:20].values



df2 = df.loc[df['author'].isin(selected_authors),:]



g = sns.boxplot(x="author", y="score", data=df2).set_title('Score distribution in Pitchfork reviews depending on author')

plt.xticks(rotation=90)
g = sns.distplot(df["score"]).set_title('Comparison of expected and observed score distribution of Pitchfork reviews')

plt.xticks(np.arange(min(df["score"]), max(df["score"])+1, 1.0))
selected_authors = df['role'].value_counts(normalize=True).index[0:10].values



df2 = df.loc[df['role'].isin(selected_authors),:]



g = sns.boxplot(x="role", y="score", data=df2).set_title('Score distribution in Pitchfork reviews depending on role')

plt.xticks(rotation=90)
df["ds"]=df["date"]

df["y"]=df["score"]



model = fbprophet.Prophet()

model.fit(df)



future = model.make_future_dataframe(periods=1)

forecast = model.predict(future)
fig = model.plot(forecast)
fig2 = model.plot_components(forecast)
most_important_genres = df['genre'].value_counts()[0:10]

print(most_important_genres)

selected_genres=most_important_genres[[0,1,3,4,6]].index.tolist()

selected_genres
res = df[(df['genre']!= "No info")].groupby(['root_month','genre'])['artist'].agg('count')

res = res / res.groupby(level=0).sum()

res = res.to_frame().reset_index()

res=res[res['genre'].isin(selected_genres)].rename(columns={"root_month": "ds", "artist": "y"}).reset_index()

res
def get_prediction(df):

    df['smoothed_count']=0 

    list_genres = df.genre.unique()



    for article in list_genres:

       # article_df = df.loc[df['genre'] == article]

       # my_model = fbprophet.Prophet()

       # my_model.fit(article_df)

       # future_dates = my_model.make_future_dataframe(periods=1)

       # forecast = my_model.predict(future_dates)

       # forecast=forecast.loc[forecast['ds']<=np.max(df['ds'])]

       # df.loc[df['genre'] == article,'smoothed_count'] = forecast['yhat'].values

        df.loc[df['genre'] == article,'smoothed_count']= df.loc[df['genre'] == article,'y'].rolling(6).mean().values

    

    return df
res2 = get_prediction(res)

res2["smoothed_count"]=pd.to_numeric(res2["smoothed_count"])

from  matplotlib.ticker import PercentFormatter

ax = sns.lineplot(x="ds", y="smoothed_count",hue="genre", data=res2)

ax.yaxis.set_major_formatter(PercentFormatter(1))

ax.set_title('Time-series evolution of % of reviews of this style over the total volume')