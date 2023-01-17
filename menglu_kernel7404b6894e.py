import random

import math

import time

import string

import nltk

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer,PorterStemmer

import tensorflow_hub as hub

import tensorflow as tf

import pyLDAvis.gensim

from tqdm import tqdm

import seaborn as sns

import pandas as pd

import numpy as np

import pyLDAvis

import gensim

import spacy

import json

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import os

import plotly.graph_objects as go
##### take a look at data 

train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

print(train.head())



train.describe()

train.info()

train.isnull().sum()



test= pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

print(test.head())



test.describe()

test.info()

test.isnull().sum()

# Grouping confirmed cases per country

grouped_country_train = train.groupby(["Country/Region"],as_index=False)["ConfirmedCases"].last().sort_values(by="ConfirmedCases",ascending=False)



# Using just first 10 countries with most cases in training dataset

most_common_countries_train = grouped_country_train.head(10)

# FUNCTION TO SHOW ACTUAL VALUES ON BARPLOT



def show_valushowes_on_bars(axs):

    def _show_on_single_plot(ax):

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.2f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center")



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)



# Barplot: confirmed,recovered and death cases per country



plt.figure(figsize=(13,23))



# AXIS 1

plt.subplot(311)

vis_1 = sns.barplot(x="Country/Region",y="ConfirmedCases",data=most_common_countries_train)

vis_1.set_title("Confirmed cases in training dataset")

show_valushowes_on_bars(vis_1)

most_common_countries_train_df = train.groupby(["Country/Region", "Date"])[["ConfirmedCases", "Fatalities"]].sum().reset_index()

print("# of Entries:", most_common_countries_train_df.shape[0])

print("# of Non-Zero Entries:", most_common_countries_train_df[most_common_countries_train_df.ConfirmedCases > 0].shape[0])

print("# of Countries:", most_common_countries_train_df["Country/Region"].nunique())

print("# of Countries with confirmed cases:", most_common_countries_train_df[most_common_countries_train_df.ConfirmedCases > 0]["Country/Region"].nunique())

SAMPLED_COUNTRIES = ["Italy", "Spain", "Germany", "Iran", "Korea,South", "Switzerland", "United Kingdom", "Netherlands"]

# Reference: https://plot.ly/python/time-series/

fig = go.Figure(

    [

        go.Scatter(

            x=most_common_countries_train_df[most_common_countries_train_df["Country/Region"] == name]['Date'], 

            y=most_common_countries_train_df[most_common_countries_train_df["Country/Region"] == name]['ConfirmedCases'], 

            name=name

        ) for name in SAMPLED_COUNTRIES

    ],

    layout_title_text="Cumulative number of cases in Selected Countries"

)

fig.update_layout(

    yaxis_type="log",

    margin=dict(l=20, r=20, t=50, b=20),

    template="plotly_white")

fig.show()
train_df = train.groupby(["Country/Region", "Date"])[["ConfirmedCases", "Fatalities"]].sum().reset_index()

train_df.describe()

train_df.info()

train_df.isnull().sum()

print(train_df.head())
from sklearn import preprocessing

test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)

test["Lat"]  = test["Lat"].fillna(12.5211)

test["Long"]  = test["Long"].fillna(69.9683)

test.isnull().sum()

test["Date"].min()



train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))

train["Date"]  = train["Date"].astype(int)

train["Lat"]  = train["Lat"].fillna(12.5211)

train["Long"]  = train["Long"].fillna(69.9683)

train.isnull().sum()

train["Date"].min()



x = train[['Lat', 'Long', 'Date']]

y1 = train[['ConfirmedCases']]

y2 = train[['Fatalities']]

x_test = test[['Lat', 'Long', 'Date']]



tol = [1e-4, 1e-3, 1e-2]

alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]

alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]



bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}



bayesian = BayesianRidge()

bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

bayesian_search.fit(x,y1)

bayesian_search.best_params_

bayesian_confirmed = bayesian_search.best_estimator_

test_bayesian_pred = bayesian_confirmed.predict(x_test)



from sklearn.tree import DecisionTreeClassifier

Tree_model = DecisionTreeClassifier(criterion='entropy')

##

Tree_model.fit(x,y1)

pred1 = Tree_model.predict(x_test)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]

##

Tree_model.fit(x,y2)

pred2 = Tree_model.predict(x_test)

pred2 = pd.DataFrame(pred2)

pred2.columns = ["Death_prediction"]
Sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

Sub.columns

sub_new = Sub[["ForecastId"]]

res = pd.concat([pred1,pred2,sub_new],axis=1)

res.head()

res.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

res = res[['ForecastId','ConfirmedCases', 'Fatalities']]

res["ConfirmedCases"] = OP["ConfirmedCases"].astype(int)

res["Fatalities"] = OP["Fatalities"].astype(int)
#### By using DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor  

regressor = DecisionTreeRegressor(random_state = 0) 

regressor.fit(x,y1)

pred_r1 = regressor.predict(x_test)

pred_r1 = pd.DataFrame(pred_r1)

pred_r1.columns = ["ConfirmedCases_prediction"]

regressor.fit(x,y2)

pred_r2 = regressor.predict(x_test)

pred_r2 = pd.DataFrame(pred_r2)

pred_r2.columns = ["Death_prediction"]

res_dr = pd.concat([sub_new,pred_r1,pred_r2],axis=1)

res_dr.head()

res_dr.columns = [ 'ForecastId','ConfirmedCases', 'Fatalities']

res_dr["ConfirmedCases"] = OP_dr["ConfirmedCases"].astype(int)

res_dr["Fatalities"] = OP_dr["Fatalities"].astype(int)

res_dr.to_csv("submission.csv",index=False)
res_dr.head()