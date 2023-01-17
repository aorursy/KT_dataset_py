# Necessary librarys
import os  # it's a operational system library, to set some informations
import random  # random is to generate random values
from ast import literal_eval
import pandas as pd  # to manipulate data frames
import numpy as np  # to work with matrix
from scipy.stats import kurtosis, skew  # it's to explore some statistics of numerical values

import matplotlib.pyplot as plt  # to graphics plot
import seaborn as sns  # a good library to graphic plots

import squarify  # to better understand proportion of categorys - it's a treemap layout algorithm

# Importing librarys to use on interactive graphs
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

import json  # to convert json in df
from pandas.io.json import json_normalize  # to normalize the json file
from datetime import datetime

# to set a style to all graphs
plt.style.use('fivethirtyeight')
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', 500)


def NumericalColumns(df):  # fillna numeric feature
    df['totals.pageviews'].fillna(1, inplace=True)  # filling NA's with 1
    df['totals.newVisits'].fillna(0, inplace=True)  # filling NA's with 0
    df['totals.bounces'].fillna(0, inplace=True)  # filling NA's with 0
    df['trafficSource.isTrueDirect'].fillna(False, inplace=True)  # filling boolean with False
    df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)  # filling boolean with True
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].fillna(0.0).astype(float)  # filling NA with zero
    df['totals.pageviews'] = df['totals.pageviews'].astype(int)  # setting numerical column as integer
    df['totals.newVisits'] = df['totals.newVisits'].astype(int)  # setting numerical column as integer
    df['totals.bounces'] = df['totals.bounces'].astype(int)  # setting numerical column as integer
    df["totals.hits"] = df["totals.hits"].astype(float)  # setting numerical to float
    df['totals.visits'] = df['totals.visits'].astype(int)  # seting as int
    df['totals.totalTransactionRevenue'] = df["totals.totalTransactionRevenue"].fillna(0.0).astype(float)
    df['trafficSource.adwordsClickInfo.page'] = df['trafficSource.adwordsClickInfo.page'].fillna(0)
    return df  # return the transformed dataframe


def delete_constant(df_train):
    discovering_consts = [col for col in df_train.columns if df_train[col].nunique() == 1]
    # drop constant columns and hits
    df_train.drop(discovering_consts, axis=1, inplace=True)

    return (df_train)


def process_custom(df_train):
    df_train['customDimensions'] = df_train['customDimensions'].replace(to_replace=r'\'', value='\"', regex=True)

    df_train['customDimensions'] = df_train['customDimensions'].apply(literal_eval)
    df_train['customDimensions'] = df_train['customDimensions'].map(lambda a: 0 if (a == []) else a[0]['value'])

    df_train['totals.transactionRevenuelog'] = df_train['totals.transactionRevenue'].apply(lambda x: np.log1p(x))
    return (df_train)


# This function is to extract date features
def date_process(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")  # seting the column as pandas datetime
    df["_weekday"] = df['date'].dt.weekday  # extracting week day
    df["_day"] = df['date'].dt.day  # extracting day
    df["_month"] = df['date'].dt.month  # extracting day
    df["_year"] = df['date'].dt.year  # extracting day
    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
    df['on_weekend'] = df['date'].apply(lambda x: x.dayofweek >= 5).astype('bool')
    df['at_night'] = df['date'].apply(lambda x: x.hour <= 5 or x.hour >= 21).astype('bool')

    return df  # returning the df after the transformation


def na_values(df_train):
    # values to convert to NA
    na_vals = ['unknown.unknown', '(not set)', 'not available in demo dataset',
               '(not provided)', '(none)', '<NA>']

    df_train = df_train.replace(na_vals, np.nan, regex=True)
    return (df_train)


def categorical_data(df_train):
    cat_cols = ['channelGrouping', 'device.operatingSystem', 'geoNetwork.region', 'geoNetwork.metro', 'geoNetwork.city',
                'trafficSource.source',
                '_day', '_month', '_weekday', '_year', 'totals.newVisits', 'device.browser', 'device.deviceCategory',
                'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.networkDomain',
                'geoNetwork.region', 'geoNetwork.subContinent', 'trafficSource.keyword', 'trafficSource.medium',
                'trafficSource.referralPath', 'trafficSource.source', ]
    for column in cat_cols:
        df_train[column] = df_train[column].astype('category')

    return (df_train)




# Code to transform the json format columns in table
def json_read(data_frame):
    cols = list(pd.read_csv(data_frame, nrows=1))

    columns = ['device', 'geoNetwork', 'totals', 'trafficSource']  # Columns that have json format
    for df in pd.read_csv(data_frame, converters={column: json.loads for column in columns},
                          dtype={'fullVisitorId': 'str'}, usecols =[i for i in cols if i != 'hits'], chunksize=100000):
        print(df.head(2))
        # Importing the dataset
        df = df.replace(to_replace=r'\'', value='\"', regex=True)
        columns = ['device', 'geoNetwork', 'totals', 'trafficSource']  # Columns that have json format

        for column in columns:  # loop to finally transform the columns in data frame
            # It will normalize and set the json to a table
            column_as_df = json_normalize(df[column])
            # here will be set the name using the category and subcategory of json columns
            column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
            column_as_df=column_as_df.set_index(df.index)
            # after extracting the values, let drop the original columns
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

        df = na_values(df)

        df = NumericalColumns(df)
        df = process_custom(df)
        df = date_process(df)
        df=delete_constant(df)
        df.to_csv("New_train_v2.csv", index=False, header=True, mode='a')
        

if __name__ == "__main__":

# We will import the data using the name and extension that will be concatenated with dir_path
      json_read('../input/train_v2.csv')

    
import pandas as pd  # to manipulate data frames
testCSV=pd.read_csv('../input/test_v2.csv',sep=',')
testCSV.head()
import pandas as pd
trainCSV = pd.read_csv('New_train_v2.csv', sep=',', encoding='utf-8', low_memory=False)
trainCSV.head()
# the categorical variables for train  data
categorical_features_train = trainCSV.select_dtypes(include=[np.object])
categorical_features_train
from datetime import timedelta
cool_columns = ['fullVisitorId', 'date', 'geoNetwork.city', 'geoNetwork.country', 'device.browser',
                'visitNumber',  'totals.totalTransactionRevenue','_weekday','_day','_month','_year','_visitHour','on_weekend']

trainCSV2 = trainCSV[cool_columns]
trainCSV2.head()
trainCSV2.loc[:,['totals.totalTransactionRevenue']].mean()
#exemple*
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
tmp = trainCSV['date'].value_counts().to_frame().reset_index().sort_values('index')
#Visualization for Visits by date
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
tmp = trainCSV2['date'].value_counts().to_frame().reset_index().sort_values('index')
tmp = tmp.rename(columns = {"index" : "dateX", "date" : "visits"})

tr = go.Scatter(mode="lines", x = tmp["dateX"].astype(str), y = tmp["visits"])
layout = go.Layout(title="Visits by Date", height=400)
fig = go.Figure(data = [tr], layout = layout)
iplot(fig)
# Visualization for Visits by monthly revenue
tmp = trainCSV2.groupby("date").agg({"totals.totalTransactionRevenue" : "mean"}).reset_index()
tmp = tmp.rename(columns = {"date" : "dateX", "totals.totalTransactionRevenue" : "mean_revenue"})
tr = go.Scatter(mode="lines", x = tmp["dateX"].astype(str), y = tmp["mean_revenue"])
layout = go.Layout(title="Monthly Revenue by Date", height=400)
fig = go.Figure(data = [tr], layout = layout)
iplot(fig)
#Create a train and validation sets 
from sklearn.model_selection import train_test_split
features = [c for c in trainCSV2.columns]
features.remove("totals.totalTransactionRevenue")
trainCSV2["totals.totalTransactionRevenue" ] = np.log1p(trainCSV2["totals.totalTransactionRevenue"].astype(float))
train_x, valid_x, train_y, valid_y = train_test_split(trainCSV2[features], 
                                                      train["totals.totalTransactionRevenue"], test_size=0.25, random_state=20)
#Create LGBM model and train it !
import lightgbm as lgb 

lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 50, "learning_rate" : 0.02, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9}
    
lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_val = lgb.Dataset(valid_x, label=valid_y)
model = lgb.train(lgb_params, lgb_train, 700, valid_sets=[lgb_val], early_stopping_rounds=150, verbose_eval=20)



clf = RandomForestClassifier(class_weight='balanced')
param_grid = {
    'min_samples_leaf' : [2, 5, 10, 20], # Best:
    'max_depth': [2, 10, 20], # Best:
    'n_estimators': [5, 20, 100, 200] # Best:      
}

search = GridSearchCV(clf, param_grid)
search.fit(train_x, train_y)

y_predicted_probability = search.predict_proba(valid_x)[:,1]
plot_roc_curve(valid_y, y_predicted_probability, title="ROC in test set RF")
plot_precision_recall_curve(valid_y, y_predicted_probability)

all_features =pd.DataFrame({'feature': train_x.columns, 'importance': search.best_estimator_.feature_importances_})
all_features = all_features.sort_values(by=['importance'], ascending=False).set_index('feature')
all_features[all_features.importance>0].plot.bar()

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
clf = XGBClassifier()
param_grid = {
    'max_depth':[8],    
    'n_estimators': [3], 
    'learning_rate' : [0.1], 
    'min_child_weight' : [100], 
    'reg_lambda': [30] 
}

search = GridSearchCV(clf, param_grid)
search.fit(train_x, train_y)

y_predicted_probability = search.predict_proba(valid_x)[:,1]
plot_roc_curve(valid_y, y_predicted_probability, title="ROC in test set RF")
plot_precision_recall_curve(valid_y, y_predicted_probability)

all_features =pd.DataFrame({'feature': train_y.columns, 'importance': search.best_estimator_.feature_importances_})
all_features = all_features.sort_values(by=['importance'], ascending=False).set_index('feature')
all_features[all_features.importance>0].plot.bar()

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)