import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['axes.facecolor'] = 'white'
air_bnb = pd.read_csv(r"../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
air_bnb.head()
air_bnb.info()
air_bnb.isnull().sum()
air_bnb.hist(figsize = (10,10));
air_bnb['name'].sample(10)
df = air_bnb.copy()
df = df.drop(['id','name','host_name','neighbourhood','host_id','last_review'],axis = 1)
df
df.isnull().sum()
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df
df.isnull().sum()
df.dtypes
from sklearn import preprocessing

num_cols = ['latitude','longitude','price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']



x = df[num_cols].values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

standard_scaler = preprocessing.StandardScaler()

x_scaled = min_max_scaler.fit_transform(x)

x_scaled = standard_scaler.fit_transform(x_scaled)



scaled_num_cols = pd.DataFrame(x_scaled,columns=num_cols)



def convert_to_onehot(dataset):

    from sklearn.preprocessing import OneHotEncoder

    

    obj_cols = dataset.select_dtypes(include = [object])

    for col in obj_cols:

        ohe = OneHotEncoder(handle_unknown='ignore')

        ohe.fit(dataset[[col]])

        ohe_df = pd.DataFrame(ohe.transform(dataset[[col]]).toarray(),columns=ohe.get_feature_names())

        ohe.index = dataset.index

        dataset = dataset.drop([col],axis = 1)

        dataset = pd.concat([dataset,ohe_df],axis = 1)

    return dataset
df = convert_to_onehot(df)

df
df_sample = df.sample(500)
import scipy.cluster.hierarchy as sch # draw dendrogram



fig, ax=plt.subplots(figsize=(30, 15))

dg=sch.dendrogram(sch.linkage(df_sample, method='ward'), ax=ax)