import sys

import scipy

import numpy

import matplotlib

import pandas

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
url = "../input/nypddata/NYPD_Arrests_Data__Historic_.csv"

full = ["ARREST_KEY", "ARREST_DATE", "PD_CD", "LAW_CAT_CD", "ARREST_BORO", "ARREST_PRECINCT", "AGE_GROUP", "PERP_SEX", "PERP_RACE", "Latitude", "Longitude"]

demo = ["ARREST_KEY","LAW_CAT_CD", "PD_DESC", "ARREST_BORO", "AGE_GROUP", "PERP_SEX", "PERP_RACE", "Latitude", "Longitude"]

types = ["ARREST_DATE", "PD_CD", "PD_DESC", "LAW_CAT_CD", "ARREST_BORO", "ARREST_PRECINCT"]
dataset = pandas.read_csv(url, usecols = demo, index_col = 0)
dataset
dataset.PD_DESC.unique()
listDesc = ['INTOXICATED DRIVING,ALCOHOL', 'ALCOHOLIC BEVERAGE CONTROL LAW','IMPAIRED DRIVING,ALCOHOL']

dataset = dataset[dataset['PD_DESC'].isin(listDesc)]

dataset

sns.countplot(x='PD_DESC',data=dataset)
listAge = ['25-44', '18-24', '45-64', '65+', '<18']  #To remove bad age values

dataset = dataset[dataset['AGE_GROUP'].isin(listAge)]

dataset

sns.countplot(x='AGE_GROUP',data=dataset)
sns.countplot(x='PERP_SEX', data=dataset)
sns.countplot(x="PERP_RACE",data=dataset)
listCat = ['V', 'F', 'M']  #To remove bad age values

dataset = dataset[dataset['LAW_CAT_CD'].isin(listCat)]

dataset

sns.countplot(x="LAW_CAT_CD",data=dataset)
sns.countplot(x="ARREST_BORO",data=dataset)
listCat = ['V', 'M', 'F']

dataset = dataset[dataset['LAW_CAT_CD'].isin(listCat)]

dataset
dataset.ARREST_BORO.unique()
listBoro=['Q', 'K', 'M', 'S', 'B']

dataset = dataset[dataset['ARREST_BORO'].isin(listBoro)]

dataset
dataset.AGE_GROUP.unique()
dataset.PERP_SEX.unique()
dataset.PERP_RACE.unique()