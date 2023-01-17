import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
headers=["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]
auto=pd.read_csv("../input/auto.csv",names=headers)

auto.head()
auto.shape
auto.describe()
auto.info()
missing_values =["?"]

auto=pd.read_csv('../input/auto.csv',names=headers, na_values = missing_values)

auto
auto.isnull()
auto.isnull().sum(axis=0)
auto.drop(columns=['price'])
a=auto[['stroke','normalized-losses','bore','horsepower','peak-rpm']].mean()

auto[['stroke','normalized-losses','bore','horsepower','peak-rpm']].replace(to_replace = np.nan, value = a)
b=auto['num-of-doors'].count()

auto['num-of-doors'].replace(to_replace = np.nan, value = b)
auto['num-of-doors'].value_counts()
b=auto['num-of-doors'].value_counts().idxmax()

auto['num-of-doors'].replace(to_replace = np.nan, value = b)
auto[['bore','stroke','price','peak-rpm']]=auto[['bore','stroke','price','peak-rpm']].astype('float')
auto[['bore','stroke','price','peak-rpm']].dtypes
auto["highway-mpg"] = 235/auto["highway-mpg"]

auto.rename(columns={"highway-mpg":'highway-L/100km'}, inplace=True)

auto.head()
from sklearn import preprocessing

auto[['height']] = auto[['height']].values.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(auto[['height']])

df_normalized = x_scaled

df_normalized
binwidth = (max(auto["horsepower"])-min(auto["horsepower"]))/4

bins = np.arange(min(auto["horsepower"]), max(auto["horsepower"]), binwidth)

bins
group_names = ['Low', 'Medium', 'High']
auto_store=auto['horsepower-binned'] = pd.cut(auto['horsepower'], bins, labels=group_names,include_lowest=True )

auto_store=auto[['horsepower','horsepower-binned']].head(20)
df=pd.DataFrame(auto_store)

df
auto["horsepower-binned"].value_counts()
plt.bar(group_names, df["horsepower-binned"].value_counts())



# set x/y labels and plot title

plt.xlabel("horsepower")

plt.ylabel("count")

plt.title("horsepower bins")
auto.columns
dummy_variable_1 = pd.get_dummies(auto["fuel-type"])

dummy_variable_1.head()
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)

dummy_variable_1.head()
df = pd.concat([auto, dummy_variable_1], axis=1)



auto.drop("fuel-type", axis = 1, inplace=True)
auto.head()
dummy_variable_2 = pd.get_dummies(df["aspiration"])

dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

dummy_variable_2.head()
df = pd.concat([df, dummy_variable_2], axis=1)

df.drop("aspiration", axis = 1, inplace=True)
df