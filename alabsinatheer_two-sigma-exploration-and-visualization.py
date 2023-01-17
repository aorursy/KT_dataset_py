import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.plotly as py

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 120)
# read-in the training data

with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
# I will keep a copy of the dataframe for later use

df1 = df.copy()
df.head()
# let's check the size of the dataframe

print("Number of rows:", df.shape[0])

print("Number of columns:", df.shape[1])
# Extract the basic statistics 

df.describe()
# Let's look how many missing values in each column 

# We will sort missing values to bring column with the highest number of missing values at the top

# We will see the number of missing values in each of the first 30 columns



df.isnull().sum().sort(axis=0, ascending=False, inplace=False).head(30)
# But for now I am going to inpute any missing values with the mean

# I will deal with this major issue later on



df = df.fillna(df.mean()['derived_0':'y'])
# Double check the missing values

df.isnull().sum().head(30)
# Let's see the distribution of the target variable



df["y"].hist(bins = 30, color = "orange")

plt.xlabel("Target Variable")

plt.ylabel("Frequency")
# Take absolute values 

#df.loc[:, "derived_0": "technical_44"] = df.loc[:, "derived_0": "technical_44"].abs()
# Groupby the target variable

df_f = (df.groupby(pd.cut(df["y"], [-0.087,-0.067,-0.047,-0.027,-0.007,0.013,0.033,0.053,0.073,0.094], right=False))

        .mean())
df_f.head()
# The correlation matrix

cor_mat = df_f.corr(method='pearson', min_periods=1).sort(axis=0, ascending=False, inplace=False)

cor_mat.head(20) # Look at the first 20 columns
# The correlation with the target variable sorted in a descending order

# Look at the first 30 parameters

cor_mat.iloc[0,:].sort(axis=0, ascending=False, inplace=False).head(30) 
alpha = plt.figure()

plt.scatter(df_f["technical_0"], df_f["y"], alpha=.1, s=400)

plt.xlabel("technical_0") 

plt.ylabel("Target variable")

plt.show()
alpha = plt.figure()

plt.scatter(df_f["technical_24"], df_f["y"], alpha=.1, s=400)

plt.xlabel("technical_24") 

plt.ylabel("Target variable")

plt.show()
alpha = plt.figure()

plt.scatter(df_f["technical_44"], df_f["y"], alpha=.1, s=400)

plt.xlabel("technical_44") 

plt.ylabel("Target variable")

plt.show()
df_a = df1.dropna(axis=0)
len(df_a)
# Again groupby the target variable

df_f = (df_a.groupby(pd.cut(df_a["y"], [-0.087,-0.067,-0.047,-0.027,-0.007,0.013,0.033,0.053,0.073,0.094], right=False))

        .mean())
# The correlation matrix

cor_mat = df_f.corr(method='pearson', min_periods=1).sort(axis=0, ascending=False, inplace=False)

cor_mat.head(20)
# The correlation with the target variable sorted in a descending order

cor_mat.iloc[0,:].sort(axis=0, ascending=False, inplace=False).head(20)
cor_matt = cor_mat.iloc[0,:].sort(axis=0, ascending=False, inplace=False)
# Extract the most 10 correlated variables

cor_matt = cor_matt.keys()[1:10]
for i in cor_matt:

    alpha = plt.figure()

    plt.scatter(df_f[i], df_f["y"], alpha=.1, s=400)

    plt.xlabel(i) 

    plt.ylabel("Target Variable")

    plt.show()
# Let's take a 1000 sample of the data to explore 

# We will use raw data which has the missing data removed from it

df_m = df_a.sample(n=1000)
# Plot the most correlated variables 

for i in cor_matt:

    alpha = plt.figure()

    plt.scatter(df_m["timestamp"], df_m[i], alpha=.5)

    plt.xlabel("timestamp") 

    plt.ylabel(i)

    plt.show()