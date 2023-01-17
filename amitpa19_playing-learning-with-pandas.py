# find out your current directory

import os

os.getcwd()
# if you want to set a different working directory

#os.chdir("path")
# to get a list of all files in the directory

os.listdir()
# import pandas and numpy libraries

import pandas as pd

import numpy as np
# import a csv file from local machine

#df = pd.read_csv("beer.csv")

#df.head(4)
# import a csv file from an online database

df_Web = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

df_Web.head(4)
# description of index, entries, columns, data types, memory info

df_Web.info() 
# check out first few rows

df_Web.head(5) # head
# number of columns & rows

df_Web.shape 
# column names

df_Web.columns 
# number of unique values of a column

df_Web["species"].nunique()
# show unique values of a column

df_Web["species"].unique()
# number of unique values alltogether

df_Web.columns.nunique()
# value counts

df_Web['species'].value_counts()
df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
# show null/NA values per column

df_Web.isnull().sum()
# show null/NA values per column

df.isnull().sum()
# show NA values as % of total observations per column

df.isnull().sum()*100/len(df)
# drop all rows containing null

df1= df.dropna()

df1.isnull().sum()*100/len(df)
# drop all columns containing null

df2 = df.dropna(axis=1)

# show NA values as % of total observations per column

df2.isnull().sum()*100/len(df)
# drop columns with less than 5 NA values

df3 = df.dropna(axis=1, thresh=5)

# show NA values as % of total observations per column

df3.isnull().sum()*100/len(df)
# replace all na values with -9999

df4 = df.fillna(-9999)

df4.head(4)
# fill na values with NaN

df5 = df.fillna(np.NaN)

df5.head(4)
# fill na values with strings

df6=df.fillna("data missing")

df6.head(4)
# fill missing values with mean column values

df7=df.fillna(df.mean())

df7.head(4)
# replace na values of specific columns with mean value

df["sepal_length"] = df["sepal_length"].fillna(df["sepal_length"].mean())

df.isnull().sum()*100/len(df)
# interpolation of missing values (useful in time-series)

df7 = df["sepal_length"].interpolate()
df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
# select a column

df["sepal_length"]
# select multiple columns and create a new dataframe X

X = df[["sepal_length", "sepal_width", "species"]]

X
# select a column by column number

df.iloc[:, [1,3,4]]
# drop a column from dataframe X

X = X.drop("sepal_length", axis=1)

X
# save all columns to a list

df.columns.tolist()
# Rename columns

df.rename(columns={"sepal_length": "Sepal_Length", "sepal_width": "Sepal_Width"})
# sorting values by column "sepalW" in ascending order

df.sort_values(by = "sepal_width", ascending = True)
# add new calculated column

df['newcol'] = df["sepal_length"]*2

df.head(4)
# create a conditional calculated column

df['newcol'] = ["short" if i<3 else "long" for i in df["sepal_width"]] 

df.head(4)
# select rows 3 to 10

df.iloc[3:10,]
# select rows 3 to 49 and columns 1 to 3

df.iloc[3:50, 1:4]
# randomly select 10 rows

df.sample(10)
# find rows with specific strings

df[df["species"].isin(["setosa"])]
# conditional filtering

df[df.sepal_length >= 5]
# filtering rows with multiple values e.g. 0.2, 0.3

df[df["petal_width"].isin([0.2, 0.3])]
# multi-conditional filtering

df[(df.petal_length > 1) & (df.species=="Iris-setosa") | (df.sepal_width < 3)]
# drop rows

df.drop(df.index[1]) # 1 is row index to be deleted
# data grouped by column "species"

X = df.groupby("species")

X.head()
# return mean values of a column ("sepal_length" ) grouped by "species" column

df.groupby("newcol")["sepal_length"].mean()
# return mean values of ALL columns grouped by "species" category

df.groupby("species").mean()
# get counts in different categories

df.groupby("species").nunique() 