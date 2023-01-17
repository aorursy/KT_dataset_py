import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

data.home_score.plot(kind = 'line', color = 'g',label = 'Home Score',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.away_score.plot(color = 'r',label = 'Away Score',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')    

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

data.plot(kind='scatter', x='home_score', y='away_score',alpha = 0.5,color = 'red')

plt.xlabel('home_score')              # label = name of label

plt.ylabel('away_score')

plt.title('Home Score - Away Score ') 
# Histogram

data.home_score.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
x = data['home_score'] > 5

data[x]
y = data['away_score'] > 5

data[y]
data[(data['home_score']>3) & (data['away_score']>3)]
data.head() # head shows first 5 rows
data.tail() # tail shows last 5 rows
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble 

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column

data.info()
# value_counts() : Frequency counts 

print(data['home_team'].value_counts(dropna="False")) # if there are nan values that also be counted.
print(data['away_team'].value_counts(dropna="False"))
print(data['city'].value_counts(dropna="False"))
# outliers: The value is considerably higher or lower from rest of the data

# count: Number of entries

# mean: Average of entries

# std: Standart deviation

# min: Minimum entry

# 25%: First quantile

# 50%: Median or Second quantile

# 75%: Third quantile

data.describe()
# Box Plots: Visualize basic statistics like outliers, min/max or quantiles

data.boxplot(column='home_score', by='neutral')

plt.show()
#Tidy Data - melt()

data_new = data.head()

data_new
# Melt etmek : Datayı farklı bir yapıya büründürmek

melted = pd.melt(frame = data_new, id_vars = 'home_team', value_vars = ['away_team', 'home_score'])

print(melted)
# Concatenating Data - We can concatenate two DATAFRAME

data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1, data2], axis = 0, ignore_index = True)

conc_data_row
data1 = data['home_score'].head()

data2 = data['away_score'].head()

conc_data_col = pd.concat([data1, data2], axis = 1)

conc_data_col
# Data Types

data.dtypes
data['home_team'] = data['home_team'].astype('category')

data['away_score'] = data['away_score'].astype('float')

data.dtypes
# Building Data Frame From Scratch

 # Data Frames From Dictionary

country = ["Spain", "France"]

population = ["11", "12"]

list_label = ["country", "population"]

list_col = [country, population]

zipped = list(zip(list_label, list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["capital"] = ["madrid", "paris"]

df
# Broadcasting

df["income"] = 0 #Broadcasting entire column

df
# Visual Exploratory Data Analysis

 # Plotting all data

data1 = data.loc[:, ["home_score", "away_score"]]

data1.plot()

# it is confusing
data1.plot(subplots = True)

plt.show()
# Scatter plot

data1.plot(kind = "scatter", x = "home_score", y = "away_score")

plt.show()
# Hist plot

data1.plot(kind = "hist", y="home_score", bins = 50, range=(0,5), normed = False)

plt.show()
# Indexing Pandas Time Series

time_list = ["1992-03-08", "1992-04-12"]

print(type(time_list[1])) # date is string

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object)) 
import warnings

warnings.filterwarnings("ignore")

data2 = data.head()

date_list = ["1992-01-10", "1992-02-10", "1992-03-10", "1993-03-15", "1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

data2 = data2.set_index("date")

data2
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
# Resampling Pandas Time Series

data2.resample("A").mean()
data2.resample("M").mean()
# INDEXING DATA FRAMES

data = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')

data.head()
# indexing using square brackets

data["home_score"][1]
# using column attribute and row label

data.home_score[1]
# using loc accessor

data.loc[1,"home_score"]
# selecting only some columns

data[["home_score","away_score"]]
# SLICING DATA FRAME

 # Difference between selecting columns: series and dataframe

print(type(data['home_score'])) # series

print(type(data[['home_score']])) # data frames
# sciling and indexing series

data.loc[1:10,"home_score":"away_score"] 
# reverse slicing

data.loc[10:1:-1,"home_score":"away_score"] 
# from something to end

data.loc[1:10,"tournament":]
# FILTERING DATA FRAMES

 # creating boolean series

boolean = data.home_score > 20

data[boolean]
# combining filters

first_filter = data.home_score > 5

second_filter = data.away_score > 3

data[first_filter & second_filter]
# filtering column based others

data.home_score[data.away_score>15]
# TRANSFORMING DATA

def div(n):

    return n/2

data.home_score.apply(div)
# or we can use lambda function

data.home_score.apply(lambda n : n/2)
# defining column using other columns

data["total_score"] = data.home_score + data.away_score

data.head()
# INDEX OBJECTS AND LABELED DATA

print(data.index.name)

data.index.name = "index_name"

data.head()
# overwrite index

 # first copy of our data to data3 then change index

data3 = data.copy()

 # lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(100,41640,1)

data3.head()
### HIERARCHICAL INDEXING

data = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')

data.head()

data1 = data.set_index(["country","city"])

data1.tail(20)
# PIVOTING DATA FRAMES

dic = {"treatment":["A","A","B","B"], "gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
 # pivoting

df.pivot(index="treatment", columns="gender",values="response")
# STACKING AND UNSTACKING DATAFRAME

df1 = df.set_index(["treatment","gender"])

df1
df1.unstack(level=0)
df.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
# MELTING DATA FRAMES

 # reverse of pivoting

df

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# CATEGORICALS AND GROUPBY

df.groupby("treatment").mean()
df.groupby("treatment").age.mean()
df.groupby("treatment")[["age","response"]].min()
df.info()