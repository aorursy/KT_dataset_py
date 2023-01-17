# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns #for better and easier plots



#plotting directly without needing to call plot.show()

%matplotlib inline 



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
data = pd.read_csv("../input/master.csv", parse_dates=True) #loading the file, parse_dates = True
data.head() #checking the first 5 top entries
print(data.shape) #checking the shape of the data
data.groupby("year")['suicides_no'].sum().plot()#grouping by year, making it easier to look for trends

plt.title("Distribution of suicides by year", fontsize=20)
#let's use a countplot and group by countries, to have an idea of the distributionof suicide rates per country



sns.set(rc={'figure.figsize':(10,20)}) #setting the figure size

ax = sns.countplot(y="country", data=data.sort_values(ascending=False, by="suicides_no"))

#using countplot, assign country to y to make the plot horizontal

plt.yticks(fontsize=13) #rotating the labels to make it readable

plt.title("Suicide rates by countries", fontsize=20) #title
sns.set(rc={'figure.figsize':(22,4)}) #setting the figure size

by_country = data.groupby("country")['suicides_no'].sum()

by_country.sort_values(ascending=False).head(50).plot(kind='bar')#to make the chart clearer, gonna show only the top 50

#grouping by country, making it easier to look for trends

plt.xticks(fontsize=15)

plt.title("Distribution of suicides by country", fontsize=20)



ax = sns.countplot(x = "age", hue="generation", data=data )#using countplot, assign country to y to make the plot horizontal

ax = sns.set(rc={'figure.figsize':(10,4)})

plt.xticks(rotation = 45, fontsize=13) #rotating the labels to make it readable

plt.title("Suicide rates by Age", fontsize=15) #title
g = sns.FacetGrid(data, col="generation", hue="sex")

g.map(plt.scatter,"population","suicides_no", alpha=.7)

g.add_legend()
g = sns.FacetGrid(data, col="generation", hue="sex")

g.map(plt.scatter,"HDI for year","suicides_no", alpha=.7)

g.add_legend()
corr = data.corr()

corr["suicides_no"].sort_values(ascending=False)#relative to suicides numbers
corr['HDI for year'].sort_values(ascending=False) #now let's take a look at the correlation relative to HDI
#only shows null values. 

## shows the percentage of null values

def missing_values_calculate(trainset): 

    nulldata = (trainset.isnull().sum() / len(trainset)) * 100

    nulldata = nulldata.drop(nulldata[nulldata == 0].index).sort_values(ascending=False)

    ratio_missing_data = pd.DataFrame({'Ratio' : nulldata})

    return ratio_missing_data.head(30)
missing_values_calculate(data)
data.groupby("HDI for year")['suicides_no'].sum().plot()
sns.set(rc={'figure.figsize':(22,4)}) #setting the figure size

data.groupby('country')['HDI for year'].mean().sort_values(ascending=False).head(20).plot(kind='bar') #top 20 countries by HDI

plt.xticks(fontsize=15)

plt.title("TOP 20 countries by Human development index", fontsize=20)
sns.set(rc={'figure.figsize':(22,4)}) #setting the figure size

data["HDI for year"] = data["HDI for year"].round(2) #making it easier to read, rounding the value up to 2 decimals.

data.groupby(["country","HDI for year"])["suicides_no"].sum().sort_values(ascending=False).head(80).plot(kind="bar", stacked=True)

plt.xticks(fontsize=15)

plt.title("Distribution of suicides by country and HDI", fontsize=20)
##to be continued