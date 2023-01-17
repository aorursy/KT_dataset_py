import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
gdata = pd.read_csv('../input/googleplaystore.csv') 

#change backward slash to forward slash to avoid unicode error message
gdata.head() #by default it displays 5 rows
gdata.shape
gdata.describe() #statistics
gdata.boxplot()
gdata.hist()
gdata.info()
gdata.isnull()
gdata.isnull().sum()
#in boxplot we saw one value is outlier. But now criss check how many are outliers

gdata[gdata["Rating"] > 5]
#drop it

gdata.drop([10472], inplace=True)
gdata[10470:10474] #to check if it has been dropped
gdata.boxplot() #now we can see there are no outliers, all are under rating 5
gdata.hist() #it's rightly skewed
threshold = len(gdata) * 0.1 

threshold

#this is 10% of 10840 (total no. of rows), so using this value we will check which col has 10% values(i.e 90% empty)
gdata.dropna(thresh = threshold, axis = 1, inplace = True) #axis is 1 bcz we are doing for col
print(gdata.isnull().sum())
gdata.shape
def compute_median(series):

    return series.fillna(series.median())
gdata.Rating = gdata["Rating"].transform(compute_median)
#check for null values again

gdata.isnull().sum()
#check the modes of Type, Current Ver and Android Ver to be sure before filling missing values

print(gdata["Type"].mode())

print(gdata["Current Ver"].mode())

print(gdata["Android Ver"].mode())
# fill the missing values

gdata["Type"].fillna(str(gdata["Type"].mode().values[0]), inplace=True)

gdata["Current Ver"].fillna(str(gdata["Type"].mode().values[0]), inplace=True)

gdata["Android Ver"].fillna(str(gdata["Type"].mode().values[0]), inplace=True)
gdata.isnull().sum()
# convert the attributes like: Price (which is in $), Reviews and Installs to numeric (without symbols also) 

gdata["Price"] = gdata["Price"].apply(lambda x: str(x).replace("$", '') if "$" in str(x) else str(x))

gdata["Price"] = gdata["Price"].apply(lambda x: float(x))

gdata["Reviews"] = pd.to_numeric(gdata["Reviews"], errors= "coerce")

gdata["Installs"] = gdata["Installs"].apply(lambda x: str(x).replace("+", '') if "+" in str(x) else str(x))

gdata["Installs"] = gdata["Installs"].apply(lambda x: str(x).replace(",", '') if "," in str(x) else str(x))

gdata["Installs"] = gdata["Installs"].apply(lambda x: float(x))
gdata.head(10)
gdata.describe() #previously we had only Rating, now we have 4 columns
grp = gdata.groupby("Category")

x = grp["Rating"].agg(np.mean)

y = grp["Price"].agg(np.sum)

z = grp["Reviews"].agg(np.mean)

w = grp["Installs"].agg(np.mean)



print(x, "\n")

print(y, "\n")

print(z, "\n")

print(w)
plt.figure(figsize=(15,5))

plt.plot(x, "o", color="r")

plt.xticks(rotation = 90)

plt.title("Category wise Rating")

plt.ylabel("Rating-->")

plt.xlabel("Categories-->")

plt.show()
plt.figure(figsize=(15,5))

plt.plot(y, "r--", color="r")

plt.xticks(rotation = 90)

plt.title("Category wise Price")

plt.ylabel("Price-->")

plt.xlabel("Categories-->")

plt.show()
plt.figure(figsize=(15,5))

plt.plot(z, "g^", color="r")

plt.xticks(rotation = 90)

plt.title("Category wise Reviews")

plt.ylabel("Reviews-->")

plt.xlabel("Categories-->")

plt.show()
plt.figure(figsize=(15,5))

plt.plot(w, "bs", color="r")

plt.xticks(rotation = 90)

plt.title("Category wise Installs")

plt.ylabel("Installs-->")

plt.xlabel("Categories-->")

plt.show()