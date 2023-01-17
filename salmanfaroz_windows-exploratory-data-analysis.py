from pandas import read_csv, Grouper, DataFrame, concat

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline
ws=pd.read_csv("../input/windows-store/msft.csv")
ws.head(5)
ws.axes
ws.nunique()
ws.isnull().mean().sort_values(ascending=False)
#for removing null rows

ws=ws.dropna()



ws.isnull().mean().sort_values(ascending=False)
sns.set(font_scale=1.4)

#for bar plot

ws["Rating"].value_counts().plot(kind='bar', figsize=(7, 5), rot=0)

plt.xlabel("Ratings", labelpad=14)

plt.ylabel("Number of apps", labelpad=14)

plt.title("Number of App Counts by Ratings", y=1.02)
sns.set(font_scale=1.4)

#for barh plot

ws["Category"].value_counts().plot(kind='barh', figsize=(15, 5), rot=0)

plt.xlabel("Ratings", labelpad=14)

plt.ylabel("Category of the App", labelpad=14)

plt.title("Number of App by Category", y=1.02);
rt=ws.loc[:,["Name","Rating","No of people Rated"]]



#sorting the values by rating 

rt=rt.sort_values(by="Rating",ascending=False)



#taking only 5 ratings

rt_people=rt[rt["Rating"]==5.0]



#sorting the ratings by people count

rt_top20=rt_people.sort_values(by="No of people Rated",ascending=False)



rt_top20.head(20).plot.barh(x='Name', y='No of people Rated',figsize=(9, 10), rot=0)

plt.xlabel("No of people Rated")

plt.ylabel("Name of the App", labelpad=14)

plt.title("\n\nTop 20 App with High number of users and ratings", y=1.02);
#preprocessing the price column

ws["Price"]=ws["Price"].str.replace("Free","0")

ws["Price"]=ws["Price"].str.replace("₹ ","")

ws["Price"]=ws["Price"].str.replace(",","")

ws["Price"]=ws["Price"].astype("float")



pr=ws["Price"].sort_values()

free=0

cost=0

for i in pr:

    if i==0.0:

        free+=1

    else:

        cost+=1

#Total sum of app price based on free and cost.

top=[('Free',free),('Cost',cost)]



labels, ys = zip(*top)

xs = np.arange(len(labels)) 

width = 1



plt.bar(xs, ys, width, align='center', color=("blue","orange"))

plt.title("Count of free and cost Apps")

plt.xticks(xs, labels) 

plt.yticks(ys)

# Converting Date column values to Date format

ws['Date']= pd.to_datetime(ws['Date'], format="%d-%m-%Y")



# soring the values based on Date

dte = ws.sort_values(by='Date', ascending=True)



# Setting index as the date 

dte.index = dte.Date



# Resampling the data based on the year

yr=dte.resample('Y').mean()



# Setting fiqure size

sns.set(rc={'figure.figsize':(10,5)})

sns.barplot(x=yr.index.year, y="No of people Rated", data=yr)

plt.xlabel("Year", labelpad=17)

plt.ylabel("Number of users", labelpad=14)



plt.title("Number of people rated yearly - Average ", y=1.01);
sns.set(rc={'figure.figsize':(18,5)})

sns.boxplot(x="Rating",y="No of people Rated",data=ws,hue="Category")



# adding legend outside of the plot

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.boxplot(x="Rating",y="Price",data=ws)

plt.title("Rating vs Price")