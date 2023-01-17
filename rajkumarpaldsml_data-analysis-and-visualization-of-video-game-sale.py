import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as mtic

import datetime

import os

from mpl_toolkits import mplot3d
game_data = pd.read_csv("../input/vgsales/vgsales.csv")

game_data
game_data.head(10)
game_data.isnull().sum()
print('Categorical Value Count for the following columns:\n')

print('Year:')

print(game_data['Year'].value_counts())

print('\Publisher:')

print(game_data['Publisher'].value_counts())
game_data.dropna(inplace=True)
game_data.isnull().sum()
plt.figure(figsize=(20, 15))

df = game_data.groupby(["Name"])

bplot3 = sns.barplot(x='Global_Sales',y='Name', 

                     data=df.Global_Sales.sum().sort_values(ascending=False).head(10).reset_index())

bplot3.set_title(label = 'Most Popular Games',  fontsize=20)
plt.figure(figsize=(15, 10))

df = game_data.groupby(["Publisher"])

bplot=sns.barplot("Global_Sales", "Publisher", 

            data =df.Global_Sales.sum().sort_values(ascending=False).head(20).reset_index())

bplot.set_title("Global Sales with respect to Publisher", fontsize=20)
plt.figure(figsize=(15, 10))

bplot1 = sns.barplot("Year","Global_Sales", data= game_data,estimator=np.sum, ci=None, 

                    saturation=1, palette="colorblind")

plt.xticks(rotation=90)

bplot1.set_title("Year Wise Global Sales", fontsize=20)
plt.figure(figsize=(15, 10))

sum_sales = game_data[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]].sum().reset_index()

sum_sales.columns = ["Region","Total_Sales"]

sns.barplot(data=sum_sales,x="Region", y="Total_Sales")
plt.figure(figsize=(15, 10))

cplot = sns.countplot(x = "Genre", data = game_data, palette="dark")

cplot.set_title(label = 'Count of Each Genre',  fontsize=20)
plt.figure(figsize=(15, 10))

bplot4 = sns.barplot(x="Genre", y="Global_Sales", data = game_data, ci=None, palette="pastel")

bplot4.set_title(label = 'Average Global_Sales Genre wise',  fontsize=20)
plt.figure(figsize=(15, 10))

sctplot=sns.scatterplot(x="Genre", y="NA_Sales", data=game_data)

sctplot.set_title(label = 'NA_Sales for each Genre',  fontsize=20)
plt.figure(figsize=(15, 10))

strplot=sns.stripplot(x="Genre", y="EU_Sales", data=game_data)

strplot.set_title(label = 'EU_Sales for each Genre',  fontsize=20)
plt.figure(figsize=(15, 10))

pntplot=sns.pointplot(x="Genre", y="JP_Sales", data=game_data)

pntplot.set_title(label = 'JP_Sales for each Genre',  fontsize=20)
ploting_fig = plt.figure(figsize=(20, 10))

adsubplt = ploting_fig.add_subplot(111, projection='3d')

adsubplt.scatter(game_data['NA_Sales'],game_data['EU_Sales'],game_data['JP_Sales'])

adsubplt.set_xlabel('NA_Sales')

adsubplt.set_ylabel('EU_Sales')

adsubplt.set_zlabel('JP_Sales')

adsubplt.set_title(label = '3D Ploting for NA_Sales, JP_Sales and EU_Sales',  fontsize=20)

plt.show()
plt.figure(figsize=(20, 10))

bplot2 = sns.barplot(x='Platform', y='Global_Sales', data=game_data, ci=None, palette='deep')

bplot2.set_title(label = 'Most Favourite Plateforms',  fontsize=20)