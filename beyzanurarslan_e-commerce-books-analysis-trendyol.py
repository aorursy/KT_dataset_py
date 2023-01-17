#importing libraries
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import seaborn as sns
import matplotlib.pyplot as plt
#importing data
data= pd.read_excel('../input/booksdata/books2.xlsx')
data
data.info()
for i in range(data.shape[0]):
    data["Price"][i]= data["Price"][i][:-2].replace(",",".")
data["Price"]= data["Price"].astype(float)
data["Price"]
data.describe()
#correlation and visualization
print(data.corr())
sns.regplot(data["Price"], data["Seller Rank"])
plt.show()
#histograms and scatter plots
sns.pairplot(data)
plt.figure(figsize = (6, 4))
sns.boxplot(data["Price"])
#printing the outlier's features
from scipy import stats
z = np.abs(stats.zscore(data["Price"]))
data.loc[data["Price"][(z > 2)].index]
data.loc[data["Price"][(z < 2)].index]["Price"].describe()
data2= data[data.Seller.isin(data["Seller"].value_counts().index[:10])]
plt.figure(figsize = (14, 10))
ax = sns.boxplot(x="Seller", y="Price", data=data2)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)
sns.barplot(data["Author"].value_counts().index[:10],data["Author"].value_counts().values[:10] )
plt.xticks(rotation=90)
plt.grid(alpha=0.2)
plt.title("Authors' Rate")
plt.show()
sns.barplot(data["Publisher"].value_counts().index[:10],data["Publisher"].value_counts().values[:10] )
plt.xticks(rotation=90)
plt.grid(alpha=0.2)
plt.title("Publishers' Rate")
plt.show()
sns.barplot(data["Seller"].value_counts().index[:10],data["Seller"].value_counts().values[:10] )
plt.xticks(rotation=90)
plt.grid(alpha=0.2)
plt.title("Sellers' Rate")
plt.show()
#let's visualize it

a= list(data["Seller"].value_counts().values[:10] /data.shape[0])
a.append(1-np.sum(list(data["Seller"].value_counts().values[:10] /data.shape[0])))
b= list(data["Seller"].value_counts().index[:10])
b.append("Other")

plt.rcParams["figure.figsize"] = (25,15)
theme = plt.get_cmap('hsv')

explode = (0.1,0,0,0,0,0,0,0,0,0,0)  

fig1, ax1 = plt.subplots()
theme = plt.get_cmap('jet')
ax1.set_prop_cycle("color", [theme(1. * i / len(a)) for i in range(len(a))])

ax1.pie(a, explode=explode, labels=b, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 16})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Sellers' Total Rate", fontsize=20)
plt.show()