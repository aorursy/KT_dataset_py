# importing some useful libraries.

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
# loading and exploring the data

data = pd.read_csv("../input/candy-data.csv")

data.head()
data.info()
data.describe()
datax = data.iloc[:,1:7]
datax.sum(axis=0,numeric_only=True)
# Now Let's see it viusally

plt.figure(figsize=(20,16))
plt.suptitle("CORRELATION MAP", fontsize=18)
sns.heatmap(data.corr(), annot=True, fmt="0.2f", cmap="coolwarm", vmin=-1.0, vmax=1.0)
plt.tight_layout(pad=5.0)
plt.show()
plt.figure(figsize=(30,10))

plt.suptitle("Chocolate Plots", fontsize=20)

plt.subplot(1,3,1)
plt.title("Wining precentage with and without Chocolate",fontsize=16)
sns.violinplot(data=data,x="chocolate",y="winpercent",color="brown",palette="gray_r")
plt.xlabel("Has Chocolate ?")

plt.subplot(1,3,3)
plt.title("Total percentage of Candies with and without Chocolate",fontsize=16)
plt.pie(data.chocolate.value_counts(), autopct='%1.1f%%',colors=["#F7D8AD","#F7EDDF"],labels=["Doesn't have Chocolate","Does have Chocolate"],shadow=True)

plt.subplot(1,3,2)
plt.title("Price percentage with and without Chocolate",fontsize=16)
sns.boxplot(y="pricepercent", x="chocolate", data=data, palette="gray")
plt.xlabel("Has Chocolate ?")

plt.show()
plt.figure(figsize=(30,10))

plt.suptitle("Caramel Plots", fontsize=20)

plt.subplot(1,3,1)
plt.title("Wining precentage with and without Caramel",fontsize=16)
sns.violinplot(data=data,x="caramel",y="winpercent",color="brown",palette="YlOrRd")
plt.xlabel("Has Caramel ?")

plt.subplot(1,3,3)
plt.title("Total percentage of Candies with and without Caramel",fontsize=16)
plt.pie(data.caramel.value_counts(), autopct='%1.1f%%',colors=["#D8822F","#B35719"],labels=["Doesn't have caramel","Does have caramel"],shadow=True)

plt.subplot(1,3,2)
plt.title("Price percentage with and without Caramel",fontsize=16)
sns.boxplot(y="pricepercent", x="caramel", data=data, palette="YlOrRd")
plt.xlabel("Has Caramel ?")

plt.show()
plt.figure(figsize=(30,10))

plt.suptitle("Fruit Plots", fontsize=20)

plt.subplot(1,3,1)
plt.title("Wining precentage with and without Fruits",fontsize=16)
sns.violinplot(data=data,x="fruity",y="winpercent",color="brown",palette="husl")
plt.xlabel("Has Fruits ?")

plt.subplot(1,3,3)
plt.title("Total percentage of Candies with and without Fruits",fontsize=16)
plt.pie(data.fruity.value_counts(), autopct='%1.1f%%',colors=["#4ECDC4","#FF6B6B"],labels=["Doesn't have Fruits","Does have Fruits"],shadow=True)

plt.subplot(1,3,2)
plt.title("Price percentage with and without Fruits",fontsize=16)
sns.boxplot(y="pricepercent", x="fruity", data=data, palette="gist_stern_r")
plt.xlabel("Has Fruits ?")

plt.show()
plt.figure(figsize=(30,10))

plt.suptitle("Peanut Plots", fontsize=20)

plt.subplot(1,3,1)
plt.title("Wining precentage with and without Peanut",fontsize=16)
sns.violinplot(data=data,x="peanutyalmondy",y="winpercent",palette="YlOrBr_r")
plt.xlabel("Has Peanut ?")

plt.subplot(1,3,3)
plt.title("Total percentage of Candies with and without Peanut",fontsize=16)
plt.pie(data.peanutyalmondy.value_counts(), autopct='%1.1f%%',colors=["#D0B078","#9C6B40"],labels=["Doesn't have peanut","Does have peanut"],shadow=True)

plt.subplot(1,3,2)
plt.title("Price percentage with and without Peanut",fontsize=16)
sns.boxplot(y="pricepercent", x="peanutyalmondy", data=data, palette="YlOrBr")
plt.xlabel("Has Peanut ?")

plt.show()
plt.figure(figsize=(30,10))

plt.suptitle("Nougat Plots", fontsize=20)

plt.subplot(1,3,1)
plt.title("Wining precentage with and without Nougat",fontsize=16)
sns.violinplot(data=data,x="nougat",y="winpercent",palette="pink")
plt.xlabel("Has Nougat ?")

plt.subplot(1,3,3)
plt.title("Total percentage of Candies with and without Nougat",fontsize=16)
plt.pie(data.nougat.value_counts(), autopct='%1.1f%%',colors=["#5e5b52","#fcd757"],labels=["Doesn't have Nougat","Does have Nougat"],shadow=True)

plt.subplot(1,3,2)
plt.title("Price percentage with and without Nougat",fontsize=16)
sns.boxplot(y="pricepercent", x="nougat", data=data, palette="pink")
plt.xlabel("Has Nougat ?")

plt.show()
plt.figure(figsize=(30,10))

plt.suptitle("Crisped Rice or Wafer Plots", fontsize=20)

plt.subplot(1,3,1)
plt.title("Wining precentage with and without crispedricewafer",fontsize=16)
sns.violinplot(data=data,x="crispedricewafer",y="winpercent",palette="Spectral_r")
plt.xlabel("Has Crisped Rice or Wafer ?")

plt.subplot(1,3,3)
plt.title("Total percentage of Candies with and without Crisped Rice or Wafer",fontsize=16)
plt.pie(data.crispedricewafer.value_counts(), autopct='%1.1f%%',colors=["#52eb86","#f0c200"],labels=["Doesn't have \nCrisped Rice or Wafer","Does have \nCrisped Rice or Wafer"],shadow=True)

plt.subplot(1,3,2)
plt.title("Price percentage with and without Crisped Rice or Wafer",fontsize=16)
sns.boxplot(y="pricepercent", x="crispedricewafer", data=data, palette="Spectral_r")
plt.xlabel("Has Crisped Rice or Wafer ?")

plt.show()
# I think I need to eat some candies now.