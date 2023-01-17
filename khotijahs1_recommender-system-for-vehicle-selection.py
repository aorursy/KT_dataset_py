import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Data = pd.read_csv("../input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv")



Data.info()

Data[0:10]
cnt_pro = Data['name'].value_counts()  [:50]

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of Data', fontsize=12)

plt.xlabel('name', fontsize=9)

plt.xticks(rotation=90)

plt.show();
cnt_pro = Data['owner'].value_counts()

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of Data', fontsize=12)

plt.xlabel('owner', fontsize=12)

plt.xticks(rotation=80)

plt.show();
cnt_pro = Data['seller_type'].value_counts()

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of seller_type', fontsize=12)

plt.xlabel('seller_type', fontsize=12)

plt.xticks(rotation=80)

plt.show();
top_sell = Data.sort_values(by='selling_price', ascending=False)

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_sell.seller_type, x=top_sell.selling_price)

plt.xticks()

plt.xlabel('selling_price')

plt.ylabel('seller_type')

plt.title('The selling price of car')

plt.show()
top_sell = Data.sort_values(by='year', ascending=True)[:30]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=top_sell.selling_price, x=top_sell.year)

plt.xticks()

plt.xlabel('year')

plt.ylabel('selling_price')

plt.title('The selling price by vehicle year')

plt.show()
# here we are comparing the selling_price with name (vehicle)

# first group the name and get max,min and avg selling price of the vehicle

display(Data[["name","selling_price",]].groupby(["name"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="Blues"))

# here we are ploting these values using lineplot

Data[["name","selling_price",]].groupby(["name"]).agg(["max",'mean',"min"]).plot(kind="line",color =["red","black","blue"])

plt.title("Which car is most selling price?", fontsize=20)

plt.xticks(np.arange(17),['Maruti Swift Dzire VDI','Maruti Alto 800 LXI','Maruti Alto LXi','Hyundai EON Era Plus','Maruti Alto LX'],rotation=90,fontsize=15)

plt.ylabel("selling_price",fontsize=15)

plt.xlabel(" ")

plt.show()

print("History's Best Selling Vehicles")

display(Data.loc[Data.groupby(Data["name"])["selling_price"].idxmax()][["name",

                                                                  "selling_price"]].style.background_gradient(cmap="copper"))
Data['year'] = Data["year"].astype("int")

print('time series selling price cars')

display(Data[["year",'name','selling_price']].groupby(["name",

                                                         "year"]).agg("sum").sort_values(by="selling_price",

                                                          ascending = False).head(10).style.background_gradient(cmap='Greens'))

# here we are plotting them

sns.lineplot(Data["year"],Data['selling_price'],hue=Data["seller_type"])

plt.title("time series selling vehicles by seller_type",fontsize=20)

plt.xticks(fontsize=18)

plt.xlabel(" ")

plt.show()

display(Data[Data["owner"]=="First Owner"][["name","transmission","year","km_driven","fuel","seller_type",

                                       "selling_price"]].sort_values(by="selling_price", ascending= False).head(5).style.background_gradient(cmap="spring"))

display(Data[Data["owner"]=="Second Owner"][["name","transmission","year","km_driven","fuel","seller_type",

                                       "selling_price"]].sort_values(by="selling_price", ascending= False).head(5).style.background_gradient(cmap="spring"))



display(Data[Data["owner"]=="Third Owner"][["name","transmission","year","km_driven","fuel","seller_type",

                                       "selling_price"]].sort_values(by="selling_price", ascending= False).head(5).style.background_gradient(cmap="spring"))



display(Data[Data["owner"]=="Fourth & Above Owner"][["name","transmission","year","km_driven","fuel","seller_type",

                                       "selling_price"]].sort_values(by="selling_price", ascending= False).head(5).style.background_gradient(cmap="spring"))



display(Data[Data["owner"]=="Test Drive Car"][["name","transmission","year","km_driven","fuel","seller_type",

                                       "selling_price"]].sort_values(by="selling_price", ascending= False).head(5).style.background_gradient(cmap="spring"))





sns.barplot(Data["owner"],Data["km_driven"],hue= Data["seller_type"],palette="spring")

plt.xticks(rotation=80)

plt.title("seller_type : km_driven comparsion")
def recommend_vehicle(x):

    y = Data[["fuel",'name',"km_driven","transmission","selling_price"]][Data["fuel"] == x]

    y = y.sort_values(by="selling_price",ascending=False)

    return y.head(15)
recommend_vehicle("Diesel")
recommend_vehicle("Petrol")
recommend_vehicle("CNG")
recommend_vehicle("LPG")
recommend_vehicle("Electric")
def recommend_vehicle(x):

    y = Data[["owner",'name',"km_driven","transmission","selling_price"]][Data["owner"] == x]

    y = y.sort_values(by="selling_price",ascending=False)

    return y.head(15)
recommend_vehicle("First Owner")
recommend_vehicle("Second Owner")
recommend_vehicle("Third Owner")
recommend_vehicle("Fourth & Above Owner")
recommend_vehicle("Test Drive Car")
def recommend_vehicle(x):

    y = Data[["seller_type",'name',"km_driven","transmission","selling_price"]][Data["seller_type"] == x]

    y = y.sort_values(by="selling_price",ascending=False)

    return y.head(15)
recommend_vehicle("Individual")
recommend_vehicle("Dealer")
recommend_vehicle("Trustmark Dealer")