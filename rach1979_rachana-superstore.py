%reset -f 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn as sns
path ="../input"

os.chdir(path)

sp = pd.read_csv("superstore_dataset2011-2015.csv",encoding="ISO-8859-1")
x = sp.sort_values('Profit', ascending=False)

top20 = x.head(20)

top20[['Customer Name', 'Profit']] 
sns.barplot(x = "Customer Name", y= "Profit", data=top20)  # plotting of top 20 profitable customers
sns.countplot("Segment", data = sp)           #Distribution of custome Segment
sp.groupby('Customer ID').apply(lambda x: pd.Series(dict(store_visit=x.shape[0]))).reset_index()
sp1 = sp.groupby('Customer ID').apply(lambda x: pd.Series(dict(store_visit=x.shape[0]))).reset_index()

sp1.loc[sp1.store_visit == 1, ['Customer ID', 'store_visit']]     # Customers who have visited the store only once
sns.boxplot("Order Priority","Profit",data= sp)    # relationship of Order Priority and Profitability : 

                                                   # Profits slightly higher when Order priority is Medium
sns.countplot("Market",data = sp)                                # distribution of customers marketwise
sns.countplot("Region", hue= "Market", data = sp)                 # distribution of customers regionwise, marketwise