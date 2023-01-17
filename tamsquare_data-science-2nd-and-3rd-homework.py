# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/vgsales.csv")
data.info()
data['Year'].value_counts(dropna =False)
data["Year"].fillna('Empty',inplace = True)
data["Publisher"].fillna('Empty',inplace = True)
data['Year'].value_counts(dropna =False)
data.info()
data.head()
data.tail()
data.describe()
data.dtypes
data.Genre.unique()
plt.figure(figsize=(20,20))
plt.subplot(4,1,1)
plt.scatter(data.Global_Sales,data.EU_Sales,color="Red",alpha=0.5,label="Europe Sales")
plt.legend(loc="lower right")
plt.subplot(4,1,2)
plt.scatter(data.Global_Sales,data.JP_Sales,color="Gold",alpha=0.5,label="Japan Sales")
plt.legend(loc="lower right")
plt.subplot(4,1,3)
plt.scatter(data.Global_Sales,data.NA_Sales,color="Blue",alpha=0.5,label="North America Sales")
plt.legend(loc="lower right")
plt.subplot(4,1,4)
plt.scatter(data.Global_Sales,data.Other_Sales,color="Green",alpha=0.5,label="Other Sales")
plt.legend(loc="lower right")
plt.xlabel("Global Sales of Video Games")
average_sales = sum(data.Global_Sales)/len(data.Global_Sales)
average_sales
data["Sales_Score"] = ["Good" if i > average_sales else "Bad" for i in data.Global_Sales]
data.loc[:10,["Sales_Score","Global_Sales"]]
data.loc[3753:3763,["Sales_Score","Global_Sales"]]

data.boxplot(column='Rank',by = 'Sales_Score',grid=False,figsize=(15,10))
plt.xlabel("Sales Score")
plt.ylabel("Video Game Sales Rankings in World")