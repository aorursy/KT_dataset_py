# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%time data=pd.read_csv("/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv")

print(data.shape)
data.head()
data.isnull().sum()
data.info()
data.describe()
print("Dataset contains {} row and {} colums".format(data.shape[0],data.shape[1]))
plt.figure(figsize=(14,6))

plt.style.use('fivethirtyeight')

ax= sns.countplot('Gender', data=data , palette = 'copper')

ax.set_xlabel(xlabel= "Gender",fontsize=18)

ax.set_ylabel(ylabel = "Gender count", fontsize = 18)

ax.set_title(label = "Gender count in supermarket", fontsize = 20)

plt.show()
data.groupby(['Gender']). agg({'Total':'sum'})
plt.figure(figsize= (14,6))

ax = sns.countplot(x = "Customer type", data = data, palette = "rocket_r")

ax.set_title("Type of customers", fontsize = 25)

ax.set_xlabel("Customer type", fontsize = 16)

ax.set_ylabel("Customer Count", fontsize = 16)

data.groupby(['Customer type']). agg({'Total':'sum'})
plt.figure(figsize=(14,6))

ax = sns.countplot(x = "Customer type", hue = "Branch", data = data, palette= "rocket_r")

ax.set_title(label = "Customer type in different branch", fontsize = 25)

ax.set_xlabel(xlabel = "Branches", fontsize = 16)

ax.set_ylabel(ylabel = "Customer Count", fontsize = 16)
plt.figure(figsize = (14,6))

ax = sns.countplot(x = "Payment", data = data, palette = "tab20")

ax.set_title(label = "Payment methods of customers ", fontsize= 25)

ax.set_xlabel(xlabel = "Payment method", fontsize = 16)

ax.set_ylabel(ylabel = " Customer Count", fontsize = 16)

plt.figure(figsize = (14,6))

ax = sns.countplot(x="Payment", hue = "Branch", data = data, palette= "tab20")

ax.set_title(label = "Payment distribution in all branches", fontsize= 25)

ax.set_xlabel(xlabel = "Payment method", fontsize = 16)

ax.set_ylabel(ylabel = "Peple Count", fontsize = 16)
plt.figure(figsize=(14,6)) 

ax = sns.boxplot(x="Branch", y = "Rating" ,data =data, palette= "RdYlBu")

ax.set_title("Rating distribution between branches", fontsize = 25)

ax.set_xlabel(xlabel = "Branches", fontsize = 16)

ax.set_ylabel(ylabel = "Rating distribution", fontsize = 16)

data["Time"]= pd.to_datetime(data["Time"])
data["Hour"]= (data["Time"]).dt.hour
plt.figure(figsize=(14,6)) 

SalesTime = sns.lineplot(x="Hour", y ="Quantity", data = data).set_title("product sales per Hour")
plt.figure(figsize=(14,6)) 

rating_vs_sales = sns.lineplot(x="Total", y= "Rating", data=data)
plt.figure(figsize=(10,6)) 

ax = sns.boxenplot(x = "Quantity", y = "Product line", data = data,)

ax.set_title(label = "Average sales of different lines of products", fontsize = 25)

ax.set_xlabel(xlabel = "Qunatity Sales",fontsize = 16)

ax.set_ylabel(ylabel = "Product Line", fontsize = 16)
plt.figure(figsize=(14,6))

ax = sns.countplot(y='Product line', data=data, order = data['Product line'].value_counts().index)

ax.set_title(label = "Sales count of products", fontsize = 25)

ax.set_xlabel(xlabel = "Sales count", fontsize = 16)

ax.set_ylabel(ylabel= "Product Line", fontsize = 16)
plt.figure(figsize=(14,6))

ax = sns.boxenplot(y= "Product line", x= "Total", data = data)

ax.set_title(label = " Total sales of product", fontsize = 25)

ax.set_xlabel(xlabel = "Total sales", fontsize = 16)

ax.set_ylabel(ylabel = "Product Line", fontsize = 16)

plt.figure(figsize = (14,6))

ax = sns.boxenplot(y = "Product line", x = "Rating", data = data)

ax.set_title("Average rating of product line", fontsize = 25)

ax.set_xlabel("Rating", fontsize = 16)

ax.set_ylabel("Product line", fontsize = 16)
plt.figure(figsize = (14,6))

ax= sns.stripplot(y= "Product line", x = "Total", hue = "Gender", data = data)

ax.set_title(label = "Product sales on the basis of gender")

ax.set_xlabel(xlabel = " Total sales of products")

ax.set_ylabel(ylabel = "Product Line")
plt.figure(figsize = (14,6))

ax = sns.relplot(y= "Product line", x = "gross income", data = data)

# ax.set_title(label = "Products and Gross income")

# ax.set_xlabel(xlabel = "Total gross income")

# ax.set_ylabel(ylabel = "Product line")