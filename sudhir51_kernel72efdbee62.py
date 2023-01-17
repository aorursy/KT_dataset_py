# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





%matplotlib inline

%reload_ext autoreload

%autoreload 2
data = pd.read_csv("/kaggle/input/vikram-data2/productMonthlySaleData.csv")

data2 = pd.read_csv("/kaggle/input/main-data/MainCleanedData.csv")
data.head()
data = data.drop("Unnamed: 0", axis = 1)
data.head()
for col in data.columns:

    print(col, " Unique values :", data[col].unique())
#Ploting percentage sales by each store.

data["Store Codes"] = ["Store " + str(i) for i in data["Store Code"]]

f, ax = plt.subplots(figsize=(6, 6))

explode = (0, 0, 0, 0, 0, 0.05)

sale_count = data.groupby("Store Codes")["avg Product Sale"].sum().plot(kind='pie',startangle=100, explode= explode, shadow=True, colormap="RdBu", autopct='%1.1f%%', fontsize=12, legend = True, label = "")

_ = ax.axis('equal')

__ = ax.set_title("Portion of quantity sold by each store.", fontsize=15)

___ = ax.legend(title="Stores", loc="best", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)





#plt.setp(autotexts, size=8, weight="bold")



#Monthly Sales proportion of each months.

import calendar

monthly_sum = data.groupby("Sale Month")["avg Product Sale"].sum()

monthly_sum = monthly_sum.sort_index().to_dict()

monthly_data = {calendar.month_abbr[i]:v for i,v in monthly_sum.items()}

#data = data.sort_values("Sale Month")

f, ax = plt.subplots(figsize=(6, 6))

explode = [.04,.04, 0 ,0, 0, 0, 0, 0,.03, .04, .04, .09]

sale_count = pd.Series(monthly_data).plot(kind='pie',startangle=126, explode= explode, shadow=True, colormap="viridis_r", autopct='%1.1f%%', fontsize=11, legend = True, label = "")

_ = ax.axis('equal')

__ = ax.set_title("Proportion of total Sales by each Month.", fontsize=15)

___ = ax.legend(title="Stores", loc="best", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)



ax = data.groupby(["Sale Month","Store Codes"])["avg Product Sale"].sum().unstack('Store Codes').plot(kind='barh',title ="Monthly sales share of each store.", stacked=True, colormap="gnuplot_r",figsize=(10, 8))

ax.set_xlabel("Average Product Sale")
dd
data2.head()
data2["Store Codes"] = ["Store " + str(i) for i in data2["Store Code"]]

barp = data2.groupby(["Sale Week","Store Codes"])["Total Sales Amount"].sum().unstack('Store Codes').plot(kind='line', title ="Weekly sales of each store.",figsize=(20, 8), style=['+-','o-','.--','s:', '.-', '--'],markerfacecolor='black')

barp.set_ylabel("Total Sales Amount")