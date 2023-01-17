# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



sales=pd.read_csv("../input/vgsales.csv")

sales.head()
sales.describe() 
sales.drop("Rank", axis=1,inplace=True)
sales.groupby("Genre").size()
sales.groupby("Genre").size().plot.pie(autopct="%1.1f%%", explode=(0.1,0,0,0,0,0,0,0,0,0,0.1,0), radius=2, startangle=90, shadow=True)
sales[["NA_Sales", "EU_Sales","JP_Sales","Other_Sales"]].sum().plot(kind='pie', autopct="%1.1f%%", explode=(0.1,0,0,0), startangle=270, radius=2)
sales.groupby("Year")["Global_Sales"].sum().plot(kind="line", grid=True, legend=True)
sales.groupby("Year")["NA_Sales"].sum().plot(kind="line", grid=True, legend=True)

sales.groupby("Year")["EU_Sales"].sum().plot(kind="line", grid=True, legend=True)

sales.groupby("Year")["JP_Sales"].sum().plot(kind="line", grid=True, legend=True)

sales.groupby("Year")["Other_Sales"].sum().plot(kind="line", grid=True, legend=True)
sales.groupby("Platform").size().plot(kind="bar")
print(sales.groupby("Publisher").size().idxmax())

print(sales.groupby("Publisher").size().max())
sns.lmplot(x="Global_Sales", y="NA_Sales", data=sales, fit_reg=True)

sns.lmplot(x="Global_Sales", y="EU_Sales", data=sales, fit_reg=True)

sns.lmplot(x="Global_Sales", y="JP_Sales", data=sales, fit_reg=True)