import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sales = pd.read_csv("/kaggle/input/sales-data/SalesData.csv")
sales.head()
region_sales = sales.groupby("Region")["Sales2015","Sales2016"].sum()
region_sales
plt.figure(figsize=(12,6))

region_sales.plot(kind="bar",figsize=(8,8))

plt.ylabel('Sales')

plt.title("Sales by region for 2016 with 2015")

plt.show()
print("We can conclude that sales in 2016 is more as compared to sales in 2015 in all regions. East region has contributed the maximum.")
sales.head()
sales_pie = sales.groupby("Region")["Sales2016"].sum()
sales_pie
plt.pie(sales_pie,autopct="%1.0f%%",labels=["Central","East","West"],shadow=True,explode=[0.0,0.1,0.0],colors=['r', 'g', 'b'])

plt.title("Contributing factors to the sales for each region in 2016")

plt.show()
print("East region has contributed the maximum in 2016.")
sales_region_tier = sales.groupby(["Region","Tier"])['Sales2015','Sales2016'].sum()

sales_region_tier
sales_region_tier.plot(kind="bar",figsize=(8,8))

plt.ylabel("Sales")

plt.title("Total sales of 2015 and 2016 with respect to Region and Tiers")

plt.show()
#stacked bar chart

sales_region_tier.plot(kind="barh",stacked=True,figsize=(8,8))

plt.xlabel("Sales")

plt.ylabel("Region/Tier")

plt.title("Total sales of 2015 and 2016 with respect to Region and Tiers")

plt.show()
print("We can conclude that East region and High tier in 2016 have contributed the maximum. Also sales in 2015 and sales in 2016 both is maximum in East region and High tier")
sales.head()
#grouping the data based on Region and State to find the total sales in 2015 and 2016

sales_state = sales.groupby(['Region',"State"])['Sales2015','Sales2016'].sum()
sales_state
#filtering out sales for East region in 2015 and 2016 

sales_east = sales_state.loc["East"]
sales_east
sales_east.plot(kind="bar",figsize=(12,8),width=0.8)

plt.ylabel("Sales")

plt.title("Sales comparison between 2015 and 2016 for East Region")

plt.show()
print("NY state registered a decline in sales in 2016 as compared to 2015")
sales.head()
#grouping the data based on tier and division to find the total sum of sales in 2015 and 2016

sales_division_tier =sales.groupby(["Tier","Division"])["Units2015","Units2016"].sum()
sales_division_tier
high_tier = sales_division_tier.loc["High"]
high_tier
high_tier.plot(kind="bar",figsize=(12,8),width=0.7)
print("No division show decline in number of units sold in 2016 compared to 2015")
sales.head(2)
month =sales["Month"]
quarter = []

for x in month :

    if x in ["Jan","Feb","Mar"]:

        quarter.append("Q1")

    elif x in ["Apr","May","Jun"]:

        quarter.append("Q2")

    elif x in ["Jul","Aug","Sep"]:

        quarter.append("Q3")

    else:

        quarter.append("Q4")
quarter
#create a new column "Qtr"

sales["Qtr"]= pd.Series(quarter)
sales.head(2)
#grouping data based on "Qtr" to find total sales in 2015 and 2016

qtr_sales = sales.groupby("Qtr")["Sales2015","Sales2016"].sum()
qtr_sales
qtr_sales.plot(kind="bar")

plt.ylabel("Sales")

plt.title("Quarter wise sales in 2015 and 2016")

plt.show()
sales.head()
#grouping the data based on "Qtr" and "Tier" to find total sales in 2016

qtr_pivot = sales.pivot_table(index='Qtr',columns='Tier',values='Sales2016')
qtr_pivot
#Qtr Q1

plt.pie(x=qtr_pivot.loc["Q1",:],autopct="%1.0f%%",labels=["High","Low","Med","Out"],colors=['deepskyblue', 'darkorange', 'darkgreen'])

plt.show()
#Qtr Q2

plt.pie(x=qtr_pivot.loc["Q2",:],autopct="%1.0f%%",labels=["High","Low","Med","Out"],colors=['deepskyblue', 'darkorange', 'darkgreen'])

plt.show()
#Qtr Q3

plt.pie(x=qtr_pivot.loc["Q3",:],autopct="%1.0f%%",labels=["High","Low","Med","Out"],colors=['deepskyblue', 'darkorange', 'darkgreen'])

plt.show()
#Qtr Q4

plt.pie(x=qtr_pivot.loc["Q4",:],autopct="%1.0f%%",labels=["High","Low","Med","Out"],colors=['deepskyblue', 'darkorange', 'darkgreen'])

plt.show()