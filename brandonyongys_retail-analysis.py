# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data visualization

import seaborn as sns # another data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
annual_sales_data = pd.read_csv("/kaggle/input/retail-business-sales-20172019/business.retailsales2.csv")

order_sales_data = pd.read_csv("/kaggle/input/retail-business-sales-20172019/business.retailsales.csv")
annual_sales_data.head()
order_sales_data.head()
annual_sales_data["Avg order gross sales"] = annual_sales_data["Gross Sales"]/annual_sales_data["Total Orders"]

annual_sales_data["Avg order total sales"] = annual_sales_data["Total Sales"]/annual_sales_data["Total Orders"]

annual_sales_data["Avg order discounts"] = -annual_sales_data["Discounts"]/annual_sales_data["Total Orders"]

annual_sales_data["Avg order returns"] = -annual_sales_data["Returns"]/annual_sales_data["Total Orders"]

annual_sales_data["Avg order shipping"] = annual_sales_data["Shipping"]/annual_sales_data["Total Orders"]

annual_sales_data["Avg order returns and discounts"] = -(annual_sales_data["Returns"] + annual_sales_data["Discounts"])/annual_sales_data["Total Orders"]
annual_sales_data.head()

# Total orders = monthly total orders

# Gross sales = monthly total sales from the total orders, excluding discounts and returns 

# Net sales = gross sales - (discounts + returns)

# Total sales = net sales + shipping (I assume that shipping cost is paid by the customers)
fig, ax = plt.subplots(3,1, figsize = (16,10), sharex = "all")

sns.lineplot(x = "Month", y = "Total Orders", data = annual_sales_data, hue = "Year", ax = ax[0], palette = "bright", sort = False)

ax[0].set_title("Monthly total orders")



sns.barplot(x = "Month", y = "Gross Sales", data = annual_sales_data, hue = "Year", ax = ax[1])

ax[1].set_title("Monthly gross sales")



sns.barplot(x = "Month", y = "Total Sales", data = annual_sales_data, hue = "Year", ax = ax[2])

ax[2].set_title("Monthly total sales")

plt.xticks(rotation=45)
fig, ax = plt.subplots(5,1, figsize = (10,15), sharex = "all")

sns.lineplot(x = "Month", y = "Avg order gross sales", data = annual_sales_data, hue = "Year", ax = ax[0], palette = "bright", sort = False)

ax[0].set_title("Monthly average gross sales per order")

ax[0].set_ylim(0)



sns.lineplot(x = "Month", y = "Avg order returns", data = annual_sales_data, hue = "Year", ax = ax[1], palette = "bright", sort = False)

ax[1].set_title("Monthly average returns per order")



sns.lineplot(x = "Month", y = "Avg order discounts", data = annual_sales_data, hue = "Year", ax = ax[2], palette = "bright", sort = False)

ax[2].set_title("Monthly average discounts per order")



sns.lineplot(x = "Month", y = "Avg order shipping", data = annual_sales_data, hue = "Year", ax = ax[3], palette = "bright", sort = False)

ax[3].set_title("Monthly average shipping per order")

ax[3].set_ylim(0)



#sns.lineplot(x = "Month", y = "Avg order returns and discounts", data = annual_sales_data, hue = "Year", ax = ax[2], palette = "bright", sort = False)

#ax[2].set_title("Monthly average returns and discounts per order")

#ax[2].set_ylim(0)



sns.lineplot(x = "Month", y = "Avg order total sales", data = annual_sales_data, hue = "Year", ax = ax[4], palette = "bright", sort = False)

ax[4].set_title("Monthly average total sales per order")

ax[4].set_ylim(0)



plt.xticks(rotation=45)
pivot_1 = order_sales_data.groupby(["Product Type"])[["Total Net Sales"]].agg("sum")

pivot_1["Sales proportions"] = (pivot_1["Total Net Sales"]/pivot_1["Total Net Sales"].sum())

pivot_1.sort_values("Sales proportions", ascending = False, inplace = True)

pivot_1["Cumulative sales proportions"] = pivot_1["Sales proportions"].cumsum().apply(lambda x: "%.2f" % round(100*x,2))



pivot_2 = order_sales_data.groupby(["Product Type"])[["Total Net Sales"]].agg("mean")

pivot_2["Total Net Sales"] = pivot_2["Total Net Sales"].apply(lambda x: "%.2f" % round(x,2))

pivot_2.rename(columns = {"Total Net Sales":"Average sales per order"}, inplace = True)



pivot_3 = order_sales_data.groupby(["Product Type"])[["Total Net Sales"]].agg("count")

pivot_3.rename(columns = {"Total Net Sales":"No of orders"}, inplace = True)



pivot_1 = pivot_1.join(pivot_2)

pivot_1 = pivot_1.join(pivot_3)

pivot_1
plt.figure(figsize=(20, 6))

sns.boxplot(y = "Net Quantity", data = order_sales_data, x = "Product Type")

plt.xticks(rotation = 45)
pivot_4 = order_sales_data.groupby(["Product Type"])[["Net Quantity"]].agg(["count","sum", "median", "mean"])

pivot_4[("Net Quantity", "mean")] = pivot_4[("Net Quantity", "mean")].apply(lambda x: int(x))



pivot_4
order_sales_data["Discounts proportions"] = (order_sales_data["Discounts"].apply(lambda x: abs(x))/order_sales_data["Gross Sales"]).apply(lambda x: round(100*x, 2))

order_sales_data["Returns proportions"] = (order_sales_data["Returns"].apply(lambda x: abs(x))/order_sales_data["Gross Sales"]).apply(lambda x: round(100*x, 2))
plt.figure(figsize=(20, 6))

sns.stripplot(x='Product Type', y='Discounts proportions', data = order_sales_data, jitter=True, split=True)

plt.xticks(rotation = 45)

plt.title("Distribution of discounts proportions for each product type")
plt.figure(figsize=(20, 6))

sns.stripplot(x='Product Type', y='Returns proportions', data = order_sales_data, jitter=True, split=True)

plt.xticks(rotation = 45)

plt.title("Distribution of returns proportions for each product type")
order_sales_data
product_type = ["Art & Sculpture", "Basket", "Home Decor", "Jewelry","Kitchen"]

for product in product_type:

    temp = order_sales_data[order_sales_data["Product Type"] == product]

    plt.figure(figsize = (5,5))

    sns.scatterplot(x = "Discounts proportions", y = "Returns proportions", data = temp)

    plt.title("Returns vs discounts rate for %s" %product)