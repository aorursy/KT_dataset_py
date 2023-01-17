# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
files = [file for file in os.listdir("/kaggle/input/sales-data-for-eda/Sales_Data")]
all_months_data = pd.DataFrame()
for file in files:
    df = pd.read_csv("/kaggle/input/sales-data-for-eda/Sales_Data/"+file)
    all_months_data = pd.concat([all_months_data, df])
all_months_data.to_csv("all_data.csv", index=False)
all_data = pd.read_csv("all_data.csv")
all_data.head()
all_data = all_data.dropna(how="all")
all_data = all_data[all_data["Order Date"].str[0:2] != "Or"]
#making a new column called month and cleaning it and making it's values intergers 
all_data["Month"] = all_data["Order Date"].str[0:2]
all_data["Month"].dropna()
all_data["Month"] = all_data["Month"].astype("int32")
#transforming values in Quantity Ordered and Price Each columns to a numeric
all_data["Quantity Ordered"] = pd.to_numeric(all_data["Quantity Ordered"])
all_data["Price Each"] = pd.to_numeric(all_data["Price Each"])

all_data.head()
all_data["Sales"] = all_data["Quantity Ordered"] * all_data["Price Each"]
all_data.head()
sale_month = all_data.groupby("Month")["Sales"].sum()
months = range(1,13)
plt.bar(months, sale_month)
plt.xticks(months)
plt.xlabel("Months")
plt.ylabel("Sales in million $")
plt.title("Sales for each month")
plt.legend()
plt.show()
#getting city and state from the Purchase Address column
def get_city(address):
    return address.split(",")[1]

def get_state(address):
    return address.split(",")[2][1:3]

all_data["City"] = all_data["Purchase Address"].apply(lambda x: get_city(x) + " " + get_state(x))

sales = all_data.groupby("City")["Sales"].sum()
sales.plot.bar()
plt.ylabel("Sales by million $")
sales
#converting Order Date from a string type into a datetime type column
all_data["Order Date"] = pd.to_datetime(all_data["Order Date"])
#creating an hour column and getting it from the Order Date column
all_data["Hour"] = all_data["Order Date"].dt.hour
#creating an Minute column and getting it from the Order Date column
all_data["Minute"] = all_data["Order Date"].dt.minute

#Ploting to see when do sales go up 
hours = [hour for hour, df in all_data.groupby("Hour")]
plt.plot(hours, all_data.groupby("Hour").count())
plt.xticks(hours)
plt.grid()
plt.show()
all_data.head()
df =  all_data.loc[all_data["Order ID"].duplicated(keep=False)]
df["Grouped"] = df.groupby("Order ID")["Product"].transform(lambda x: ", ".join(x))
df = df[["Order ID", "Grouped"]].drop_duplicates()
df.head()
from itertools import combinations
from collections import Counter

count = Counter()

for row in df["Grouped"]:
    row_list = row.split(", ")
    count.update(Counter(combinations(row_list, 2)))

count.most_common()
product_quantity = all_data.groupby("Product")["Quantity Ordered"].sum()
products = [product for product, df in all_data.groupby("Product")]
prices = all_data.groupby("Product")["Price Each"].mean()
fig, ax1 = plt.subplots()
ax2= ax1.twinx()
ax1.bar(products, product_quantity, color="g", alpha=0.5)
ax2.plot(products, prices, "b-")

ax1.set_xlabel("Product Names")
ax1.set_ylabel("Product Quantity", color="g")
ax2.set_ylabel("Prices", color="b")
ax1.set_xticklabels(products,rotation="vertical", size=10)
plt.show()