import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Read csv
raw_data = pd.read_csv("../input/Lemonade.csv")
avg_sales = raw_data["Sales"].mean()
print("Average in sales : " + str(avg_sales))
below_avg = raw_data.loc[lambda x: x["Sales"] < avg_sales]
print(below_avg)
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

ax.set_xlabel("Temperature")
ax.set_ylabel("Sales")

plt.scatter(raw_data["Temperature"],raw_data["Sales"])

plt.show()
Sales_Day = raw_data.drop(columns=["Temperature", "Rainfall", "Flyers", "Price", "Date"])

Sales_Day['Day'] = pd.Categorical(Sales_Day['Day'], categories=["Sunday", "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"], ordered=True)
Ordered_Sales_Day = Sales_Day.sort_index()

g_by_day = Ordered_Sales_Day.groupby(["Day"]).mean()

print(g_by_day)

x = g_by_day.index
y = g_by_day["Sales"]

import math

low = min(y)-1
high = max(y)
plt.ylim([math.ceil(low-0.7*(high-low)), math.ceil(high+0.7*(high-low))])

plt.bar(x,y,align='center') # A bar chart
plt.xlabel('Day')
plt.ylabel('Sales')

plt.show()
