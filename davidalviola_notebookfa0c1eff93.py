# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/wearables/DataChallenge_Wearables - Wearables.csv')

data

# What is the mean price of all wearable devices?

import statistics



# Create list of all prices

raw_prices = data["Price"]



# Remove values that are not numbers

clean_prices = raw_prices.dropna()



# Format output to 2 decimal places since it should be a dollar amount

print(format(statistics.mean(clean_prices), ".2f"))
# What is the max price of devices on the wrist?



# Filter by Body.Location

wrist_devices = data[(data["Body.Location"] == 'Wrist')]



print(format(max(wrist_devices["Price"]), ".2f"))
# What is the median price for the fitness category?



# Filter by category

fitness_devices = data[(data["Category"] == "Fitness")]



# Remove non-numerical rows

raw_price = fitness_devices["Price"]

clean_price = raw_price.dropna()



print(format(statistics.median(clean_price), ".2f"))
# Which category has the highest number of devices and how many does it have?

from collections import Counter



Counter(data["Category"])
# Create a histogram that shows distribution of price

import matplotlib.pyplot as plot

plot.figure(figsize=(10, 10))

plot.xlabel("Price ($)")

plot.ylabel("# of Devices")

plot.hist(data["Price"], bins = 20)
# Create a bar chart showing average price for each category. Which category has the highest average price?

import seaborn as sns



sns.catplot(data=data, x="Category", y="Price", kind="bar", height=10)