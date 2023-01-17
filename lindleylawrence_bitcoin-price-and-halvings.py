# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pyplot



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(os.path.join(dirname, filename))

df.info()
df = df[["Date","Price"]]

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(by="Date",ascending=True)

df

# df.loc[df["Date"] == "2020-05-20"]
# Bitcoin 12 months after the 1st halvening

# Having dates

# 1st = November 28, 2012

# 2nd = July 9, 2016 

# 3rd = May 11, 2020



from datetime import datetime





df = pd.read_csv(os.path.join(dirname, filename),index_col= 0,parse_dates = True)

df = df[["Price"]]

df = df.sort_values(by="Date",ascending=True)



# 12 months after 1st halvening

fig = plt.figure()

ax = fig.add_subplot(1,1,1,)

first_half = df["Price"]



first_half.plot(figsize=(15,10),ax=ax,style = "k-")



# Create annotations

halvening_data = [

    (datetime(2012, 11, 28), 'Price: U$12.40'),

    (datetime(2013, 11, 29), 'Price: U$1,079.90')

]



# Set annotations

for date, label in halvening_data:

    ax.annotate(label, xy=(date, first_half.asof(date) + 20),xytext=(date, first_half.asof(date) + 55),

                arrowprops=dict(facecolor='blue', headwidth=10, width=5,headlength=4),

                horizontalalignment='right', verticalalignment='bottom')



# Zoom in on 2010-2013

ax.set_xlim(['10/10/2010', '29/11/2013'])



ax.axvline(x = "2012-11-28",color="orange")

ax.set_ylim([0, 1300])



ax.set_title("BTC price chart : 12 months after 1st halvening")

plt.style.use(['seaborn'])

plt.show()







amount_halvening = df.loc[df.index == "2012-11-29"]

amount_halvening = amount_halvening["Price"][0]

amount_halvening



year_after_halvening = df.loc[df.index == "29/11/2013"]

year_after_halvening = year_after_halvening["Price"][0]

year_after_halvening



price_difference = year_after_halvening - amount_halvening

price_increase_times = round(price_difference / amount_halvening,2)



print("Price at halvening: U$",amount_halvening)

print("Price 12 months after halvening: U$",year_after_halvening)

print("Difference in price: U$",price_difference)

print("The price increased", price_increase_times, "times")

# 12 months after 2nd halvening

fig = plt.figure()

ax = fig.add_subplot(1,1,1,)

second_half = df["Price"]



second_half.plot(figsize=(15,10),ax=ax,style = "k-")



# Create annotations

halvening_data = [

    (datetime(2016, 7, 9), 'Price: U$651.80'),

    (datetime(2017, 7, 9), 'Price: U$2,511.40')

]



# Set annotations

for date, label in halvening_data:

    ax.annotate(label, xy=(date, second_half.asof(date) + 75),xytext=(date, second_half.asof(date) + 825),

                arrowprops=dict(facecolor='blue', headwidth=10, width=5,headlength=4),

                horizontalalignment='left', verticalalignment='top')



# Zoom in on 2010-2013

ax.set_xlim(['29/11/2013', '09/07/2017'])



ax.axvline(x = "2016-07-09",color="orange")

ax.set_ylim([100, 5000])



ax.set_title("BTC price chart : 12 months after 2nd halvening")



plt.show()





plt.style.use(['seaborn'])

amount_halvening = df.loc[df.index == "2016-07-09"]

amount_halvening = amount_halvening["Price"][0]

amount_halvening



year_after_halvening = df.loc[df.index == "09/07/2017"]

year_after_halvening = year_after_halvening["Price"][0]

year_after_halvening



price_difference = year_after_halvening - amount_halvening

price_increase_times = round(price_difference / amount_halvening,2)



print("Price at halvening: U$",amount_halvening)

print("Price 12 months after halvening: U$",year_after_halvening)

print("Difference in price: U$",price_difference)

print("The price increased", price_increase_times, "times")
# Period after 3rd halvening

fig = plt.figure()

ax = fig.add_subplot(1,1,1,)

third_half = df["Price"]





# Price at 30 Sep 2020

third_half.plot(figsize=(15,10),ax=ax,style = "k-")

# third_half["current_price"].plot(color="g")



# Create annotations

halvening_data = [

    (datetime(2020, 5, 11), 'Price: U$9,512.30'),

    (datetime(2020, 9, 4), 'Price: U$10,277.90')

]



# Set annotations

for date, label in halvening_data:

    ax.annotate(label, xy=(date, third_half.asof(date) + 475),xytext=(date, third_half.asof(date) + 4125),

                arrowprops=dict(facecolor='blue', headwidth=10, width=5,headlength=4),

                horizontalalignment='left', verticalalignment='top')



# Zoom in on 2010-2013

ax.set_xlim(['09/07/2017', '09/04/2020'])



ax.axvline(x = "2020-05-11",color="orange")

ax.set_ylim([3000, 20000])



ax.set_title("BTC price chart : Period after 3rd halvening")



plt.show()





plt.style.use(['seaborn'])

amount_halvening = df.loc[df.index == "2020-05-11"]

amount_halvening = amount_halvening["Price"][0]

amount_halvening



year_after_halvening = df.loc[df.index == "09/04/2020"]

year_after_halvening = year_after_halvening["Price"][0]

year_after_halvening



price_difference = year_after_halvening - amount_halvening

price_increase_times = round(price_difference / amount_halvening,2)



print("Price at halvening: U$",amount_halvening)

print("Current price 4 months after halvening: U$",year_after_halvening)

print("Difference in price: U$",price_difference)

print("The price increased", price_increase_times, "times")