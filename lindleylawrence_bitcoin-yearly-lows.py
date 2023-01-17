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
df = pd.read_csv(os.path.join(dirname, filename))



df
df = df[["Date","Price"]]

df
df['Year'] = pd.to_datetime(df['Date']).dt.to_period('Y')

df
df["Year"] = df["Year"].astype(str)

year_list = df["Year"].unique().tolist()



year_list
print("Bitcoin Yearly Lows")

print("Year","Price(USD)")



price_list = []



for year in year_list:

    df_year = df.loc[df["Year"] == year]

    price = df_year["Price"].min()

    price_list.append(price)

    print(year,price)

    

# df.loc[df["Year"] == year_list[0]]
yearly_low_df = pd.DataFrame()

yearly_low_df["Year"] = year_list

yearly_low_df["Price"] = price_list



yearly_low_df.set_index("Year")

yearly_low_df = yearly_low_df.sort_values(by="Year")

yearly_low_df
import matplotlib.pyplot as plt

import matplotlib as mpl





yearly_low_df.plot(kind="bar",x="Year",y="Price",figsize=(12,6))

plt.xlabel("Year")



plt.xticks(rotation=45)

plt.ylabel("USD/btc price")

plt.title("Bitcoin yearly lows")



    

plt.style.use(['ggplot'])