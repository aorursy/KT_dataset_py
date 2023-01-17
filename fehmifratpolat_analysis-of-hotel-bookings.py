# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
df.head()
df.info()
df.isnull().sum()
average_children = round(df["children"].mean())
df["children"] = df["children"].fillna(value=average_children)
df.isnull().sum()
df["country"].value_counts().head(10)
df["country"] = df["country"].fillna(value="PRT")
df.drop(["company"], axis = 1, inplace = True)

df.drop(["agent"], axis = 1, inplace = True)
df.isnull().sum()
sns.countplot(x="hotel", data=df)

plt.title("Hotel Type")

plt.xlabel("Types")

plt.ylabel("Number of Hotels")
sns.countplot(x="is_canceled", data = df)

plt.title("Cancellation")

plt.xlabel("Cancel, 0:No, 1:Yes")

plt.ylabel("Number of Cancellations")
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

plt.figure(figsize=(25,10))

sns.countplot(x= "total_nights", data = df)

plt.title('Total Nights Stayed')

plt.xlabel('Total Nights')

plt.ylabel('Number of Stays')
top_countries = df["country"].value_counts().nlargest(10).astype(int)
top_countries.index
plt.figure(figsize=(25,10))

sns.barplot(x=top_countries.index, y=top_countries, data=df)
plt.figure(figsize=(20,5))

grouped_month = df.groupby("arrival_date_month")["hotel"].count()

months = grouped_month.index



sns.barplot(x=months, y=grouped_month)

plt.xlabel("Months")

plt.ylabel("Number of guests")

plt.show()