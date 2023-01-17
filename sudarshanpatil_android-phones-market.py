import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/android-devices-and-mobiles/mobiles1.csv")
df.head()
df.dtypes
sns.set()

plt.figure(figsize = (10,6))

sns.distplot(df["rating"], bins = 10)

plt.title("Rating Distribution", size = 20)

plt.xlabel("Ratings for the Android Phones")

plt.ylabel("Count")

plt.show()
battery_in_mah = df.battery.str[:4].astype(int)
sns.set()

plt.figure(figsize = (10,6))

sns.distplot(battery_in_mah, bins = 10)

plt.title("Distribution of Capacity of Battery in android phones", size = 20)

plt.xlabel("Battery amount for the Android Phones")

plt.ylabel("Count")

plt.show()
print("Average battery capacity of android phone is ", int(battery_in_mah.mean()), "MAH")
# Distinct display types in the android phones 

distinct_display_types = set([temp.split()[-2] for temp in df['display']])

distinct_display_types
df['type_of_display'] = [temp.split()[4] for temp in df['display']]
plt.figure(figsize= (8,8))

plt.title("Most occuring display type in android phones", size = 18)

sns.barplot(x ="index", y="type_of_display", data=df['type_of_display'].value_counts().reset_index()[:5], palette="Spectral")

plt.ylabel("Count of phone")

plt.xlabel("Categories of the display")

plt.show()
df["ram_in_gb"] = [i[0] for i in df.memory]

ram_in_gb =  df["ram_in_gb"].value_counts().reset_index().sort_values(by="ram_in_gb", ascending = False)
plt.figure(figsize= (8,8))

plt.title("Most occuring Ram size type in android phones", size = 18)

sns.barplot(x ="index", y="ram_in_gb", data=ram_in_gb, palette="inferno")

plt.ylabel("Count of phone")

plt.xlabel("RAM in GB")

plt.show()