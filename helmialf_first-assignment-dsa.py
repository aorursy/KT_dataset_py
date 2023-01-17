# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/used_car_data.csv")
print(df.head())
def get_first_word(string):
    if string is np.nan or string == None:
        return np.nan
    str_split = string.split(" ")
    if str_split[0].strip() == "null":
        return np.nan
    return str_split[0].strip()

def encode_owner_type(string):
    if string == "First":
        return 1
    if string == "Second":
        return 2
    if string == "Third":
        return 3
    if string == "Fourth & Above":
        return 4
    

df["Merk"] = df.apply(lambda row: get_first_word(row["Name"]), axis=1)
df["Engine"] = df.apply(lambda row: get_first_word(row["Engine"]), axis=1)
df["Power"] = df.apply(lambda row: get_first_word(row["Power"]), axis=1)
df["Owner_Type"] = df["Owner_Type"] = df.apply(lambda row: encode_owner_type(row["Owner_Type"]), axis=1)

print(df[["Merk", "Engine", "Power", "Owner_Type"]])
#soal 1
print(df["Merk"].value_counts())
# soal 2
print(df["Location"].mode())
# soal 3
df_year_freq = df["Year"].value_counts().rename_axis('Year').reset_index(name='Freq')
df_year_freq = df_year_freq.sort_values(by ='Year' , ascending=True)

print(df_year_freq)
plt.plot(df_year_freq["Year"], df_year_freq["Freq"])
plt.show()

# soal 4
print(len(df[df["Kilometers_Driven"] < 100000].index)) #count how many row 
# soal 5

km_mean = df["Kilometers_Driven"].mean()
km_std = df["Kilometers_Driven"].std()
print(km_mean)
print(km_std)
# soal 6

km_driven = df["Kilometers_Driven"]
iqr = km_driven.quantile(.75) - km_driven.quantile(.25)
lower = km_driven.quantile(.25)- 1.5*iqr
upper = km_driven.quantile(.75) + 1.5*iqr
print("Before outlier removal : ", km_driven.count())
km_driven = km_driven[km_driven.between(lower, upper)]
print("After outlier removal : ", km_driven.count())

#outlier removal from original df
df = df[df["Kilometers_Driven"].between(lower, upper)]
# soal 7
df_year_km = df[["Year", "Kilometers_Driven"]]
km_year_corr = df_year_km.corr(method="pearson")
print(km_year_corr)
# soal 8

print(df["Owner_Type"].value_counts())

# soal 9
# ASUMSI 1 kg = 1 liter
df["Mileage"] = df.apply(lambda row: get_first_word(row["Mileage"]), axis=1)
df["Mileage"] = df["Mileage"].astype(float)
temp = df[["Fuel_Type", "Mileage"]]
print(temp.groupby(["Fuel_Type"]).mean())
# soal 10
df_soal_10 = df[["Year", "Kilometers_Driven", "Owner_Type", "Mileage", "Engine" ,"Power","Seats", "Price"]]
df_soal_10["Engine"] = df_soal_10["Engine"].astype(float)
df_soal_10["Power"] = df_soal_10["Power"].astype(float)
df_soal_10 = df_soal_10.dropna()
corr = df_soal_10.corr(method="pearson")

plt.figure(figsize=(18,18))
sns.heatmap(corr,annot=True,cmap='RdYlGn')

plt.show()
print(df.skew())
# soal 3
f, ax = plt.subplots(figsize=(15,8))
sns.distplot(df['Year'])
plt.xlim()
plt.figure(figsize=(20, 10))
sns.catplot(x="Merk", kind="count", palette="ch:.25", height=8, aspect=2, data=df);
plt.xticks(rotation=90);
df = df[df["Fuel_Type"] != "Electric"]

var = 'Fuel_Type'
data = pd.concat([df['Price'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x=var, y="Price", data=data)
fig.axis(ymin=0, ymax=165);
var = 'Year'
data = pd.concat([df['Price'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 10))
fig = sns.boxplot(x=var, y="Price", data=data)
fig.axis(ymin=0, ymax=165);
plt.xticks(rotation=90);