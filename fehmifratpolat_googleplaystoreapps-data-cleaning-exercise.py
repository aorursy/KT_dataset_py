# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
df.head(3)
df.columns
df.info()
df.shape
def missing_value_of_data(df):
    total=df.isnull().sum().sort_values(ascending=False)
    percentage=round(total/df.shape[0]*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


missing_value_of_data(df)
df["App"].nunique()
df["App"].value_counts()
df[df["App"] == "ROBLOX"]
df.drop_duplicates(subset = "App", inplace = True)
df["App"].value_counts()
df.shape
df["Category"].value_counts()
df[df["Category"] == "1.9"]
df.at[10472, "Category"] = "PHOTOGRAPHY"
df.at[10472, "Installs"] = "1,000,000+"
df.at[10472, "Price"] = "0"
df.at[10472, "Last Updated"] = "July 20, 2018"
df.at[10472, "Current Ver"] = "1.0.19"
df.at[10472, "Android Ver"] = "4.0 and up"
df[df["App"] == "Life Made WI-Fi Touchscreen Photo Frame"]
df["Category"].value_counts()
df["Rating"].value_counts()
df[df["Rating"] == 19.0]
average_rating = df["Rating"].mean()
df.at[10472, "Rating"] = round(average_rating, 1)
df.loc[df["App"] == "Life Made WI-Fi Touchscreen Photo Frame"]
df["Rating"].isnull().sum()
df["Rating"] = df.groupby("Category").transform(lambda x: x.fillna(round(x.mean(),1)))
df["Reviews"].value_counts()
df["Type"].value_counts()
df.loc[df["Type"] == "0"]
df.at[10472, "Type"] = "Free"
df["Type"].value_counts()
df["Type"].isnull().sum()
df[df["Type"].isnull() == True]
df.at[9148, "Type"] = "Free"
labels = df["Type"].value_counts().index
sizes = df["Type"].value_counts()
explode = (0, 0.1)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=30)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Percentage of Free and Paid Apps")
plt.show()
df["Size"].value_counts()
df["Size"].isnull().sum()
df["Size"].unique()
def size_convert(i):
    if "M" in i:
        return float(i[:-1]) * 1000
    elif "K" in i:
        return float(i[:-1])
    else:
        return i
df["Size"] = df["Size"].apply(size_convert)
df["Installs"].isnull().sum()
df["Installs"].unique()
df["Installs"] = df["Installs"].apply(lambda x: x.replace(',',''))
df["Installs"] = df["Installs"].apply(lambda x: x.replace('+',''))
df["Installs"] = df["Installs"].apply(lambda x: int(x))
sorted_value = sorted(list(df["Installs"].unique()))
df["Installs"].replace(sorted_value,range(0,len(sorted_value),1), inplace = True )
plt.figure(figsize = (10,10))
sns.regplot(x = "Installs", y = "Rating", color = 'r',data=df);
plt.title('Rating vs Installs',size = 20)
df.info()
df["Content Rating"].isnull().sum()
df["Content Rating"].unique()
df[df["Content Rating"].isnull() == True]
df.at[10472, "Content Rating"] = "Everyone"
df["Genres"].unique()
df.loc[df["Genres"] == "February 11, 2018"]
df.at[10472, "Genres"] = "Photography"
df["Current Ver"].nunique()
df["Current Ver"].isnull().sum()
df["Current Ver"].fillna("1.0", inplace = True)
df["Android Ver"].unique()
df["Android Ver"].value_counts()
def and_version(i):
    if str(i) == "4.4W and up":
        return "4.4 and up"
    elif "-" in str(i):
        return str(i.split(" ")[0]) + " and up"
    else:
        return i

df["Android Ver"] = df["Android Ver"].apply(and_version)
df["Android Ver"].value_counts()
df.loc[df["Android Ver"].isnull() == True]
df.at[10472, "Price"] = "0"
df.at[10472, "Last Updated"] = "July 20, 2018"
df.at[10472, "Current Ver"] = "1.0.19"
df.at[10472, "Android Ver"] = "4.0 and up"
df["Android Ver"].fillna("4.0 and up", inplace = True)
df.info()
