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
path="/kaggle/input/avocado-prices/avocado.csv"

df=pd.read_csv(path)

df.head()#for first 5 dataset
df.tail() # for last 5 datasets
df.info()
df.describe()
#averge of price

df["AveragePrice"].mean()
df["type"].value_counts()
df["region"].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt
df.hist(bins=50,figsize=(25,15))

plt.show()
#max average price of new york

df[df["region"]=="NewYork"]["AveragePrice"].max()
# averge price of Newyork

df[df["region"]=="NewYork"]["AveragePrice"].mean()
# Max average price from all regions

df[df["AveragePrice"]==df["AveragePrice"]].max()
# Min average price from all regions

df[df["AveragePrice"]==df["AveragePrice"]].min()
df[df["region"]=="SanFrancisco"].max()
plt.figure(figsize=(15,9))

plt.scatter(x=df["region"],y=df["AveragePrice"])

plt.xticks(rotation=90)

plt.show()
# define category wise 

plt.figure(figsize=(19,10))

sns.scatterplot(x=df["region"],y=df["AveragePrice"],hue=df["type"])

plt.xticks(rotation=90)

plt.show()
# Average Price per year

sns.jointplot(x=df["year"], y=df["AveragePrice"],data=df,kind="hex", color="#4CB396")

plt.show()