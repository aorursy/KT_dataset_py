import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
df.head()
df.shape #checking total no. of rows and columns in our data
df["Category"].unique() #checking total no.of unique categories
df["Rating"].unique()
df["Rating"].replace({19:5},inplace=True)
df["Rating"].unique()
filt=(df["Rating"]==5)

df[filt]#apps with  5 Ratings
df.head()
plt.figure(figsize=(8,4))

sns.distplot(df.Rating)
#from the above graph we can see that most of the graphs have rating near to 4.5
df.Installs=df.Installs.apply(lambda x: x.strip('+'))

df.Installs=df.Installs.apply(lambda x: x.replace(',',''))

df.Installs=df.Installs.replace('Free',np.nan)

#cleaning the datas of the install Column
df.Installs.dropna(inplace=True)
plt.figure(figsize=(8,4))

sns.set_style("darkgrid")

sns.set_context("poster")

sns.distplot(df["Installs"],color="Orange")
#Plot showing distribution of installs
plt.figure(figsize=(10,6))

sns.set_context("paper")

sns.countplot(df["Content Rating"])