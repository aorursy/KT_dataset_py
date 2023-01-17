# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import math 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings("ignore")



from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/pubg-weapon-stats/pubg-weapon-stats.csv")
df.head()
df.info()
kolon = []

for i in df.columns:

   i = i.replace(" ", "_")

   kolon.append(i.replace("(","").strip(")"))

                

df.columns = kolon

df.columns
kategorik_degisken = df.select_dtypes(include=["object"]).columns

for i in kategorik_degisken:

    df[i] = pd.Categorical(df[i])

df.dtypes
df.columns
silah_isimleri = pd.DataFrame(df.Weapon_Name.unique(), columns=["Silah_İsimleri"])

silah_isimleri
df.pivot_table( ["Bullet_Type","Damage", "Damage_Per_Second","Shots_to_Kill_Head","Shots_to_Kill_Chest"] , index = ["Weapon_Type" , "Weapon_Name"])
df.describe().T
df.columns[df.isnull().any()]
df.isnull().sum()
for i in df.columns:

    if ("BDMG" in i) | ("HDMG" in i):

        df[i].fillna(0, inplace=True)

df.isnull().sum()
df.Range.mean()
df.Range.median()
df.Range.fillna(df.Range.mean(), inplace=True)

df.isnull().sum()
df.Bullet_Speed.mean()
df.Bullet_Speed.median()
df.groupby("Weapon_Type").aggregate({"Bullet_Speed": [np.mean, np.median]})
gruplanan_hiz_sozlugu =dict(df.groupby("Weapon_Type")["Bullet_Speed"].median())

for x,y in gruplanan_hiz_sozlugu.items():

    if math.isnan(float(y)) :

        gruplanan_hiz_sozlugu[x] = 0.0

        

df_bullet = df[df["Bullet_Speed"].isnull().values]

indeks = df_bullet.index.tolist()

deger = []



for i in df_bullet["Weapon_Type"] :

    deger.append(gruplanan_hiz_sozlugu[i])



df_bullet.drop("Bullet_Speed" , axis=1)

df_bullet["Bullet_Speed"] = deger

df[df["Bullet_Speed"].isnull().values] = df_bullet



df.isnull().sum()
df["Bullet_Type"].unique()
def box_plot(df , f1, f2):

        sns.boxplot(x=f1 , y=f2 , data = df)

        plt.show();
x_ekseni = ["Damage","Damage_Per_Second", "Magazine_Capacity", "Range", "Bullet_Speed", "Rate_of_Fire"]

y_ekseni = "Weapon_Type"

for i in x_ekseni:

    box_plot(df,i,y_ekseni)
sns.set()

plt.figure(figsize=(25,10))

sns.countplot(df["Weapon_Type"])

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(25,10))

sns.barplot(ax=axes[0], x="Weapon_Type", y="Shots_to_Kill_Head" ,data=df)

sns.barplot(ax=axes[1], x="Weapon_Type", y="Shots_to_Kill_Chest" , data=df);
df[["Bullet_Type" , "Damage"]].groupby(["Bullet_Type"] , as_index=False).mean()
(sns.FacetGrid(df,

              hue="Bullet_Type",

              height=5,

              xlim = (0,300))

   .map(sns.kdeplot, "Damage" , shade=True)

   .add_legend()

);
sns.barplot(y="Weapon_Type" , x = "Range" , data=df)
df[["Weapon_Name" , "Damage"]].sort_values(by = "Damage" , ascending = False)[:10].reset_index(drop=True)
df[["Weapon_Name" , "Range"]].sort_values(by = "Range" , ascending = False)[:5].reset_index(drop=True)
wt_range=df[["Weapon_Type" , "Range"]].groupby(["Weapon_Type"],as_index=False).mean().sort_values(by="Range").reset_index(drop=True)

wt_range
sns.pairplot(data=df.select_dtypes(["float64" , "int64"]) , vars = ["Range","Damage","Damage_Per_Second","Magazine_Capacity"]);