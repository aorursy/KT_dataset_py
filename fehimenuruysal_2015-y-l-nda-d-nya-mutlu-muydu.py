# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings("ignore")



from collections import Counter



import scipy.stats as stats



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")

df_2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

df_2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

df_2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

df_2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
df_2015.head()
df_2015.info()
def duzenle(df):

    kolon = []

    for i in df.columns:

        if "(" in str(i) :

            index = i.find("(")

            i = i.replace(i[index-1:] , "")

        elif "." in str(i):

            index = i.find(".")

            i = i.replace(i[index:] , "")

        i = i.replace(" " , "_")

        kolon.append(i)

    df.columns = kolon

verisetleri = [df_2015, df_2016, df_2017, df_2018, df_2019]



for i in verisetleri:

    duzenle(i)



df_2015.columns
kategorik_degisken = df_2015.select_dtypes(include=["object"]).columns

for cat in kategorik_degisken:

    df_2015[cat] = pd.Categorical(df_2015[cat])

    

df_2015.dtypes
for i in verisetleri:

    print("\n-------------------------\n" , i.isnull().sum())
ulkeler = list(df_2015["Country"].unique())
len(ulkeler)
df_ulkeler = pd.DataFrame(ulkeler)

df_ulkeler
ulke_mutluluk_orani = []





for i in ulkeler:

    deger = df_2015[df_2015["Country"] == i]

    ulke_mutluluk_skoru = sum(deger.Happiness_Score)/len(deger)

    ulke_mutluluk_orani.append(ulke_mutluluk_skoru)
df = pd.DataFrame({"Country": ulkeler , "Happiness_Ratio" : ulke_mutluluk_orani})

yeni_index = (df["Happiness_Ratio"].sort_values(ascending=False)).index.values

sirali_veri = df.reindex(yeni_index)
plt.figure(figsize = (25,15))

sns.barplot(x = sirali_veri["Country"][:50] , y=sirali_veri["Happiness_Ratio"][:50])

plt.xticks(rotation = 90)

plt.xlabel("Ülkeler" , fontsize=18 , color = "blue")

plt.ylabel("Mutluluk Skoru Ortalaması (0-10)" , fontsize= 20 , color = "blue")

plt.show()
bolgeler = df_2015["Region"].unique()

df_bolgeler = pd.DataFrame(bolgeler)

df_bolgeler
bolge_mutluluk_orani = []

for i in bolgeler:

    deger = df_2015[df_2015["Region"] == i]

    bolge_mutluluk = sum(deger.Happiness_Score)/len(deger)

    bolge_mutluluk_orani.append(bolge_mutluluk)

df_bolgeler["Happiness_Ratio"] = bolge_mutluluk_orani
df = pd.DataFrame({"Region": bolgeler , "Happiness_Ratio" : bolge_mutluluk_orani})
plt.figure(figsize=(15,10))

sns.barplot(x=df["Region"] , y = df["Happiness_Ratio"], palette = sns.cubehelix_palette(10))

plt.xticks(rotation = 45)

plt.xlabel("Bölge" , fontsize = 15 , color = "purple")

plt.ylabel("Mutluluk Skoru (0-10)" , fontsize = 20 , color = "purple")

plt.show()
ekonomi = []

aile = []

saglik = []

ozgurluk = []

trust = []



for i in bolgeler :

    deger = df_2015[df_2015["Region"] == i]

    ekonomi.append(sum(deger.Economy)/len(deger))

    aile.append(sum(deger.Family)/len(deger))

    saglik.append(sum(deger.Health)/len(deger))
f,ax = plt.subplots(figsize=(20,15))

sns.barplot(x = ekonomi , y=bolgeler , color = "yellow" , alpha=0.5 , label="ekonomi")

sns.barplot(x = aile , y=bolgeler , color = "green" , alpha=0.5,label="aile")

sns.barplot(x = saglik , y=bolgeler , color = "blue" , alpha=0.5,label="sağlık")



ax.legend(loc="lower right")



plt.ylabel("Bölgeler" , fontsize=25) 

plt.xlabel("Etki Oranı" , fontsize =25)

plt.show();
df_2015["Economy"] = df_2015["Economy"]/max(df_2015["Economy"])

df_2015["Happiness_Score"] = df_2015["Happiness_Score"]/max(df_2015["Happiness_Score"])

data = pd.concat([df_2015["Region"],df_2015["Economy"] , df_2015["Happiness_Score"] , df_2015["Trust"]] , axis = 1)
f,ax = plt.subplots(figsize=(20,10))

plt.xticks(rotation=45)

sns.pointplot(x="Region" , y="Economy" , data = data , color = "red")

sns.pointplot(x="Region" , y ="Happiness_Score" , data = data , color="green")

sns.pointplot(x="Region" , y ="Trust" , data = data , color="blue")

plt.ylabel("Bölgesel Dağılım")

plt.grid()
sns.jointplot(df_2015.Economy , df_2015.Happiness_Score, kind="kde" , size=7).annotate(stats.pearsonr)

plt.show()
sns.jointplot(df_2015.Economy , df_2015.Health , ratio = 3 , size = 7 , color="g").annotate(stats.pearsonr)

plt.show()
labels = bolgeler

explode = np.zeros(10)

explode[1:3] = 0.2

sizes = bolge_mutluluk_orani
plt.figure(figsize=(7,7))

plt.pie(sizes, explode = explode , labels = labels , autopct="%1.1f%%" )

plt.show()
sns.lmplot(x="Happiness_Score" , y ="Health" , data=df_2015)

plt.show()
sns.kdeplot(df_2015.Happiness_Score , df_2015.Freedom , shade=True , cut=2, color = "green")

plt.show()
sns.violinplot(data=df_2015.Standard_Error , color="cyan" , inner="points")

plt.show()
sns.pairplot(df_2015);
f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(df_2015.corr() , annot=True , linewidth=.8 , linecolor="cyan" , fmt=".1f" ,ax = ax)

plt.show()
plt.figure(figsize=(15,8))

sns.boxplot(x="Region" , y = "Happiness_Score", data = df_2015, palette="PRGn")

plt.xticks(rotation=90)

plt.show()
durum = []

for i in df_2015.Happiness_Score:

    if i <= 0.75:

        durum.append("Mutsuz")

    else:

        durum.append("Mutlu")

    

    

df_2015["State"] = durum

df_2015.head()
plt.figure(figsize=(7,5))

sns.swarmplot(x="Region" , y="Happiness_Score", hue="State",  data = df_2015)

plt.xticks(rotation=90)

plt.show()
sns.countplot(df_2015.State)

plt.show()