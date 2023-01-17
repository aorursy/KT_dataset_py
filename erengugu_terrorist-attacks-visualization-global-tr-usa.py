# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/gtd/globalterrorismdb_0718dist.csv",encoding='windows-1252')
df.head()
df.columns.tolist()
trdf = df[(df.country_txt == "Turkey")] # Republic of Turkey's Data
trdf.head()
mostGlob = df.country_txt.value_counts()[:21]

plt.figure(figsize=(16,12))

ax = sns.barplot(x=mostGlob,y=mostGlob.index,palette="rocket")

plt.title("Most 20 Countries Attacks")

plt.xlabel("# of Attacks")

plt.show()
dfRegion = df.groupby('region_txt').sum()

labels = dfRegion.index

explode = [0,0,0,0,0,0,0,0,0,0,0,0]

size = dfRegion["nkill"].values



plt.figure(figsize=(16,16))

plt.pie(size,explode=explode,labels=labels,autopct="%1.1f%%",pctdistance=0.5)

plt.title("Terrorist Attacks by Region")

plt.show()
dfYear = list(zip(df.iyear.value_counts().index,df.iyear.value_counts().values))

dfYear = sorted(dfYear)

terrorByYear = []

terrorByCounts = []

for i in dfYear:

    terrorByYear.append(i[0])

    terrorByCounts.append(i[1])



plt.figure(figsize=(18,7))

sns.barplot(x=terrorByYear,y=terrorByCounts,palette=sns.color_palette("rocket"))

plt.xticks(rotation=90)

plt.xlabel("Year")

plt.ylabel("# of Attacks")

plt.show()
byYear = df.groupby("iyear").sum()

plt.figure(figsize=(18,7))

byYear.nkill.plot(grid=True)

plt.legend(loc="opper left")

plt.xticks(np.arange(byYear.index.min(),2018,5))

plt.xlabel("Year")

plt.ylabel("Kills")

plt.title("Number of Kills")

plt.show()
regionYear = pd.crosstab(df.iyear,df.region_txt)

regionYear.plot(color=sns.color_palette("bright",12),grid=True)

figure=plt.gcf()

figure.set_size_inches(20,8)

plt.xlabel("Year")

plt.ylabel("# of Attacks")

plt.title("Attacks by Year and Region")

plt.show()
plt.figure(figsize=(20,8))

sns.barplot(x=df.attacktype1_txt.value_counts().index,y=df.attacktype1_txt.value_counts().values,palette="Greens")

plt.xlabel("Methods of Attacks")

plt.ylabel("Numbers of Attacks")

plt.xticks(rotation=45, ha="right")

plt.title("Attacks Type")

plt.show()
trdf.head()
yeartr = trdf["iyear"].value_counts()

listofYear = sorted(list(zip(yeartr.index,yeartr.values)))

attackYear, attackCounts = zip(*listofYear)

attackYear, attackCounts = list(attackYear), list(attackCounts)



f,ax = plt.subplots(figsize=(20,9))

sns.pointplot(x=attackYear,y=attackCounts,color="Green")

plt.xlabel("Year")

plt.xticks(rotation=90)

plt.ylabel("# of Attacks")

plt.grid()
trdf.gname
terrorTr = trdf.gname.value_counts()

terror15 = sorted(list(zip(terrorTr.values[:16],terrorTr.index[:16])),reverse=True)

attacksTr, terrorsTr = zip(*terror15)

attacksTr, terrorsTr = list(attacksTr), list(terrorsTr)



plt.figure(figsize=(20,7))

sns.barplot(x=terrorsTr,y=attacksTr)

plt.xticks(rotation=70)

plt.xlabel("Terror Groups")

plt.ylabel("# of Attacks")

plt.show()
citiesTr = trdf["provstate"].value_counts()

citiesList = sorted(list(zip(citiesTr.values[:26],citiesTr.index[:26])),reverse=True)

citiesAttacked25, citiesCount25 = zip(*citiesList)

citiesAttacked25, citiesCount25 = list(citiesAttacked25), list(citiesCount25)



plt.figure(figsize=(20,9))

sns.barplot(x=citiesAttacked25,y=citiesCount25)

plt.xlabel("# of Attacks")

plt.show()
terrors = trdf.gname.value_counts()[:5].to_frame()

terrors.columns=["gname"]

kill = trdf.groupby("gname")["nkill"].sum().to_frame()

terrors.merge(kill,left_index=True,right_index=True,how="left").plot.bar()

fig=plt.gcf()

fig.set_size_inches(20,8)

plt.show()
usadf = df[(df.country_txt == "United States")] # Republic of Turkey's Data
usadf.head()
yearusa = usadf["iyear"].value_counts()

listofYear = sorted(list(zip(yearusa.index,yearusa.values)))

attackYear, attackCounts = zip(*listofYear)

attackYear, attackCounts = list(attackYear), list(attackCounts)



f,ax = plt.subplots(figsize=(20,9))

sns.pointplot(x=attackYear,y=attackCounts,color="Green")

plt.xlabel("Year")

plt.xticks(rotation=90)

plt.ylabel("# of Attacks")

plt.grid()
terrorUSA = usadf.gname.value_counts()

terror15 = sorted(list(zip(terrorUSA.values[:16],terrorUSA.index[:16])),reverse=True)

attacksUSA, terrorsUSA = zip(*terror15)

attacksUSA, terrorsUSA = list(attacksUSA), list(terrorsUSA)



plt.figure(figsize=(20,7))

sns.barplot(x=terrorsUSA,y=attacksUSA)

plt.xticks(rotation=70)

plt.xlabel("Terror Groups")

plt.ylabel("# of Attacks")

plt.show()
citiesUSA = usadf["provstate"].value_counts()

citiesList = sorted(list(zip(citiesUSA.values[:26],citiesUSA.index[:26])),reverse=True)

citiesAttacked25, citiesCount25 = zip(*citiesList)

citiesAttacked25, citiesCount25 = list(citiesAttacked25), list(citiesCount25)



plt.figure(figsize=(20,9))

sns.barplot(x=citiesAttacked25,y=citiesCount25)

plt.xlabel("# of Attacks")

plt.show()
terrors = usadf.gname.value_counts()[:5].to_frame()

terrors.columns=["gname"]

kill = usadf.groupby("gname")["nkill"].sum().to_frame()

terrors.merge(kill,left_index=True,right_index=True,how="left").plot.bar()

fig=plt.gcf()

fig.set_size_inches(20,8)

plt.show()
df.head()
gerdf = df[(df.country_txt=="Germany")]
Japandf = df[(df.country_txt=="Japan")]
citiesGermany = gerdf["provstate"].value_counts()

citiesList = sorted(list(zip(citiesGermany.values[:26],citiesGermany.index[:26])),reverse=True)

citiesAttacked25, citiesCount25 = zip(*citiesList)

citiesAttacked25, citiesCount25 = list(citiesAttacked25), list(citiesCount25)



plt.figure(figsize=(20,9))

sns.barplot(x=citiesAttacked25,y=citiesCount25)

plt.xlabel("# of Attacks")

plt.show()
citiesJapan = Japandf["provstate"].value_counts()

citiesList = sorted(list(zip(citiesJapan.values[:26],citiesJapan.index[:26])),reverse=True)

citiesAttacked25, citiesCount25 = zip(*citiesList)

citiesAttacked25, citiesCount25 = list(citiesAttacked25), list(citiesCount25)



plt.figure(figsize=(20,9))

sns.barplot(x=citiesAttacked25,y=citiesCount25)

plt.xlabel("# of Attacks")

plt.show()