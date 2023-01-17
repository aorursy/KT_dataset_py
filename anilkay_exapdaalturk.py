# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

data.head()
turkey=data[data["country"]=="Turkey"]

import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(data=turkey,x="type")
turkey[data["type"]=="TV Show"]
liste=[]

for oyuncular in turkey["cast"]:

    oyuncular_splitted=oyuncular.split(",")

    for oyuncu in oyuncular_splitted:

        liste.append(oyuncu.strip())
peoples=pd.Series(liste)

sortedpeoples=peoples.value_counts(sort=True)

sortedpeoples[0:10]
sortedpeoples[10:20]
print("Only ",len(set(peoples))," people. Sector is small.")
print("Only ",len(set(turkey["director"]))," people. Sector is small.")
turkey["director"].value_counts()
listedinall=[]

for listed in turkey["listed_in"]:

    for l in listed.split(","):

        listedinall.append(l.strip())

    
set(listedinall)
listedinallpd=pd.Series(listedinall)

plt.figure(figsize=(15,10))



ax=sns.countplot(listedinallpd)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

plt.show()
movies=turkey[turkey["type"]=="Movie"]

durations=[]

for duration in movies["duration"]:

    durs=duration.split()

    durations.append(int (durs[0]))
durationspd=pd.Series(durations)

print("Mean Duration: ",durationspd.mean())

print("Median Duration: ",durationspd.median())

print("Max Duration: ",durationspd.max())

print("Min Duration: ",durationspd.min())
turkey.to_csv("turkey.csv",index=False)