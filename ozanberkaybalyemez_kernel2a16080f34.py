# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv") #dosya okuma

df = df.drop(["Sex","NOC","Games","Season","City"],axis=1) # hem aklımda kalsın diye hem de verilere göz gezdirirken rahatlık olsun diye kaldırdığım kolonlar

boxing = df[(df.Sport == "Boxing") & (df.Event == "Boxing Men's Heavyweight") & (df.Height > 1.0) & (df.Weight > 1.0)] #sadece erkek ağır siklet boks kategorisin almak için kullanılan kod, height ve weight verilerine bakacağım için Nan olarak gözüken değerleri kaldırmak istedim.



gold = boxing[boxing.Medal == "Gold"] # madalyalarına göre boksörler

silver = boxing[boxing.Medal == "Silver"] # madalyalarına göre boksörler

bronze = boxing[boxing.Medal == "Bronze"] # madalyalarına göre boksörler

nan = boxing[boxing.Medal != ("Gold","Silver","Bronze")] # madalyalarına göre boksörler







plt.scatter(gold.Weight,gold.Height,color="red",label= "gold") #scatter kodları

plt.scatter(silver.Weight,silver.Height,color="green",label= "silver")

plt.scatter(bronze.Weight,bronze.Height,color="blue",label= "bronze")

plt.legend()

plt.xlabel("H")

plt.ylabel("W")

plt.show()


