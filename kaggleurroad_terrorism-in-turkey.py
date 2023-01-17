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


pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
terro_data=pd.read_csv('/kaggle/input/terrorism-in-turkey-19962017/TableOfTurkey.csv')
data_copy=terro_data.copy()
data_copy.head()
predictors_without_categoricals = data_copy.select_dtypes(exclude=['object'])
predictors_without_categoricals.head()
data_copy=data_copy.describe(include="all")
data_copy.head()
tr=data_copy.copy()

istkill = tr[tr["city"] == "Istanbul"]["Killed"].sum()

istwound = tr[tr["city"] == "Istanbul"]["Wounded"].sum()



ankkill = tr[tr["city"] == "Ankara"]["Killed"].sum()

ankwound = tr[tr["city"] == "Ankara"]["Wounded"].sum()



izmkill = tr[tr["city"] == "Izmir"]["Killed"].sum()

izmwound = tr[tr["city"] == "Izmir"]["Wounded"].sum()



westkill = istkill + izmkill + ankkill

westwound = istwound + izmwound + ankwound

westintwound = int(westwound)
cizkill = (tr[tr["city"] == "Cizre"]["Killed"].sum()) + (tr[tr["city"] == "Cizre district"]["Killed"].sum())

cizwound = (tr[tr["city"] == "Cizre"]["Wounded"].sum()) + (tr[tr["city"] == "Cizre district"]["Wounded"].sum())



diykill = tr[tr["city"] == "Diyarbakir"]["Killed"].sum()

diywound = tr[tr["city"] == "Diyarbakir"]["Wounded"].sum()



sirkill = tr[tr["city"] == "Sirnak"]["Killed"].sum()

sirkwound = tr[tr["city"] == "Sirnak"]["Wounded"].sum()



yukskill = (tr[tr["city"] == "Yuksekova"]["Killed"].sum()) + (tr[tr["city"] == "Yuksekova district"]["Killed"].sum())

yukswound = (tr[tr["city"] == "Yuksekova"]["Wounded"].sum()) + (tr[tr["city"] == "Yuksekova district"]["Wounded"].sum())



cukkill = tr[tr["city"] == "Cukurca"]["Killed"].sum()

cukwound = tr[tr["city"] == "Cukurca"]["Wounded"].sum()



vankill = tr[tr["city"] == "Van"]["Killed"].sum()

vanwound = tr[tr["city"] == "Van"]["Wounded"].sum()



bingkill = tr[tr["city"] == "Bingol"]["Killed"].sum()

bingwound = tr[tr["city"] == "Bingol"]["Wounded"].sum()



semdkill = (tr[tr["city"] == "Semdinli"]["Killed"].sum()) + (tr[tr["city"] == "Semdinli district"]["Killed"].sum())

semdwound = (tr[tr["city"] == "Semdinli"]["Wounded"].sum()) + (tr[tr["city"] == "Semdinli district"]["Wounded"].sum())



eastkill = semdkill + bingkill + vankill + cizkill + cukkill + diykill + sirkill + yukskill

eastwound = semdwound + bingwound + vanwound + cizwound + cukwound + diywound + sirkwound + yukswound

eastintwound = int(eastwound)
n_groups = 2

y = np.array([eastkill,westkill])

z = np.array([eastwound,westwound])
fig, ax = plt.subplots(figsize=(10,6))

index = np.arange(n_groups)

bar_widht = 0.2

opacity = 0.8



killed = plt.bar(index, y, bar_widht, alpha=opacity, color = "r",label="KILLED")

wounded = plt.bar(index+bar_widht, z, bar_widht, alpha=opacity, color = "g", label="WOUNDED")



plt.xlabel("REGION",size=18)

plt.ylabel("NUMBER",size=18)

plt.title("TERRORISM IN TURKEY (East-West)",size=25)

plt.xticks(index + bar_widht, (("EAST, Killed:", eastkill, "Wounded:", eastintwound),("WEST, Killed:", westkill, "Wounded:",westintwound)))

plt.legend()



plt.show()
plt.figure(figsize=(20,10))

sns.lineplot(data=predictors_without_categoricals)


 

filename='/kaggle/input/terrorism-in-turkey-19962017/TableOfTurkey.csv'

print(filename)