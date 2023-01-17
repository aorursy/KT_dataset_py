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
import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool







from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
data.info()
data.head(20)
data.tail(20)
data = data.drop(columns = ["Current Ver","Android Ver"])
data = data.dropna(axis=0) #Drop rows which contain missing values



data["Installs"] = data["Installs"].apply(lambda x: x.replace('+', '') if '+' in str(x) else x) #replace +

data["Installs"] = data["Installs"].apply(lambda x: x.replace(',', '') if ',' in str(x) else x) #replace ,

data["Installs"] = data["Installs"].astype('float')



data["Price"] = data["Price"].apply(lambda x: x.replace('$', '') if '$' in str(x) else x) #replace $

data["Price"] = data["Price"].astype("float")



data["Reviews"] = data["Reviews"].astype("int")



data["Size"] = data["Size"].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

data["Size"] = data["Size"].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

data["Size"] = data["Size"].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

data["Size"] = data["Size"].apply(lambda x: float(str(x).replace('k', '')) / 1024 if 'k' in str(x) else x) #convert size of apps to MB

data["Size"] = data["Size"].astype("float")







data.head(20)
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(15, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

f,ax = plt.subplots(figsize=(20, 5))

sns.countplot(x=data['Category'], palette="hls")

plt.xticks(rotation=45, horizontalalignment='right',fontweight='light',fontsize='large')

plt.show()
sns.catplot(x="Category", y="Rating", kind="swarm", data=data , height= 5 , aspect=5)

plt.xticks(rotation=45, horizontalalignment='right',fontweight='light',fontsize='large')

plt.show()
Sorted_By_Reviews = data.sort_values(by=['Reviews'], ascending=False) # Sorted by reviews 

Sorted_By_Reviews.head(10)
f,ax = plt.subplots(figsize=(15, 5))

sns.barplot(x=Sorted_By_Reviews['App'][:25], y=Sorted_By_Reviews['Reviews'][:25], palette="hls")

plt.xticks(rotation=45, horizontalalignment='right',fontweight='light',fontsize='large')

plt.show()