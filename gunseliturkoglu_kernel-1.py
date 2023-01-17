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
import matplotlib.pyplot as plt

import seaborn as sns  
data = pd.read_csv("../input/youtube-new/CAvideos.csv")

data.head()
data.info()
data.corr()
f, ax = plt.subplots(figsize=(19,19))

sns.heatmap(data.corr(), annot=True, linewidths=0.6, fmt=".2f", ax=ax)

plt.show()
data.head(7)
data.columns
data.likes.plot(kind="line", color="r", label="likes", linewidth=1, alpha=0.5, grid=True,linestyle=":")

data.comment_count.plot(color="g", label="comment_count", linewidth=1, grid=True, linestyle="-.")

plt.legend(loc='upper right')

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')      


data.plot(kind='scatter', x='likes', y='dislikes',alpha = 0.5,color = 'red')

plt.xlabel('likes')              

plt.ylabel('dislikes')

plt.title('Scatter Plot') 
sns.regplot(data.likes, data.comment_count)

sns.regplot(data.likes, data.comment_count, fit_reg=False)

plt.show()
data.category_id.plot(kind="hist", bins=100, figsize=(10,10))

plt.show()
dictionary = {"Izmir":"Konak", "Istanbul":"Beşiktaş", "Ankara":"Çankaya", "Muğla":"Bodrum"}
print(dictionary.keys())

print(dictionary.values())
dictionary["Izmir"]="Çeşme"

print(dictionary)
dictionary["Aydın"]="Kuşadası"

print(dictionary)
del dictionary["Muğla"]

print(dictionary)
series = data["likes"]

print(type(series))
data_frame = data[["dislikes"]]

print(type(data_frame))
print(5>3)

print(5!=2)

print(True and False)

print(True or False)
x = data["likes"]>3000000

data[x]
data[(data["likes"]>3000000) & (data["dislikes"]<150000)]
i=20

while i!=15:

    print("i is :",i)

    i-=1

print(i, "is equal the 15")

lis = [25,26,27,28,29]

for i in lis:

    print(i)
print("index","   value" )

for index, value in enumerate(lis):

    print(index, "      ", value)
dictionary = {"Izmir":"Konak", "Istanbul":"Beşiktaş", "Ankara":"Çankaya", "Muğla":"Bodrum"}
for key,value in dictionary.items():

    print(key, ":   ", value)
for index,value in data[["likes"]][0:1].iterrows():

    print("index:", index, "  value:", value)
