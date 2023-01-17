# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv("../input/india-districts-census-2011.csv")
df = df[df["State name"] == "MAHARASHTRA"]
df

df.info()

df.describe()
import matplotlib.pyplot as plt
plt.plot(df["District name"],df["Population"])
plt.xticks(rotation=90)
plt.bar(df["District name"],df["Male"], label="Male")
plt.bar(df["District name"],df["Female"], label="Female",color='g')
plt.xticks(rotation=90)
#analysis of thane district
thane = df[df["District name"] == "Thane"]
thane
#thane district sex ratio
M = int(thane["Male"])
F = int(thane["Female"])
slices=[M,F]
labels = ["Male","Female"]
cols = ["c","m"]
plt.pie(slices,
        labels=labels,
        colors=cols,
        startangle=90,
        shadow= True,
        explode=(0,0.1),
        autopct='%1.1f%%')

plt.title('Sex ratio of Thane district')
plt.show()
int(thane["Hindus"])
#Religion wise distribution of thane districts
slices = [int(thane["Hindus"]),int(thane["Muslims"]),int(thane["Christians"]),int(thane["Sikhs"]),int(thane["Buddhists"]),int(thane["Jains"]),int(thane["Others_Religions"]),int(thane["Religion_Not_Stated"])]

plt.pie(slices,
        labels=labels,
        startangle=220,
        shadow= True,
        explode=(0,0,0,0,0,0,0,0),
        autopct='%1.1f%%',
       labeldistance=1)

plt.title('Sex ratio of Thane district')
plt.axis('equal')
plt.show()
#code to display the religion wise distribution of given district in given state