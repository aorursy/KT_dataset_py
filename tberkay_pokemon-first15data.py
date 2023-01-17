# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Pokemon.csv') #read data

data.shape #(rows,columns)
data.info()
data.describe()
data.corr()
data['Type 2'].value_counts(dropna=False)
data.dropna(inplace=True) #cleaning data
data.head(15)
fig_size = [10,5]
#Line graphs

df = data.iloc[:15,:]

plt.figure(figsize=fig_size)

plt.plot(df.Name,df.Total)

plt.ylabel("Total")

plt.xlabel("Name")

plt.title("TOTAL - NAME")

plt.grid()

plt.xticks(rotation=90)

plt.show()
# we are clarifying data points.

plt.figure(figsize=fig_size)

plt.plot(df.Name,df.Total,'-ob') 

plt.ylabel("Total")

plt.xlabel("Name")

plt.title("TOTAL - NAME")

plt.grid()

plt.xticks(rotation=90)

plt.show()
df2 = data.iloc[15:30,:]

plt.figure(figsize=fig_size)

plt.plot(df.Name,df.Total,'-ob')

plt.plot(df2.Name,df2.Total,'--r') # second graphic

plt.ylabel("Total")

plt.xlabel("Name")

plt.title("TOTAL - NAME")

plt.grid()

plt.xticks(rotation=90)

plt.legend(["df","df2"])# lejand

plt.show()
#scatter plot

plt.figure(figsize=fig_size)

df3=data.iloc[:15,:]

plt.scatter(df3.Name,df3.Attack)

plt.ylabel("Attack")

plt.xlabel("Name")

plt.xticks(rotation=90)

plt.grid()

plt.title("ATTACK")

plt.show()
#Scatter Graph

plt.figure(figsize=fig_size)

df3=data.iloc[:15,:]

plt.scatter(df3.Name,df3.Attack)

plt.scatter(df3.Name,df3.Defense)

plt.ylabel("Value")

plt.xlabel("Name")

plt.xticks(rotation=90)

plt.legend(["Attack", "Defense"])

plt.grid()

plt.title("ATTACK - DEFENSE")

plt.show()
data.head()
# Bar Plot HP

plt.figure(figsize=fig_size)

df4=data.iloc[:15,:]

plt.bar(df4.Name,df4.HP)

plt.ylabel("HP")

plt.xlabel("Name")

plt.title("HP - NAME")

plt.xticks(rotation=90)

plt.show()
data.head(15)
# Barh Plot Speed

plt.figure(figsize=fig_size)

df5=data.iloc[:15,:]

plt.barh(df4.Name,df4.Speed)

plt.ylabel("Name")

plt.xlabel("Speed")

plt.title("Speed - NAME")
#Grouped Bar Plot

plt.figure(figsize=(15,10))

df6=data.iloc[:15,:]

x_pos = np.arange(len(df6))

plt.bar(x_pos+0.2,df6['Sp. Atk'],width = 0.3)

plt.bar(x_pos-0.2,df6['Sp. Def'],width = 0.3)

plt.xticks(x_pos, df6.Name ,rotation=90)

plt.ylabel("Value")

plt.xlabel("Name")

plt.title("Sp. Atk - Sp. Def")

plt.show()
data.head(15)
data.Generation.value_counts()
#histogram

df7=data

plt.figure(figsize=fig_size)

plt.hist(df7.Generation,bins=20)

plt.grid()

plt.ylabel('Values')

plt.xlabel('Generation')

plt.title('GENERATION')

plt.show()