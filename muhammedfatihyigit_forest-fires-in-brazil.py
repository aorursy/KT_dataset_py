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
data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding = "latin1")



print("The first 10 column of our data\n")

print(data.head(10))

data.info()
print(data["year"].unique(),"\n")



print(data["state"].unique(),"\n")



print(data["month"].unique())





data["month"].replace({'Janeiro':'January', 'Fevereiro':'February', 'Março':'March', 'Abril':'April', 'Maio':'May', 'Junho':'June', 'Julho':'July'

       ,'Agosto':'August', 'Setembro':'September', 'Outubro':'October', 'Novembro':'November', 'Dezembro':'December'},inplace=True)
summation = pd.pivot_table(data,values="number",index="year",aggfunc=np.sum)

summation
y = []

for each in data["year"].unique():

    x = data[data.year==each]

    y.append(x["number"].sum())

print(y)

    
f,ax = plt.subplots(figsize=(20,10))

plt.plot(data.year.unique(),y,marker="*",color="r")

plt.title("Forest Fires per Year")

plt.xlabel("Years")

plt.ylabel("Count")

plt.xticks(list(range(1998,2018)))

plt.grid()

plt.show()
z = []

for each in data["state"].unique():

    x = data[data.state==each]

    z.append(x["number"].sum())

print(z)
f,ax = plt.subplots(figsize=(30,10))

plt.plot(data.state.unique(),z,marker="*",color="g")

plt.title("Forest Fires per State")

plt.xlabel("States")

plt.ylabel("Count")

plt.grid()

plt.show()