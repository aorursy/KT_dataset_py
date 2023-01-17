# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/world-happiness/2017.csv')
data.info()
data.corr()
data.columns
data.head()
data.tail()
data.Freedom.plot(kind = "line", color = "b", label = "Freedom",linewidth =2,alpha = 0.5,grid = True,linestyle = ':')

data.Family.plot(color = "r",label = "Rank",linewidth=2,alpha =0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title("Line Plot")

plt.show()
data.plot(kind = "scatter", x = "Freedom",y = "Family",alpha = 0.5,color = "red")

plt.xlabel("Freedom")

plt.ylabel("Family")

plt.title("Freedom Family Scatter Plot")

plt.show()
data.Freedom.plot(kind = "hist",bins= 50,figsize = (12,8))

plt.show()
data.Freedom.plot(kind = "hist",bins = 50)

plt.clf()
x = data['Freedom']>0.5

data[x]
data[np.logical_and(data['Freedom']>0.6, data['Family']>0.7)]
data[(data['Freedom']>0.5) & (data['Family']>0.7)]
threshold = sum(data.Freedom)/len(data.Freedom)

data['Freedom_Avg'] = ["high" if i > threshold else "low" for i in data.Freedom]

data.loc[:10,["Freedom_Avg","Freedom"]]
data.shape
print(data["Freedom"].value_counts(dropna = False))
data.describe()
data.head()
data.boxplot(column = "Freedom",by = "Freedom_Avg")
data_new = data.head()

data_new
melted = pd.melt(frame = data_new,id_vars = "Country",value_vars = ['Freedom','Family'])

melted
melted.pivot(index = "Country",columns = "variable",values = "value")
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis = 0,ignore_index = True)

conc_data_row
data1 = data['Freedom'].head()

data2 = data['Family'].head()

conc_data_col = pd.concat([data1,data2],axis = 1)

conc_data_col
data.head()
data.dtypes
data["Country"] = data["Country"].astype('object')
data["Country"].value_counts(dropna = False)
country = ["Serbia","Croatia"]

population = ["1M","2M"]

label = ["Country","Population"]

col = [country,population] 

zipped = list(zip(label,col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
df["Capital"] = ["Belgrad","Zagreb"]

df
df["income"] = 0

df
data2 = df.loc[:,["Country","Population"]]

data2
data1.plot(subplots = True)

plt.show()
data1.plot(kind = "scatter",x = "Freedom",y = "Family")
data1.plot(kind = "hist",y = "Happiness.Score",bins = 50,range = (0,250),normed = True)

plt.show()
data.describe()
data1["Freedom"][1]
data.Freedom[1]
data.loc[1,["Freedom"]]
data[["Freedom","Family"]]
print(type(data["Freedom"]))

print(type(data[["Family"]]))
data.loc[1:10,"Family":"Freedom"]
data.loc[10:1:-1,"Family":"Freedom"]
data.loc[1:5,"Family":]
boolean = data.Freedom > 0.63

data[boolean]
filter1 = data.Freedom > 0.5

filter2 = data.Family > 0.5

data[filter1 & filter2]
data.Family[data.Freedom < 0.56]
data["Total"] = data.Family + data.Freedom

data.head()