# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly  as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/googleplaystore.csv")

data.head()
newcolumns = []

for each in data.columns:

    each = each.lower()

    each = each.replace(" ","")

    newcolumns.append(each)

data.columns = newcolumns

data.head()

print(len(data.app))

data.drop_duplicates(subset="app", inplace=True)

print(len(data.app))
data.rating.value_counts()
data = data[data.rating != 19.0]
data.rename(columns={"size":"appsize"}, inplace = True) # I changed the column name.

data.appsize.value_counts()



data.appsize = data.appsize.apply(lambda x: str(x).replace("Varies with device","NaN") if "Varies with device" in str(x) else x)  # I have to make it nan value.

data.appsize = data.appsize.apply(lambda x: str(x).replace("M","") if "M" in str(x) else x)

data.appsize = data.appsize.apply(lambda x: float(str(x).replace("k",""))/1000 if "k" in str(x) else x) # we need to divide 1000

data.appsize = data.appsize.astype("float")
data.price = data.price.apply(lambda x: str(x).replace("$","") if "$" in str(x) else x)
data.installs.value_counts
data.installs = data.installs.apply(lambda x: str(x).replace("+","") if "+" in str(x) else x)

data.installs = data.installs.apply(lambda x: str(x).replace(",","") if "," in str(x) else x)

data.installs = data.installs.astype("int64")

data.reviews = data.reviews.astype("int64")

data.price = data.price.astype("float")
# Anymore I can examine the data.

data.sample(5)
#Let's see heatmeap.

data.corr()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
rating= data.rating.dropna()

size= data.appsize.dropna()

installs= data["installs"][data.installs!=0].dropna()

reviews= data["reviews"][data.reviews!=0].dropna()

price = data.price.dropna()



datacorr = pd.concat([rating, size, np.log(installs), np.log10(reviews), price], axis=1)



plt.figure(figsize=(15,15))

sns.pairplot(datacorr,palette="rainbow",markers= ['d'])

plt.show()
labels =data.type.value_counts(sort = True).index

sizes = data.type.value_counts(sort = True)

colors = ["lightblue","orangered"]

explode = (0.1,0)  # explode 1st slice

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percent of Paid App in store',size = 25)

plt.show()
# I want to see rate of between "installs" and "reviews"



labels =["installs", "reviews"]

sizes = [data.installs.sum(), data.reviews.sum()]

colors = ["lightblue","orangered"]

explode = (0.1,0)  # explode 1st slice

# Visualize

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.2f%%', shadow=True, startangle=270,)

plt.title('Installs count vs Reviews count For All Apps',size = 15)

plt.show()
free = data[data.type == "Free"]

labels =["free installs", "free reviews"]

sizes = [free.installs.sum(), free.reviews.sum()]

colors = ["lightblue","orangered"]

explode = (0.1,0)  # explode 1st slice

 # Visualize

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.2f%%', shadow=True, startangle=270,)

plt.title('Installs count vs Reviews count For Free Apps',size = 15)

plt.show()
paid = data[data.type == "Paid"]

labels =["paid installs", "paid reviews"]

sizes = [paid.installs.sum(), paid.reviews.sum()]

colors = ["lightblue","orangered"]

explode = (0.1,0)  # explode 1st slice

 # Visualize

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title("Installs count vs Reviews count For Paid Apps",size = 15)

plt.show()

datapaid = data[data.type == "Paid"].groupby("category")["app"].count().sort_values()

datafree = data[data.type == "Free"].groupby("category")["app"].count().sort_values()
trace1 = go.Bar(

                    x = datafree.index,

                    y = datafree.values,

                    name = "Free",

                    marker = dict(color = 'rgba(16, 52, 200, 0.8)'))

trace2 = go.Bar(

                    x = datapaid.index,

                    y = datapaid.values,

                    name = "Paid",

                    marker = dict(color = 'rgba(200, 80, 2, 0.8)'))



temp_data = [trace1,trace2]

layout = dict(title = 'Free Apps And Paid Apps Comparison',

              xaxis= dict(title= 'Categories',ticklen= 3,zeroline= False))

fig = dict(data = temp_data, layout = layout)

iplot(fig)