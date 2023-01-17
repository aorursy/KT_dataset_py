# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read Data From File And Assing to variable

data = pd.read_csv("../input/videogamesales/vgsales.csv")



#Print First 10 Row



data.head(10)

#Bar Graph For Showing the top 5 most selled game.

fig = plt.figure()

ax = fig.add_axes([0,0,2,2])



names = data["Name"].head(5)

sales = data["Global_Sales"].head(5)

ax.bar(names,sales, width= 0.5)

ax.set_xlabel("Games")

ax.set_ylabel("Global Sales (Million)")

ax.set_title("Top 5 The Most Selled Video Games")



plt.show()







#Heatmap

f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(),annot = True, lineWidth = .5, fmt = ".1f",ax=ax)
#Frequency Of The Genres



serie = data.groupby('Genre').size()



fig = plt.figure()

ax = fig.add_axes([0,0,2,2])

ax.bar(serie.index,serie.values, width= 0.5)

ax.set_xlabel("Genres")

ax.set_ylabel("Frequency")

ax.set_title("Genre Frequency Of Games")



plt.show()

datas = data[data["Global_Sales"] > 20]



serie = datas.groupby("Platform").size()





fig = plt.figure()

ax = fig.add_axes([0,0,2,2])

ax.bar(serie.index,serie.values, width= 0.7)

ax.set_xlabel("Platforms")

ax.set_ylabel("Games")

ax.set_title("Platforms Which Has Games Sold Over 20M")



plt.show()



#Video Game Count By Year

yearwisegame =  data.groupby("Year")["Name"].count().reset_index()



plt.figure(figsize=(20,10))

plt.title("Games Sold By Year")

plt.ylabel("Sold Games")

#plt.xlabel("Year")

plt.plot(yearwisegame["Year"],yearwisegame["Name"])



plt.show()

#Video Game Publishers who published games sold over 10 M And Number Of Them.

games =  data[data.Global_Sales >=10].groupby("Publisher")["Name"].count().reset_index()

plt.figure(figsize=(20,10))

plt.title("Publishers Published Games That Sold Over 10 Million Copies")

plt.ylabel("Number Of Games")

plt.xlabel("Publishers")

plt.plot(games["Publisher"],games["Name"])



plt.show()