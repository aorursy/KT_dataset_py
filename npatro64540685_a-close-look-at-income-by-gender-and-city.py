import pandas as pd

import os

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')





toy = pd.read_csv("../input/toy_dataset.csv")



toy.head()



toy.columns



toy.tail()



toy.info()



toy.describe()



toy['City'] = toy['City'].astype('category')

toy['Illness'] = toy['Illness'].astype('category')

toy['Gender'] = toy['Gender'].astype('category')





sns.boxplot(data = toy, x = "City", y = "Income")



sns.boxplot(data = toy, x = "Gender", y = "Income")



list1 = list()

mylabels = list()

for gender in toy.Gender.cat.categories:

    list1.append(toy[toy.Gender == gender].Income)

    mylabels.append(gender)

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(11.7,8.27)

h = plt.hist(list1, bins=30, stacked=True, rwidth=1, label=mylabels)

plt.title("Income Distribution by Gender",fontsize=35, color="DarkBlue", fontname="Console")

plt.ylabel("Number of Individuals", fontsize=35, color="Red")

plt.xlabel("Income", fontsize=35, color="Green")

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.legend(frameon=True,fancybox=True,shadow=True,framealpha=1,prop={'size':20})

plt.show()





g=sns.FacetGrid(toy,row='City', col='Gender', hue = 'City')

g=g.map(plt.hist, 'Income' )



h=sns.FacetGrid(toy, col = 'City')

h=h.map(sns.boxplot, 'Gender','Income' )



list1 = list()

mylabels = list()

for city in toy.City.cat.categories:

    list1.append(toy[toy.City == city].Income)

    mylabels.append(city)

sns.set_style("whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(11.7,8.27)

h = plt.hist(list1, bins=30, stacked=True, rwidth=1, label=mylabels)

plt.title("Income Distribution by City",fontsize=35, color="DarkBlue", fontname="Console")

plt.ylabel("Number of Individuals", fontsize=35, color="Red")

plt.xlabel("Income", fontsize=35, color="Green")

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.legend(frameon=True,fancybox=True,shadow=True,framealpha=1,prop={'size':20})

plt.show()
