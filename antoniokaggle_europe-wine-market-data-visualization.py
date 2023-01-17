#Import libraries

from IPython.display import display

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 
#Import dataset

dt = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")

display(dt.head())

dt.shape
plt.figure(figsize=(20,7))

a = sns.countplot(dt["country"])

a.set_xlabel("Country",fontsize=20)

a.set_ylabel("Number of different bottles produced",fontsize=15)

plt.title("Where is wine mostly produced?", fontsize= 25)

plt.xticks(rotation=90)

plt.show()
dt_eur = dt[(dt.country == "Italy") | (dt.country == "Portugal") | (dt.country == "Spain") | (dt.country == "France")]

dt_usa = dt[dt.country == "US"]



dt_eur_and_usa = pd.concat([dt_eur, dt_usa])

display(dt_eur.shape, dt_usa.shape, dt_eur_and_usa.shape)
number_of_wine_US = dt_eur_and_usa.country[dt_eur_and_usa.country == "US"].count()

number_of_wine_eur = dt_eur_and_usa.country.count() - number_of_wine_US

explode = (0.05,0.05)

sizes = [number_of_wine_US/len(dt_eur_and_usa)*100, number_of_wine_eur/len(dt_eur_and_usa)*100]



plt.figure(figsize=(12,8))

plt.pie(sizes, labels=["USA", "Europe (Fra, Ita, Spa, Por)"],colors = ["lightgreen", "lightblue"], shadow=True, startangle=90,explode = explode, autopct='%1.1f%%', textprops={'fontsize': 20})

my_circle=plt.Circle( (0,0), 0.4, color='white')

plt.title("Who produces more variety of wine?", fontsize= 20)

plt.suptitle("USA vs Europe", fontsize= 25)

plt.gca().add_artist(my_circle)

plt.show()
dt_ita = dt[dt.country == "Italy"]

dt_por = dt[dt.country == "Portugal"]

dt_spa = dt[dt.country == "Spain"]

dt_fra = dt[dt.country == "France"]





display(dt_ita.shape, dt_por.shape, dt_spa.shape, dt_fra.shape, dt_eur.shape)
plt.figure(figsize=(20,8))

a = sns.countplot(dt_eur.country)

a.set_xlabel("Country",fontsize=30)

a.set_ylabel("Types of wine produced",fontsize=25)

plt.title("Italy, Portugal, Spain and France", fontsize= 20)

plt.suptitle("Wine production in Europe", fontsize= 25)

plt.show()
plt.figure(figsize=(20,10))



a = sns.boxplot(x='country', y='price', data=dt_eur)

a.set_xlabel("Country",fontsize=30)

a.set_ylabel("Price per bottle",fontsize=30)

plt.title("Price distribution in Europe", fontsize= 25)

plt.ylim(0,200)

plt.show()
dt_ita.province[dt_ita.province == "Northeastern Italy"] = "Northeastern"
plt.figure(figsize=(20,15))





plt.subplot(2,2,1)

(dt_ita['province'].value_counts().head(5) / len(dt_ita)).plot.bar(color = "blue")

plt.title("Top 5 regions in Italy").set_fontsize('18')

plt.xticks(rotation=0)

plt.plot()



plt.subplot(2,2,2)

(dt_por['province'].value_counts().head(5) / len(dt_por)).plot.bar(color = "orange")

plt.title("Top 5 regions in Portugal").set_fontsize('18')

plt.xticks(rotation=0)

plt.plot()



plt.subplot(2,2,3)

(dt_spa['province'].value_counts().head(5) / len(dt_spa)).plot.bar(color = "green")

plt.title("Top 5 regions in Spain").set_fontsize('18')

plt.xticks(rotation=0)

plt.plot()



plt.subplot(2,2,4)

(dt_fra['province'].value_counts().head(5) / len(dt_fra)).plot.bar(color = "red")

plt.title("Top 5 regions in France").set_fontsize('18')

plt.xticks(rotation=0)

plt.plot()



plt.show()
plt.figure(figsize=(20,15))





plt.subplot(2,2,1)

(dt_ita['variety'].value_counts().head(5) / len(dt_ita)).plot.bar(color = "blue")

plt.title("Top 5 wines produced in Italy").set_fontsize('18')

plt.xticks(rotation=0)

plt.plot()



plt.subplot(2,2,2)

(dt_por['variety'].value_counts().head(5) / len(dt_por)).plot.bar(color = "orange")

plt.title("Top 5 wines produced in Portugal").set_fontsize('18')

plt.xticks(rotation=0)

plt.plot()



plt.subplot(2,2,3)

(dt_spa['variety'].value_counts().head(5) / len(dt_spa)).plot.bar(color = "green")

plt.title("Top 5 wines produced in Spain").set_fontsize('18')

plt.xticks(rotation=0)

plt.plot()



plt.subplot(2,2,4)

(dt_fra['variety'].value_counts().head(5) / len(dt_fra)).plot.bar(color = "red")

plt.title("Top 5 wines produced in France").set_fontsize('18')

plt.xticks(rotation=0)

plt.plot()



plt.show()
display(dt_ita.points.mean())

display(dt_por.points.mean())

display(dt_spa.points.mean())

display(dt_fra.points.mean())
plt.figure(figsize=(20,10))



plt.title("Where is the best wine?", fontsize= 25)

a = sns.boxplot(x='points', y='country', data=dt_eur, showmeans=True, color = "lightgreen")

a.set_xlabel("Points",fontsize=20)

a.set_ylabel("Country",fontsize=20)

plt.show()