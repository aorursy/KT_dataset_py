import pandas as pd
import plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
data1 = pd.read_csv('../input/heroes_information.csv')
data2 = pd.read_csv('../input/super_hero_powers.csv')
data1.head(2)
data2.head(2)
data1['Publisher'].unique()
marvel = data1[(data1['Publisher'] == 'Marvel Comics')]
Marvel_Height = marvel.sort_values('Height')
Top_15 = Marvel_Height[-15:]
plt.figure(figsize=(10,6))
sns.barplot(Top_15['name'], Top_15['Height'], alpha=1)
plt.xticks(rotation=45)
plt.xlabel('Name of the Superhero', fontsize=14)
plt.ylabel('Height', fontsize=14)
plt.title("Tallest Marvel Superheroes", fontsize=25)
plt.show()

dc = data1[(data1['Publisher'] == 'DC Comics')]
dc_Height = dc.sort_values('Height')
Top_15 = dc_Height[-15:]
plt.figure(figsize=(10,6))
sns.barplot(Top_15['name'], Top_15['Height'], alpha=1)
plt.xticks(rotation=45)
plt.xlabel('Name of the Superhero', fontsize=14)
plt.ylabel('Height', fontsize=14)
plt.title("Tallest DC Superheroes", fontsize=25)
plt.show()
Marvel_Weight = marvel.sort_values('Weight')
Top_15 = Marvel_Weight[-15:]
plt.figure(figsize=(10,6))
sns.barplot(Top_15['name'], Top_15['Weight'], alpha=1)
plt.xticks(rotation=45)
plt.xlabel('Name of the Superhero', fontsize=14)
plt.ylabel('Weight', fontsize=14)
plt.title("Heaviest Marvel Superheroes", fontsize=25)
plt.show()

dc = data1[(data1['Publisher'] == 'DC Comics')]
dc_Height = dc.sort_values('Weight')
Top_15 = dc_Height[-15:]
plt.figure(figsize=(10,6))
sns.barplot(Top_15['name'], Top_15['Weight'], alpha=1)
plt.xticks(rotation=45)
plt.xlabel('Name of the Superhero', fontsize=14)
plt.ylabel('Height', fontsize=14)
plt.title("Heaviest DC Superheroes", fontsize=25)
plt.show()
race = marvel.groupby('Race').size()
race = race.drop('-')
race = race[(race.values>1)]
plt.figure(figsize=(10,10))
plt.pie(race.values, labels=race.index, autopct='%1.1f%%',startangle=140)
plt.axis('equal')
plt.show()
race = dc.groupby('Race').size()
race = race.drop('-')
race = race[(race.values>1)]
plt.figure(figsize=(10,10))
plt.pie(race.values, labels=race.index, autopct='%1.1f%%',startangle=140)
plt.axis('equal')
plt.show()
gender_dc = dc.groupby('Gender').size()
gender_dc = gender_dc.drop('-')
colors = ['lightcoral','lightskyblue']
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.pie(gender_dc.values, labels=gender_dc.index, colors=colors, autopct='%1.1f%%',startangle=140)
plt.title("Gender distribution of DC Universe", fontsize=16)
plt.axis('equal')
gender_m = marvel.groupby('Gender').size()
gender_m = gender_m.drop('-')
plt.subplot(1,2,2)
plt.title("Gender distribution of Marvel Universe", fontsize=16)
plt.pie(gender_m.values, labels=gender_m.index, colors=colors, autopct='%1.1f%%',startangle=140)
plt.axis('equal')
plt.show()
GL= data1[(data1['Publisher'] == 'George Lucas')]
GL
data2.shape
d = {True:1,False:0}
colum = list(data2.columns)
for i in range(1,len(colum)):
    data2[colum[i]] = data2[colum[i]].map(d)
data2.head(2)
data2['total'] = data2.sum(axis=1)
powers = data2.sort_values('total')
powerful = powers[-20:]
plt.figure(figsize=(15,10))
sns.barplot(powerful['hero_names'], powerful['total'], alpha=1)
plt.xticks(rotation=45)
plt.xlabel('Name of the Superhero', fontsize=14)
plt.ylabel('Number of powers', fontsize=14)
plt.title("Powerful comic Superheroes", fontsize=25)
plt.show()
len(data2[(data2['Flight'] == 1)])
len(data2[(data2['Accelerated Healing'] == 1)])
len(data2[(data2['Animal Attributes'] == 1)])
data2[(data2['Animal Attributes'] == 1)][:5]
data2[(data2['Speed Force'] == 1)]
(data2[(data2['Phoenix Force'] == 1)])
(data2[(data2['Omnipresent'] == 1)])
