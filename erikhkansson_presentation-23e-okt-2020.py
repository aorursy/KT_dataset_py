import pandas as pd

import seaborn as sns

from csv import reader

from matplotlib import pyplot as plt
x = [1, 2, 3, 4]

g = [18, 13, 15, 5]

y = [1, 4, 6, 9]

z = [10, 15, 22, 32]

plt.plot(x, g)

plt.plot (x, y)

plt.plot (x, z)

plt.title("test plot")

plt.xlabel("X-axel")

plt.ylabel("G-, Y- & Z-axel")

plt.show()
ds = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")

ds.columns = ds.columns.str.replace(" ", "_") #vi tar born alla mellanrum i rubrikerna, för Pandas-skull



ds1 = pd.read_csv("../input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv")



ds2 = pd.read_csv("../input/tipsdataset/tips.csv")



ds3 = pd.read_csv("../input/pokemonstages/Pokemon123.csv", sep=";")



ds4 = pd.read_csv("../input/covidstats/corona_dag_region.csv")





sweden = ds[ds['Country'] == "Sweden"]

afghanistan = ds[ds['Country'] == "Afghanistan"]

brasilien = ds[ds['Country'] == "Brazil"]

frankrike = ds[ds['Country'] == "France"]
ds
type(ds)
file = ds4

plt.figure(figsize=(25,5))

plt.title("Totalt antal Covid-19 fall")

plt.xlabel('Datum')

plt.ylabel('Totalt antal fall')

plt.plot(file['Statistikdatum'], file['Totalt_antal_fall'])

plt.xticks(file['Statistikdatum'][::28])

plt.gca().invert_xaxis()

plt.show()
def lineplot(x_data, y_data, x_label="", y_label="", title=""):

    _, ax = plt.subplots()

    ax.plot(x_data, y_data, lw = 4, color = '#539caf', alpha = 1)

    ax.set_title(title)

    ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)

    



    

lineplot (sweden.Year, sweden.Alcohol, "År", "Alkoholkonsumtion, liter per capita", "Sverige" )
def lineplotmultiple(x_data,y_data1, y_data2, y_data3, y_data4, x_label="", y_label="", title="", legend1="", legend2="", legend3="", legend4=""):

    _, ax = plt.subplots()

    ax.plot (x_data, y_data1, lw = 2.5)

    ax.plot (x_data, y_data2, lw = 2.5)

    ax.plot (x_data, y_data3, lw = 2.5)

    ax.plot (x_data, y_data4, lw = 2.5)

    plt.title(title)

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.legend([legend1, legend2, legend3, legend4])

    plt.show



lineplotmultiple (ds.Year[0:16], sweden.Alcohol, brasilien.Alcohol, afghanistan.Alcohol, frankrike.Alcohol, "År", "Alkoholkonsumtion (liter/capita)", "Alkoholkonsumtion per land", "Sverige", "Brasilien", "Afghanistan", "Frankrike")
def barplot(x_data, y_data, x_label="", y_label="", title=""):

    _, ax = plt.subplots()

    ax.bar(x_data, y_data, color = '#2569f0', align = 'center')

    ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)

    ax.set_title(title)

    

barplot(brasilien.Year, brasilien.infant_deaths, "År", "Spädbarnsdödlighet per capita", "Brasiliens spädbarnsdödlighet, utveckling sedan år 2000")
def barplotmultiple(x_data1, x_data2, x_data3, y_data1, y_data2, y_data3, x_label="", y_label="",legend1="", legend2="", legend3=""):

    _, ax = plt.subplots()

    ax.bar(x_data1, y_data1, color = '#2569f0', align = 'center', alpha=0.5)

    ax.bar(x_data2, y_data2, color = '#ff8000', align = 'center', alpha=0.5)    

    ax.bar(x_data3, y_data3, color = '#ff99cc', align = 'center', alpha=0.5)

    plt.legend([legend1, legend2, legend3])

    ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)

    

barplotmultiple(brasilien.Year, frankrike.Year, afghanistan.Year, brasilien.infant_deaths, frankrike.infant_deaths, afghanistan.infant_deaths, "År", "Barnadödlighet per capita", "Brasilien", "Frankrike", "Afghanistan")
def barplotmultiple(x_data1, x_data2, x_data3, y_data1, y_data2, y_data3, x_label="", y_label="",legend1="", legend2="", legend3=""):

    _, ax = plt.subplots()

    ax.bar(x_data1-(1/4), y_data1, color = '#2569f0', align = 'center', alpha=0.7, width=1/4, edgecolor="#2569f0")

    ax.bar(x_data2, y_data2, color = '#ff8000', align = 'center', alpha=0.7, width=1/4, edgecolor='#ff8000')  

    ax.bar(x_data3+(1/4), y_data3, color = '#ff99cc', align = 'center', alpha=0.7, width=1/4, edgecolor='#ff99cc')

    ax.set_xticks(x_data1)

    plt.xticks(rotation=90)

    plt.legend([legend1, legend2, legend3])

    ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)

    #ax.set_title(title)



barplotmultiple(brasilien.Year, frankrike.Year, afghanistan.Year, brasilien.infant_deaths, frankrike.infant_deaths, afghanistan.infant_deaths, "År", "Spädbarnsdödlighet per capita", "Brasilien", "Frankrike", "Afghanistan")
def normalfördelning(x_data1, x_data2, y_data1, x_label1="", x_label2="",legendx="", legendy="", title=""):

    plt.figure(figsize=(10,5))

    sns.distplot(x_data1, color="skyblue", label=x_label1)

    sns.distplot(x_data2, color="red", label=x_label2)

    plt.xlabel(legendx)

    plt.ylabel(legendy)

    plt.title(title)

    plt.legend()

    plt.show()



normalfördelning(ds1[ds1['Gender'] == "Female"].Height,ds1[ds1['Gender'] == "Male"].Height, "Kvinnor", "Män", "Längd", "Längd", "Kumulerad relativ frekvens", "Normalfördelning av längd,  baserat på 500 personer")
sns.lmplot(x='Attack', y='Defense', data=ds3, height=4, aspect=2)
sns.lmplot(x='Attack', y='Defense', data=ds3, fit_reg=False, hue='Stage', height=4, aspect=2.5)

plt.title("Pokemon, försvar kontra styrka, med hänsyn till 'Stage'")

plt.ioff()
pkmnfärger = ['#78C850',  # Grass

              '#F08030',  # Fire

              '#6890F0',  # Water

              '#A8B820',  # Bug

              '#A8A878',  # Normal

              '#A040A0',  # Poison

              '#F8D030',  # Electric

              '#E0C068',  # Ground

              '#EE99AC',  # Fairy

              '#C03028',  # Fighting

              '#F85888',  # Psychic

              '#B8A038',  # Rock

              '#705898',  # Ghost

              '#98D8D8',  # Ice

              '#7038F8',  # Dragon

                   ]
plt.figure(figsize=(20,5))

sns.violinplot(x='Type 1', y='Attack', data=ds3, palette=pkmnfärger)

plt.xticks(rotation=90)

plt.xlabel("Kategori")

plt.ylabel("Attack-stat")

plt.ioff()