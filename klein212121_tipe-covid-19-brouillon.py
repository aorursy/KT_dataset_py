import numpy as np 
import pandas as pd 
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html (pour les plots)
import math
import matplotlib.dates as mdates
import PIL
import io

from IPython.display import Markdown
def bold(string):
    display(Markdown(string)) # Pour les caractères gras

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
covid19_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid19_data.rename(columns={"Country/Region":"Country_Region"}, inplace=True)
covid19_data.replace("Congo (Kinshasa)","DR Congo",inplace=True)
covid19_data.replace("Congo (Brazzaville)","Congo",inplace=True)
covid19_data.replace("Curacao","Curaçao",inplace=True)
covid19_data.replace("Faroe Islands","Faeroe Islands",inplace=True)
covid19_data.replace("Ivory Coast","Côte d'Ivoire", inplace=True)
covid19_data.replace("Macau","Macao",inplace=True)
covid19_data.replace("US","United States",inplace=True)
covid19_data.replace("UK","United Kingdom",inplace=True)
covid19_data.replace("Mainland China","China",inplace=True)
covid19_data.replace("St. Martin","Saint Martin",inplace=True)
covid19_data.replace("Saint Vincent and the Grenadines","St. Vincent & Grenadines",inplace=True)
covid19_data.replace("East Timor","Timor-Leste",inplace=True)
covid19_data.replace("Sao Tome and Principe","Sao Tome & Principe",inplace=True)
covid19_data.replace("Saint Kitts and Nevis","Saint Kitts & Nevis",inplace=True)


covid19_data.head()
china_coro = covid19_data[covid19_data["Country_Region"]== "China"].copy()
china_coro.head() # Je prends les données de la chine seulement
us_coro = covid19_data[covid19_data["Country_Region"]== "United States"].copy()
us_coro.head()   # Je prends les données de US seulement

us_china_data = pd.concat([us_coro,china_coro],join='inner') #Je les concatene 
us_china_data = us_china_data.groupby(["Province/State","Country_Region"])["Last Update","Confirmed","Deaths","Recovered"].max()
us_china_data = us_china_data.groupby(["Country_Region"])["Confirmed","Deaths","Recovered"].sum().reset_index(drop=False)
us_china_data.set_index(["Country_Region"],drop=False,inplace=True)
us_china_data.head()
# Je les trie avec groupby, je prends le maximum, ensuite j'additione les valeurs des provinces entre elles
# Comme ça j'ai l'information au niveau du pays
# Pour les autres pays le décompte a déjà été fait au niveau du pays, je n'ai qu'a prendre 
# la date la plus récente, et pour être sur, je prends aussi la valeur maximal de 
# Confirmed, Deaths et Recovered

with_provinces = covid19_data.groupby(["Country_Region"])["Last Update","Confirmed","Deaths","Recovered"].max()
print(with_provinces.shape)
with_provinces.drop(["United States","China","Diamond Princess","MS Zaandam","Others","('St. Martin',)"],inplace=True) 
#Je retire  les deux premiers ils ont déjà été traité, et j'avais remarqué d'autres valeurs qui n'étaient pas
#des pays, MS Zaandam et Diamond Princess sont des Bateaux de croisière ptdrr
#with_provinces.head(n=20)
print(with_provinces.shape)
recent_data_covid19 = pd.concat([with_provinces,us_china_data],join="inner")
print(recent_data_covid19.shape)

recent_data_covid19.drop("Bahamas, The", axis=0, inplace=True)
recent_data_covid19.drop("The Bahamas",axis=0, inplace=True)
recent_data_covid19.drop(" Azerbaijan",axis=0,inplace=True)
recent_data_covid19.drop("Gambia, The",axis=0, inplace=True)
recent_data_covid19.drop("The Gambia",axis=0, inplace=True)
recent_data_covid19.drop("Republic of the Congo",axis=0,inplace=True)

recent_data_covid19.loc["Bahamas","Confirmed"] = + 86
recent_data_covid19.loc["Azerbaijan","Confirmed"]+=1
recent_data_covid19.loc["Gambia","Confirmed"] += 2

recent_data_covid19.tail(n=10) 

pop_by_country = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")
pop_by_country.rename(columns={"Country (or dependency)":"Country_Region"},inplace=True)
#pop_by_country.set_index(["Country_Region"],drop=False,inplace=True)

pop_by_country.replace("Cape Verde","Cabo Verde",inplace=True)
pop_by_country.replace("Czech Republic (Czechia)","Czech Republic",inplace=True)
pop_by_country.replace("East Timor","Timor-Leste",inplace=True)
pop_by_country.replace("Réunion","Reunion",inplace=True)
pop_by_country.replace("State of Palestine","Palestine",inplace=True)

pop_by_country = pop_by_country.sort_values(by="Country_Region")

pop_by_country.head()
#pop_by_country.shape
for index,row in recent_data_covid19.iterrows():
    if index not in pop_by_country['Country_Region'].to_list(): # ça va transformer la colonne" name" en liste
        print(index + " n'est pas dans la liste ")
        
        # Ceux que j'ai laissé en plan sont des pays qui ont des ambiguité sur le nom du pays 
        # par rapport au contexte politique
recent_data_covid19 = pd.merge(recent_data_covid19,pop_by_country,on= "Country_Region")
dataframe_pop = recent_data_covid19["Population (2020)"].to_frame()
recent_data_covid19.drop(["Yearly Change","Net Change","Density (P/Km²)","Land Area (Km²)","Migrants (net)",\
                          "Fert. Rate","World Share","Med. Age","Urban Pop %","Population (2020)"], axis=1, inplace=True)
print(type(dataframe_pop))

bold("** EVOLUTION DE LA COVID-19 EN FONCTION DES PAYS**")

# np.round sert à arrondir les nombres
# le deuxieme nombre me donne le nombre de décimal
recent_data_covid19["Taux brut de confirmés (pour 100)"] = np.round(100*recent_data_covid19["Confirmed"]/dataframe_pop["Population (2020)"],4)
recent_data_covid19["Taux de létalité (pour 100)"] = np.round(100*recent_data_covid19["Deaths"]/recent_data_covid19["Confirmed"],2)
recent_data_covid19["Taux de guérisson (pour 100)"] = np.round(100*recent_data_covid19["Recovered"]/recent_data_covid19["Confirmed"],2)

recent_data_covid19.sort_values('Confirmed', ascending= False).style.background_gradient(cmap='Oranges',subset=["Confirmed"])\
                        .background_gradient(cmap='Reds',subset=["Deaths"])\
                        .background_gradient(cmap='Greens',subset=["Recovered"])\
                        .background_gradient(cmap='Reds',subset=["Taux de létalité (pour 100)"])\
                        .background_gradient(cmap='Greens',subset=["Taux de guérisson (pour 100)"])\
                        .background_gradient(cmap="Oranges",subset=["Taux brut de confirmés (pour 100)"])
                         
# Important: retient ça .background_gradient, subset te permet de donner le nom de la colonne 
# .style donner le genre de discriminant que tu veux

# Dataframe pour les 20 pays où la maladie a le plus touché
country_confirmed = pd.DataFrame(recent_data_covid19.loc[:,["Country_Region","Confirmed","Recovered"]])
country_confirmed.sort_values(by='Confirmed',ascending=False, inplace=True)
country_confirmed = country_confirmed.head(n=20)

# Dataframe pour les 15 pays où la maladie a un taux brut de confirmés le plus élévé
country_rate_confirmed = pd.DataFrame(recent_data_covid19.loc[:,['Country_Region','Taux brut de confirmés (pour 100)']])
country_rate_confirmed.sort_values(by='Taux brut de confirmés (pour 100)',ascending=False, inplace=True)
country_rate_confirmed = country_rate_confirmed.head(n=15)

# Dataframe pour les 15 pays où la maladie a le plus haut de mortalité
country_death_rate = pd.DataFrame(recent_data_covid19.loc[:,["Country_Region",'Taux brut de mortalité (pour 100)']])
country_death_rate.sort_values(by="Taux brut de mortalité (pour 100)",ascending=False, inplace=True)
country_death_rate = country_death_rate.head(n=15)

# Dataframe pour les 15 pays avec le plus haut taux de létalité
country_lethality_rate = pd.DataFrame(recent_data_covid19.loc[:,["Country_Region",'Taux de létalité (pour 100)']])
country_lethality_rate.sort_values(by="Taux de létalité (pour 100)",ascending=False, inplace=True)
country_lethality_rate = country_lethality_rate.head(n=15)

plt.figure(figsize=(12,7))
ax = sns.barplot(x="Confirmed", y="Country_Region", data=country_confirmed)
ax.set(xlabel="Nombre de cas confirmés (10^6)", ylabel="Pays") 
plt.title("TOP 20 DES PAYS LES PLUS TOUCHES PAR LA COVID-19")
plt.show()
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(27,10))

# Il faut absolument mettre en copy() sinon python ne sait pas
# si je modifie le dataframe de base, ou si je veux  une copie

country_rate_con1000 = recent_data_covid19.loc[recent_data_covid19["Confirmed"]>1000,:].copy()
country_rate_con1000.sort_values(by="Taux brut de confirmés (pour 100)",ascending=False, inplace=True)
country_rate_con1000 = country_rate_con1000.head(n=15)

country_rate_con5000 = recent_data_covid19.loc[recent_data_covid19["Confirmed"]>5000,:].copy() 
country_rate_con5000.sort_values(by="Taux brut de confirmés (pour 100)",ascending=False, inplace=True)
country_rate_con5000 = country_rate_con5000.head(n=15)

sns.barplot(x='Taux brut de confirmés (pour 100)',y="Country_Region", data=country_rate_con1000, ax=ax1)
ax1.set_title("TAUX BRUT DE CONFIRMES DES PAYS AYANT PLUS DE 1000 CAS DE LA COVID-19")
ax1.set_xlabel("Taux brut de confirmés (pour 100)")

sns.barplot(x="Taux brut de confirmés (pour 100)",y="Country_Region", data=country_rate_con5000, ax=ax2)
ax2.set_title("TAUX BRUT DE CONFIRMES DES PAYS AYANT PLUS DE 5000 CAS DE LA COVID-19")
ax2.set_xlabel("Taux brut de confirmés (pour 100)")

plt.show()
pd.DataFrame(pop_by_country["Density (P/Km²)"]).describe()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(27,10))

#--------------------------------------------------------------------------------------------
country_rate_con10000 = recent_data_covid19[recent_data_covid19["Confirmed"]>10000].copy()
country_rate_con10000.sort_values(by="Taux brut de confirmés (pour 100)",ascending=False, inplace=True)
country_rate_con10000 = country_rate_con10000.head(n=15)

country_rate_con100000 = recent_data_covid19[recent_data_covid19["Confirmed"]>100000].copy()
country_rate_con100000.sort_values(by="Taux brut de confirmés (pour 100)",ascending=False, inplace=True)
country_rate_con100000 = country_rate_con100000.head(n=15)

sns.barplot(x='Taux brut de confirmés (pour 100)',y="Country_Region", data=country_rate_con10000, ax=ax3)
ax3.set_title("TAUX BRUT DE CONFIRMES PAR PAYS AYANT PLUS DE 10000 CAS (POUR 100) ")
ax3.set_xlabel("Taux brut de confirmés (pour 1000)")

sns.barplot(x="Taux brut de confirmés (pour 100)",y="Country_Region",data=country_rate_con100000, ax=ax4)
ax4.set_title("TAUX BRUT DE CONFIRMES PAR PAYS AYANT PLUS DE 100000 CAS (POUR 100) ")
ax4.set_xlabel("Taux brut de confirmés (pour 100)")

plt.show()
fig, (ax5, ax6) = plt.subplots(1, 2,figsize=(27,10))


#----------------------------------------------------------------------------------------------
country_rate_reco1000 = recent_data_covid19[recent_data_covid19["Confirmed"]>1000].copy()
country_rate_reco1000.sort_values(by="Taux de létalité (pour 100)",ascending=False, inplace=True)
country_rate_reco1000 = country_rate_reco1000.head(n=15)

country_rate_reco5000 = recent_data_covid19[recent_data_covid19["Confirmed"]>5000].copy()
country_rate_reco5000.sort_values(by="Taux de létalité (pour 100)",ascending=False, inplace=True)
country_rate_reco5000 = country_rate_reco5000.head(n=15)

sns.barplot(x='Taux de létalité (pour 100)',y="Country_Region", data=country_rate_reco1000,ax=ax5)
ax5.set_title("TAUX DE LETALITE POUR LES PAYS AYANT PLUS DE 1000 CAS (POUR 100)")
ax5.set_xlabel("Taux de létalité (pour 100)")

sns.barplot(x="Taux de létalité (pour 100)",y="Country_Region",data=country_rate_reco5000, ax=ax6)
ax6.set_title("TAUX DE LETALITE POUR LES PAYS AYANT PLUS DE 5000 CAS (POUR 100)")
ax6.set_xlabel("Taux de létalité (pour 100)")

plt.show()
fig, (ax7,ax8) = plt.subplots(1, 2,figsize=(27,10))

#----------------------------------------------------------------------------------------
country_reco1000 = recent_data_covid19[recent_data_covid19["Confirmed"]>1000].copy()
country_reco1000.sort_values(by="Taux de guérisson (pour 100)",ascending=False, inplace=True)
country_reco1000 = country_reco1000.head(n=15)

country_reco5000 = recent_data_covid19[recent_data_covid19["Confirmed"]>5000].copy()
country_reco5000.sort_values(by="Taux de guérisson (pour 100)",ascending=False, inplace=True)
country_reco5000 = country_reco5000.head(n=15)

sns.barplot(x='Taux de guérisson (pour 100)',y="Country_Region", data=country_reco1000,ax=ax7)
ax7.set_title("TAUX DE GUERISSON POUR LES PAYS AYANT PLUS DE 1000 CAS (POUR 100)")
ax7.set_xlabel("Taux de mortalié (pour 100)")

sns.barplot(x="Taux de guérisson (pour 100)",y="Country_Region",data=country_reco5000, ax=ax8)
ax8.set_title("TAUX DE GUERISSON POUR LES PAYS AYANT PLUS DE 5000 CAS (POUR 100)")
ax8.set_xlabel("Taux de guérisson (pour 100)")

plt.show()
global_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
global_confirmed.head(60)
# Je veux regrouper les données par provinces
global_confirmed = global_confirmed.drop(columns = ["Lat","Long"])
global_confirmed.rename(columns={"Country/Region":"Country_Region"},inplace=True)
global_confirmed.rename(columns={"Province/State":"Province_State"},inplace=True)

#Je garde ça on sait jamais je peux en avoir besoin
global_confirmed_with_provinces = global_confirmed[pd.isnull(global_confirmed["Province_State"])== False]

global_confirmed_with_provinces.head()
# Je veux regrouper les données par pays 
global_confirmed = global_confirmed.groupby('Country_Region').sum()
global_confirmed.head(10)
#Je prend la transposée pour pouvoir ploter
global_confirmed_transposed = global_confirmed.T
global_confirmed_transposed.head()

global_confirmed_transposed.plot(y=['Korea, South','Italy','China','US','France','Germany'],\
                                 figsize=(12,7),use_index=True, marker="*")
plt.xlabel("Dates")
plt.ylabel("Nombre de cas confirmés cumulés (10^6)")
plt.title("Evolution des cas confirmés au cours du temps, mois/jour/année")
plt.show()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.sort_values(by='name',inplace=True)
#world.plot(figsize=(12,6))
#plt.show()
for index,row in global_confirmed.iterrows():
    if index not in world['name'].to_list(): # ça va transformer la colonne" name" en liste
        print(index + " n'est pas dans la liste ")

# Il n'y a pas Andorra dans world
# Il n'y a pas Antigua et Barbuda

world.replace("Bosnia and Herz.","Bosnia and Herzegovina",inplace =True)

# Il n'y a pas Burma dans world
# Il n'y a pas Cabo Verde

world.replace("Central African Rep.","Central African Republic", inplace=True)
global_confirmed.replace("Congo (Brazzaville)","Congo", inplace=True)
global_confirmed.replace("Congo (Kinshasa)","Dem. Rep. Congo",inplace=True)
global_confirmed.replace("Cote d'Ivoire","Côte d'Ivoire	", inplace=True)

# Diamond Princess n'est même pas un pays
# Dominica n'est pas dans world

world.replace("Dominican Republic","Dominican Rep.",inplace=True)
world.replace("Eq. Guinea","Equatorial Guinea",inplace=True)
world.replace("eSwatini","Eswatini",inplace=True)

# Il n'y a pas Grenada dans world
# Holy see c'est pas dans la carte, mais c'est en italie

global_confirmed.replace("Korea, South","South Korea", inplace=True)

# Le Liechtenstein n'est pas dans World
# MS Zaandam est un navire de croisiere
# Maldives n'est pas dans world
# Malta n'est pas dans world
# Mauritus n'est pas dans world
# Monaco n'est pas dans world

global_confirmed.replace("North Macedonia","Macedonia",inplace= True)

# St Kitts pas dans world
# St Lucia """"""""""
# St Vincent and the Grenadines
# San Marino pas ds world
# Seychelles
# Singapore 

world.replace("S. Sudan","South Sudan",inplace=True)
global_confirmed.replace("Taiwan*","Taiwan",inplace=True)
world.replace("United States of America","US",inplace = True)

# West Bank and Gaza je vais le placer en Israël

world.replace("W.Sahara","Western Sahara",inplace=True)


# On va laisser ça comme ça, les petits pays sont à peine visible sur la carte, ça va rien changer 
# à la compréhension de l'expansion de la maladie


