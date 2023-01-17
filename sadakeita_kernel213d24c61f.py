 # importer les pachages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Importer les données 
deces_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
confirme_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
retabli_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
etat_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')

print(deces_df.shape) # pour afficher la taille des données 
print(confirme_df.shape)
print(retabli_df.shape)
print(etat_df.shape)
etat_df.head(10) # pour affricher les dix premier etats les plus touchées
etat_df<[etat_df['contry_Region']=='france'] #afficher juste les statistique de la france
etat_df.columns # Avoir une vusibilité sur les collonnes 
# pour faire un graphique avec les dix premiers pays
etat_class_df.colonmns 
px.bar(
etat_class_df.head(10),
x= 'contry_Region',
y= 'Dearths',
 title='les 10 pays les plus touchés par coronavirus(cas decés)',
color_discrete_sequence=['pink']),
height=550,
widht=800,