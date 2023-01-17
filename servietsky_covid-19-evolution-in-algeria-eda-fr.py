# Importation des bibliotheques necessaires



import pandas as pd

import numpy as np

from difflib import SequenceMatcher

import tqdm

import xgboost as xgb

from datetime import timedelta

import folium

!pip install vincent

import vincent

from vincent import AxisProperties, PropertySet, ValueRef

import plotly.graph_objects as go

from folium.plugins import MarkerCluster



import plotly.express as px



import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importation du fichier CSV DZA-COVID19 qui contient les informations sur l'evolution du COVID19 en Algerie

# et le fichier recuperé via le liens WIKIPEDIA contiens les 48 Wilayas d'algerie avec leur population qui vas etre utile plus tard



df_COVID19 = pd.read_csv('../input/DZA-COVID19.csv', delimiter = ',')



url='https://fr.wikipedia.org/wiki/Liste_des_wilayas_d%27Alg%C3%A9rie_par_population'

df_wilaya=pd.read_html(url, header=0)[0]
# Correction des donnees dans le fichier WIKI



df_wilaya = df_wilaya[df_wilaya['Classement']!= 'TOTAL']



df_wilaya['Classement'] = df_wilaya['Classement'].str.replace('º', '').str.strip()

df_wilaya['Classement'] = df_wilaya['Classement'].str.replace('°', '').str.strip().astype(int)



df_wilaya['Nom'] = df_wilaya['Nom'].str.replace("Wilaya d'", '')

df_wilaya['Nom'] = df_wilaya['Nom'].str.replace("Wilaya de ", '')

df_wilaya['Nom'] = df_wilaya['Nom'].str.replace(" ", '').astype(str)



df_wilaya['Recensement(1987)'] = df_wilaya['Recensement(1987)'].str.replace('+', '')

df_wilaya['Recensement(1987)'] = df_wilaya['Recensement(1987)'].str.replace(',', '')

df_wilaya['Recensement(1987)'] = df_wilaya['Recensement(1987)'].str.replace("\xa0", '', regex=True).astype(str).astype(int)



df_wilaya['Recensement(1998)'] = df_wilaya['Recensement(1998)'].str.replace('+', '')

df_wilaya['Recensement(1998)'] = df_wilaya['Recensement(1998)'].str.replace(',', '')

df_wilaya['Recensement(1998)'] = df_wilaya['Recensement(1998)'].str.replace("\xa0", '', regex=True).astype(str).astype(int)



df_wilaya['Recensement(2008[1])'] = df_wilaya['Recensement(2008[1])'].str.replace('+', '')

df_wilaya['Recensement(2008[1])'] = df_wilaya['Recensement(2008[1])'].str.replace(',', '') 

df_wilaya['Recensement(2008[1])'] = df_wilaya['Recensement(2008[1])'].str.replace("\xa0", '', regex=True).astype(str).astype(int)
# Correspondance entre les Wilaya des 2 sources en se servant de la similarité entre les chaines de caracteres



def similar(a, b):

    return SequenceMatcher(None, a, b).ratio()



d = {}

for i in tqdm.tqdm(df_COVID19['Wilaya']) :

    for j in df_wilaya['Nom'] :

        d[j] = similar(i, j)

    df_COVID19.loc[df_COVID19['Wilaya'].str.contains(i), 'Wilaya'] = max(d, key=d.get)

    d = {}      
# Correction des valeurs négatives



df_COVID19.loc[df_COVID19["Cas confirmés (Cumulés)"] < 0, 'Cas confirmés (Cumulés)'] = df_COVID19.loc[df_COVID19["Cas confirmés (Cumulés)"] < 0, 'Cas confirmés (Cumulés)'] * -1
# Merger les 2 sources et traitement des anomalies



df_final = df_wilaya.merge(df_COVID19, left_on='Nom', right_on='Wilaya', how = 'left')



df_final.drop(['Cas suspects '], axis=1, inplace = True)



df_final = df_final[df_final['date '] != '15/03/2002']

df_final = df_final[df_final['date '] != '05/05/2020']

df_final.loc[(df_final.Wilaya == 'Alger') & (df_final['date '] == '03/03/2020'), 'date '] = '03/04/2020'



df_final.loc[df_final['date '].isna(), 'date '] = '01/03/2020'

df_final['Date_'] = pd.to_datetime(df_final['date '], dayfirst=True)



df_final['Year'] = df_final['Date_'].dt.year

df_final['Month'] = df_final['Date_'].dt.month

df_final['Day'] = df_final['Date_'].dt.day
# Traitement des valuers manquantes en utilisant l'un des meilleur algorithme actuels pour la regression

# je fait usage des valeurs non manquantes comme valeurs d'entrainement pour combler ceux qui sont manquantes

# traitemenent des valeurs manquantes pour le champ : Nombre de patients rétablis



X = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())][['Cas confirmés (Cumulés)',"Nouveau cas au niveau de l'Algérie ", "Décés au niveau de l'Algérie (Cumul)", 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

y = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())]['Nombre de patients rétablis '].values



model = xgb.XGBRegressor()



model.fit(X,y)



X_pred = df_final[(df_final['Nombre de patients rétablis '].isna()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())][['Cas confirmés (Cumulés)',"Nouveau cas au niveau de l'Algérie ", "Décés au niveau de l'Algérie (Cumul)", 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

output = model.predict(data=X_pred)



df_final.loc[(df_final['Nombre de patients rétablis '].isna()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull()), 'Nombre de patients rétablis '] = output.round(0)
# traitemenent des valeurs manquantes pour le champ : Cas confirmés (Cumulés)



X = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())][['Nombre de patients rétablis ', 'Nombre de décés ',"Nouveau cas au niveau de l'Algérie ", "Décés au niveau de l'Algérie (Cumul)", 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

y = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())]['Cas confirmés (Cumulés)'].values



model = xgb.XGBRegressor()



model.fit(X,y)



X_pred = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].isna()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())][['Nombre de patients rétablis ', 'Nombre de décés ',"Nouveau cas au niveau de l'Algérie ", "Décés au niveau de l'Algérie (Cumul)", 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

output = model.predict(data=X_pred)



df_final.loc[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].isna()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull()), 'Cas confirmés (Cumulés)'] = output.round(0)
# traitemenent des valeurs manquantes pour le champ : Nombre de décés





X = df_final[(df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())][['Cas confirmés (Cumulés)',"Nouveau cas au niveau de l'Algérie ", "Décés au niveau de l'Algérie (Cumul)" , 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

y = df_final[(df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())]['Nombre de décés '].values



model = xgb.XGBRegressor()



model.fit(X,y)



X_pred = df_final[(df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].isna()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())][['Cas confirmés (Cumulés)',"Nouveau cas au niveau de l'Algérie ", "Décés au niveau de l'Algérie (Cumul)", 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

output = model.predict(data=X_pred)



df_final.loc[(df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].isna()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull()), 'Nombre de décés '] = output.round(0)
# traitemenent des valeurs manquantes pour le champ : Nouveau cas au niveau de l'Algérie



X = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Wilaya"].notnull())][['Cas confirmés (Cumulés)', 'Nombre de patients rétablis ','Nombre de décés ' , 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

y = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].notnull()) & (df_final["Wilaya"].notnull())]["Nouveau cas au niveau de l'Algérie "].values



model = xgb.XGBRegressor()



model.fit(X,y)



X_pred = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].isna()) & (df_final["Wilaya"].notnull())][['Cas confirmés (Cumulés)', 'Nombre de patients rétablis ','Nombre de décés ', 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

output = model.predict(data=X_pred)



df_final.loc[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Nouveau cas au niveau de l'Algérie "].isna()) & (df_final["Wilaya"].notnull()), "Nouveau cas au niveau de l'Algérie "] = output.round(0)
# traitemenent des valeurs manquantes pour le champ : Décés au niveau de l'Algérie (Cumul)



X = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())][['Cas confirmés (Cumulés)', 'Nombre de patients rétablis ','Nombre de décés ', 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

y = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].notnull()) & (df_final["Wilaya"].notnull())]["Décés au niveau de l'Algérie (Cumul)"].values



model = xgb.XGBRegressor()



model.fit(X,y)



X_pred = df_final[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].isna()) & (df_final["Wilaya"].notnull())][['Cas confirmés (Cumulés)', 'Nombre de patients rétablis ','Nombre de décés ', 'Recensement(2008[1])', 'Year', 'Month', 'Day']].values

output = model.predict(data=X_pred)



df_final.loc[(df_final['Nombre de patients rétablis '].notnull()) & (df_final['Cas confirmés (Cumulés)'].notnull()) & (df_final[ 'Nombre de décés '].notnull()) & (df_final["Décés au niveau de l'Algérie (Cumul)"].isna()) & (df_final["Wilaya"].notnull()), "Décés au niveau de l'Algérie (Cumul)"] = output.round(0)
# Ajout des Wilaya manquantes dans le fichier COVID19 et leur attribuer la valeur 0 pour toutes les informations en rapport (c'est la raison de leur absence)



df_final.Wilaya.fillna(df_final.Nom, inplace=True)



df_final['Cas confirmés (Cumulés)'].fillna(0, inplace=True)

df_final['Nombre de décés '].fillna(0, inplace=True)

df_final['Nombre de patients rétablis '].fillna(0, inplace=True)

df_final["Nouveau cas au niveau de l'Algérie "].fillna(0, inplace=True)

df_final["Décés au niveau de l'Algérie (Cumul)"].fillna(0, inplace=True)
# Correction des valeurs qui étaient manquants et ensuite elles ont été mal predites



for i in ['Cas confirmés (Cumulés)','Nombre de décés ', 'Nombre de patients rétablis ',"Décés au niveau de l'Algérie (Cumul)"] :

    df_final[i + '_Shifted'] = df_final.groupby(['Wilaya'])[i].transform(lambda x:x.shift(periods=1, fill_value=0))



for i in ['Cas confirmés (Cumulés)','Nombre de décés ', 'Nombre de patients rétablis ',"Décés au niveau de l'Algérie (Cumul)"] :

    df_final[i + 'verif'] = df_final[i] - df_final[i + '_Shifted']

    

while df_final[(df_final['Cas confirmés (Cumulés)verif'] < 0) | (df_final['Nombre de décés verif'] < 0) | (df_final['Nombre de patients rétablis verif'] < 0) | (df_final["Décés au niveau de l'Algérie (Cumul)verif"] < 0)].shape[0] > 0 :

    

    for i in ['Cas confirmés (Cumulés)','Nombre de décés ', 'Nombre de patients rétablis ',"Décés au niveau de l'Algérie (Cumul)"] :

        df_final[i + '_Shifted'] = df_final.groupby(['Wilaya'])[i].transform(lambda x:x.shift(periods=1, fill_value=0))





    for i in ['Cas confirmés (Cumulés)','Nombre de décés ', 'Nombre de patients rétablis ',"Décés au niveau de l'Algérie (Cumul)"] :

        df_final[i + 'verif'] = df_final[i] - df_final[i + '_Shifted']

    

    df_final.loc[df_final['Cas confirmés (Cumulés)verif'] < 0, 'Cas confirmés (Cumulés)'] = df_final.loc[df_final['Cas confirmés (Cumulés)verif'] < 0, 'Cas confirmés (Cumulés)_Shifted']

    df_final.loc[df_final['Nombre de décés verif'] < 0, 'Nombre de décés '] = df_final.loc[df_final['Nombre de décés verif'] < 0, 'Nombre de décés _Shifted']

    df_final.loc[df_final['Nombre de patients rétablis verif'] < 0, 'Nombre de patients rétablis '] = df_final.loc[df_final['Nombre de patients rétablis verif'] < 0, 'Nombre de patients rétablis _Shifted']

    df_final.loc[df_final["Décés au niveau de l'Algérie (Cumul)verif"] < 0, "Décés au niveau de l'Algérie (Cumul)"] = df_final.loc[df_final["Décés au niveau de l'Algérie (Cumul)verif"] < 0, "Décés au niveau de l'Algérie (Cumul)_Shifted"]

    

df_final. drop([ 'Cas confirmés (Cumulés)_Shifted', 'Nombre de décés _Shifted', 'Nombre de patients rétablis _Shifted', "Décés au niveau de l'Algérie (Cumul)_Shifted", 'Cas confirmés (Cumulés)verif', 'Nombre de décés verif', 'Nombre de patients rétablis verif', "Décés au niveau de l'Algérie (Cumul)verif"], axis = 1, inplace = True)

# Elargir la plage de date pour toutes les Wilaya



df_final.rename(columns=lambda x: x.strip(), inplace=True)



df_wilaya_tmp = df_final[['Classement', 'Nom', 'Recensement(1987)', 'Recensement(1998)','Recensement(2008[1])', "Taux d'Alphabétisation(2008)[2]",'Taux d’accroissementannuel moyen (1998-2008)[1]']]



df_wilaya = df_final[['Date_','Wilaya']].sort_values(by = 'Date_')



df_COVID19_bfill =  df_final[['Date_','Wilaya', 'Cas confirmés (Cumulés)', 'Nombre de décés', 'Nombre de patients rétablis', "Décés au niveau de l'Algérie (Cumul)"]].sort_values(by = 'Date_')

df_COVID19_zero =  df_final[['Date_', 'Wilaya', "Nouveau cas au niveau de l'Algérie"]].sort_values(by = 'Date_')



def reindex_by_date(df):

    dates = pd.date_range(df_final.Date_.min(), df_final.Date_.max())

    return df.reindex(dates).ffill()



def reindex_by_date_2(df):

    dates = pd.date_range(df_final.Date_.min(), df_final.Date_.max())

    return df.reindex(dates, fill_value=0)



def reindex_by_date3(df):

    dates = pd.date_range(df_final.Date_.min(), df_final.Date_.max())

    return df.reindex(dates).ffill()



appended_data = []



for i in tqdm.tqdm(df_wilaya['Wilaya'].unique()):

    tmp = df_wilaya[df_wilaya['Wilaya'] == i].groupby(['Date_']).max().apply(reindex_by_date).reset_index().copy()

    appended_data.append(pd.DataFrame(tmp).fillna(method = 'ffill').fillna(method = 'bfill'))

    

df_wilaya_bfill = pd.concat(appended_data)

appended_data = []



for i in tqdm.tqdm(df_COVID19_bfill['Wilaya'].unique()):

    tmp = df_COVID19_bfill[df_COVID19_bfill['Wilaya'] == i].groupby(['Date_']).max().apply(reindex_by_date).reset_index().copy()

    appended_data.append(pd.DataFrame(tmp).fillna(method = 'ffill'))

    

Data_new_bfill = pd.concat(appended_data).fillna(0)

appended_data = []



for i in tqdm.tqdm(df_COVID19_zero['Wilaya'].unique()):

    tmp = df_COVID19_zero[df_COVID19_zero['Wilaya'] == i].groupby(['Date_']).max().apply(reindex_by_date_2).reset_index().copy()

    appended_data.append(pd.DataFrame(tmp).fillna(0))



Data_new_zero = pd.concat(appended_data).fillna(0)



Data_new_zero.loc[:,'Wilaya'] = df_wilaya_bfill['Wilaya']

Data_new_bfill.loc[:,'Wilaya'] = df_wilaya_bfill['Wilaya'] 





df_final = Data_new_bfill.merge(Data_new_zero, left_on=['index', 'Wilaya'], right_on=['index', 'Wilaya'])



df_final = df_wilaya_tmp.drop_duplicates().merge(df_final, left_on=['Nom'], right_on=['Wilaya'])



df_final.rename(columns={"index": "Date_"}, inplace = True)



# del Data_new_zero, Data_new_bfill, df_COVID19_bfill, df_COVID19_zero
# calculer des valeurs utiles pour l'analyse



for i in tqdm.tqdm(df_final.Wilaya.unique()) :

    df_final.loc[df_final['Wilaya'] == i ,'Cas confirmés Daily'] = np.r_[df_final.loc[df_final['Wilaya'] == i ,'Cas confirmés (Cumulés)'][df_final[df_final['Wilaya'] == i].first_valid_index()], np.diff(df_final.loc[df_final['Wilaya'] == i ,'Cas confirmés (Cumulés)'])]

    df_final.loc[df_final['Wilaya'] == i ,'Nombre de décés Daily'] = np.r_[df_final.loc[df_final['Wilaya'] == i ,'Nombre de décés'][df_final[df_final['Wilaya'] == i].first_valid_index()], np.diff(df_final.loc[df_final['Wilaya'] == i ,'Nombre de décés'])]

    df_final.loc[df_final['Wilaya'] == i ,'Nombre de patients rétablis Daily'] = np.r_[df_final.loc[df_final['Wilaya'] == i ,'Nombre de patients rétablis'][df_final[df_final['Wilaya'] == i].first_valid_index()], np.diff(df_final.loc[df_final['Wilaya'] == i ,'Nombre de patients rétablis'])]

    df_final.loc[df_final['Wilaya'] == i ,"Décés au niveau de l'Algérie Daily"] = np.r_[df_final.loc[df_final['Wilaya'] == i ,"Décés au niveau de l'Algérie (Cumul)"][df_final[df_final['Wilaya'] == i].first_valid_index()], np.diff(df_final.loc[df_final['Wilaya'] == i ,"Décés au niveau de l'Algérie (Cumul)"])]



#     df_final.loc[df_final['Wilaya'] == i,"Nouveau cas au niveau de l'Algérie (cumulé)"] = df_final.loc[df_final['Wilaya'] == i,"Nouveau cas au niveau de l'Algérie"].cumsum()



df_final['Cas Actifs daily'] = df_final['Nombre de patients rétablis Daily'] - df_final['Nombre de décés Daily'] - df_final['Nombre de patients rétablis Daily'] 

df_final['Cas Actifs cumulé'] = df_final['Cas confirmés (Cumulés)'] - df_final['Nombre de décés'] - df_final['Nombre de patients rétablis'] 

df_final['Ratio confirmés daily'] =  df_final['Cas confirmés Daily']/df_final['Recensement(2008[1])'] *100
# Ajouter des valeurs utiles pour l'affichage dans une carte comme : le chiffre des Wilaya, la logitude et la latitude de chaque Wilaya



Wilaya_code = {

    'Adrar': 1,

    'Chlef': 2,

    'Laghouat': 3,

    'OumElBouaghi': 4,

    'Batna': 5,

    'Béjaïa': 6,

    'Biskra': 7,

    'Béchar': 8,

    'Blida': 9,

    'Bouira': 10,

    'Tamanrasset': 11,

    'Tébessa': 12,

    'Tlemcen': 13,

    'Tiaret': 14,

    'TiziOuzou': 15,

    'Alger': 16,

    'Djelfa': 17,

    'Jijel': 18,

    'Sétif': 19,

    'Saïda': 20,

    'Skikda': 21,

    'SidiBelAbbès': 22,

    'Annaba': 23,

    'Guelma': 24,

    'Constantine': 25,

    'Médéa': 26,

    'Mostaganem': 27,

    "M'Sila": 28,

    'Mascara': 29,

    'Ouargla': 30,

    'Oran': 31,

    'ElBayadh': 32,

    'Illizi': 33,

    'BordjBouArreridj': 34,

    'Boumerdès': 35,

    'ElTarf': 36,

    'Tindouf': 37,

    'Tissemsilt': 38,

    'ElOued': 39,

    'Khenchela': 40,

    'SoukAhras': 41,

    'Tipaza': 42,

    'Mila': 43,

    'AïnDefla': 44,

    'Naâma': 45,

    'AïnTémouchent': 46,

    'Ghardaïa': 47,

    'Relizane': 48

}



df_final.loc[:,'Nom'] = df_final['Nom'].map(Wilaya_code)



df_final.loc[df_final['Wilaya'] == 'Adrar', 'lat'] = 27.870924

df_final.loc[df_final['Wilaya'] == 'Chlef', 'lat'] = 36.165253

df_final.loc[df_final['Wilaya'] == 'Laghouat', 'lat'] = 33.8

df_final.loc[df_final['Wilaya'] == 'OumElBouaghi', 'lat'] = 35.875411

df_final.loc[df_final['Wilaya'] == 'Batna', 'lat'] = 35.555278

df_final.loc[df_final['Wilaya'] == 'Béjaïa', 'lat'] = 36.7558700

df_final.loc[df_final['Wilaya'] == 'Biskra', 'lat'] = 34.850378

df_final.loc[df_final['Wilaya'] == 'Béchar', 'lat'] = 31.616667

df_final.loc[df_final['Wilaya'] == 'Blida', 'lat'] = 36.470039

df_final.loc[df_final['Wilaya'] == 'Bouira', 'lat'] = 36.374894

df_final.loc[df_final['Wilaya'] == 'Tamanrasset', 'lat'] = 22.785

df_final.loc[df_final['Wilaya'] == 'Tébessa', 'lat'] = 35.404167 

df_final.loc[df_final['Wilaya'] == 'Tlemcen', 'lat'] = 34.878333

df_final.loc[df_final['Wilaya'] == 'Tiaret', 'lat'] = 35.37103

df_final.loc[df_final['Wilaya'] == 'TiziOuzou', 'lat'] = 36.711825

df_final.loc[df_final['Wilaya'] == 'Alger', 'lat'] = 36.763056

df_final.loc[df_final['Wilaya'] == 'Djelfa', 'lat'] = 34.672787

df_final.loc[df_final['Wilaya'] == 'Jijel', 'lat'] = 36.820344

df_final.loc[df_final['Wilaya'] == 'Sétif', 'lat'] = 36.191121

df_final.loc[df_final['Wilaya'] == 'Saïda', 'lat'] = 34.830335

df_final.loc[df_final['Wilaya'] == 'Skikda', 'lat'] = 36.876174

df_final.loc[df_final['Wilaya'] == 'SidiBelAbbès', 'lat'] = 35.189937

df_final.loc[df_final['Wilaya'] == 'Annaba', 'lat'] = 36.9

df_final.loc[df_final['Wilaya'] == 'Guelma', 'lat'] = 36.462136

df_final.loc[df_final['Wilaya'] == 'Constantine', 'lat'] = 36.365

df_final.loc[df_final['Wilaya'] == 'Médéa', 'lat'] = 36.264169

df_final.loc[df_final['Wilaya'] == 'Mostaganem', 'lat'] = 35.931151

df_final.loc[df_final['Wilaya'] == "M'Sila", 'lat'] = 35.705833

df_final.loc[df_final['Wilaya'] == 'Mascara', 'lat'] = 35.396644

df_final.loc[df_final['Wilaya'] == 'Ouargla', 'lat'] = 31.935022

df_final.loc[df_final['Wilaya'] == 'Oran', 'lat'] = 35.6976541

df_final.loc[df_final['Wilaya'] == 'ElBayadh', 'lat'] = 33.683176

df_final.loc[df_final['Wilaya'] == 'Illizi', 'lat'] = 26.483333

df_final.loc[df_final['Wilaya'] == 'BordjBouArreridj', 'lat'] = 36.073215,

df_final.loc[df_final['Wilaya'] == 'Boumerdès', 'lat'] = 36.758965

df_final.loc[df_final['Wilaya'] == 'ElTarf', 'lat'] = 36.767199

df_final.loc[df_final['Wilaya'] == 'Tindouf', 'lat'] = 27.671109

df_final.loc[df_final['Wilaya'] == 'Tissemsilt', 'lat'] = 35.607222

df_final.loc[df_final['Wilaya'] == 'ElOued', 'lat'] = 33.35608

df_final.loc[df_final['Wilaya'] == 'Khenchela', 'lat'] = 35.435833

df_final.loc[df_final['Wilaya'] == 'SoukAhras', 'lat'] = 36.286389

df_final.loc[df_final['Wilaya'] == 'Tipaza', 'lat'] = 36.6178786

df_final.loc[df_final['Wilaya'] == 'Mila', 'lat'] = 36.450278

df_final.loc[df_final['Wilaya'] == 'AïnDefla', 'lat'] = 36.0729193

df_final.loc[df_final['Wilaya'] == 'Naâma', 'lat'] = 33.266667

df_final.loc[df_final['Wilaya'] == 'AïnTémouchent', 'lat'] = 35.297489

df_final.loc[df_final['Wilaya'] == 'Ghardaïa', 'lat'] = 32.483333

df_final.loc[df_final['Wilaya'] == 'Relizane', 'lat'] = 35.737344



df_final.loc[df_final['Wilaya'] == 'Adrar', 'long'] = -0.285634 

df_final.loc[df_final['Wilaya'] == 'Chlef', 'long'] = 1.334523

df_final.loc[df_final['Wilaya'] == 'Laghouat', 'long'] = 2.865143 

df_final.loc[df_final['Wilaya'] == 'OumElBouaghi', 'long'] = 7.113526 

df_final.loc[df_final['Wilaya'] == 'Batna', 'long'] = 6.178611 

df_final.loc[df_final['Wilaya'] == 'Béjaïa', 'long'] = 5.0843300

df_final.loc[df_final['Wilaya'] == 'Biskra', 'long'] = 5.728046 

df_final.loc[df_final['Wilaya'] == 'Béchar', 'long'] = -2.216667  

df_final.loc[df_final['Wilaya'] == 'Blida', 'long'] = 2.827699 

df_final.loc[df_final['Wilaya'] == 'Bouira', 'long'] = 3.901998 

df_final.loc[df_final['Wilaya'] == 'Tamanrasset', 'long'] = 5.522778 

df_final.loc[df_final['Wilaya'] == 'Tébessa', 'long'] = 8.124167 

df_final.loc[df_final['Wilaya'] == 'Tlemcen', 'long'] = -1.315 

df_final.loc[df_final['Wilaya'] == 'Tiaret', 'long'] = 1.316988 

df_final.loc[df_final['Wilaya'] == 'TiziOuzou', 'long'] = 4.045914 

df_final.loc[df_final['Wilaya'] == 'Alger', 'long'] = 3.050556 

df_final.loc[df_final['Wilaya'] == 'Djelfa', 'long'] = 3.262995 

df_final.loc[df_final['Wilaya'] == 'Jijel', 'long'] = 5.764525 

df_final.loc[df_final['Wilaya'] == 'Sétif', 'long'] = 5.413733 

df_final.loc[df_final['Wilaya'] == 'Saïda', 'long'] = 0.151713 

df_final.loc[df_final['Wilaya'] == 'Skikda', 'long'] = 6.909208 

df_final.loc[df_final['Wilaya'] == 'SidiBelAbbès', 'long'] = -0.630846 

df_final.loc[df_final['Wilaya'] == 'Annaba', 'long'] = 7.766667 

df_final.loc[df_final['Wilaya'] == 'Guelma', 'long'] = 7.426076 

df_final.loc[df_final['Wilaya'] == 'Constantine', 'long'] = 6.614722 

df_final.loc[df_final['Wilaya'] == 'Médéa', 'long'] = 2.753926 

df_final.loc[df_final['Wilaya'] == 'Mostaganem', 'long'] = 0.089176 

df_final.loc[df_final['Wilaya'] == "M'Sila", 'long'] = 4.541944 

df_final.loc[df_final['Wilaya'] == 'Mascara', 'long'] = 0.14027 

df_final.loc[df_final['Wilaya'] == 'Ouargla', 'long'] = 5.322329 

df_final.loc[df_final['Wilaya'] == 'Oran', 'long'] = -0.6337376

df_final.loc[df_final['Wilaya'] == 'ElBayadh', 'long'] = 1.019273 

df_final.loc[df_final['Wilaya'] == 'Illizi', 'long'] = 8.466667 

df_final.loc[df_final['Wilaya'] == 'BordjBouArreridj', 'long'] = 4.76108 

df_final.loc[df_final['Wilaya'] == 'Boumerdès', 'long'] = 3.474819 

df_final.loc[df_final['Wilaya'] == 'ElTarf', 'long'] = 8.313771 

df_final.loc[df_final['Wilaya'] == 'Tindouf', 'long'] = -8.147435 

df_final.loc[df_final['Wilaya'] == 'Tissemsilt', 'long'] = 1.81081  

df_final.loc[df_final['Wilaya'] == 'ElOued', 'long'] = 6.863186 

df_final.loc[df_final['Wilaya'] == 'Khenchela', 'long'] = 7.143333 

df_final.loc[df_final['Wilaya'] == 'SoukAhras', 'long'] = 7.951111 

df_final.loc[df_final['Wilaya'] == 'Tipaza', 'long'] = 2.3912362

df_final.loc[df_final['Wilaya'] == 'Mila', 'long'] = 6.264444 

df_final.loc[df_final['Wilaya'] == 'AïnDefla', 'long'] = 1.9881527

df_final.loc[df_final['Wilaya'] == 'Naâma', 'long'] = -0.316667 

df_final.loc[df_final['Wilaya'] == 'AïnTémouchent', 'long'] = -1.140373 

df_final.loc[df_final['Wilaya'] == 'Ghardaïa', 'long'] = 3.666667 

df_final.loc[df_final['Wilaya'] == 'Relizane', 'long'] = 0.555987 
px.set_mapbox_access_token("pk.eyJ1IjoibWVoZGlnYXNtaSIsImEiOiJjazkwcXplbGowNDNwM25saDBldzY0NmQwIn0.gYQr41tH3KKMOHnml_REeQ")

fig = px.scatter_mapbox(df_final[df_final['Date_'] >= '16-03-2020'], lat="lat", lon="long", color=np.log10(df_final[df_final['Date_'] >= '16-03-2020']["Nombre de décés"]+1), size=np.log10(df_final[df_final['Date_'] >= '16-03-2020']["Cas confirmés (Cumulés)"]+1),

                  color_continuous_scale="Sunsetdark", zoom=4, animation_frame=df_final[df_final['Date_'] >= '16-03-2020']['Date_'].dt.strftime('%m/%d/%Y'),

                    title='COVID 19 Evolution (Cliquez sur Play)', hover_data = [ 'Cas confirmés (Cumulés)','Nombre de décés', 'Nombre de patients rétablis'], hover_name="Wilaya",

                        labels={'animation_frame':'Date',

                          'long': 'Longitude',

                          'lat' : 'Latitude',

                          'countryterritoryCode': 'Country code',

                            'color' : 'Nombre de décés (Log10)' ,

                            'size' : 'Cas confirmés (Log10)'},width=1000, height=700,

                       )

fig.update_traces(hovertemplate =None)

fig.update(layout_coloraxis_showscale=False)

fig.update_layout(mapbox_style="dark")

fig.show()

fig = px.area(pd.melt(df_final, id_vars=['Date_'], value_vars=['Cas confirmés (Cumulés)', 'Nombre de décés', 'Nombre de patients rétablis']).groupby(['Date_', 'variable']).sum().reset_index(),

              x='Date_',  y="value", color = 'variable',

             title='Augmentations Cumulées Décès, Cas et Rétablissements',

             labels={'Date_' : 'Date',

                     'Count': 'Total',

                    'value': 'Valeur'})

fig.update_layout(hovermode="x")

fig.show()



fig = px.line(df_final, 

              x='Date_',  y="Cas confirmés (Cumulés)", color = 'Wilaya',

             title='Augmentations Cumulées des Cas par wilaya',

             labels={'Date_' : 'Date',

                     'Count': 'Total',

                    'value': 'Valeur'})

# fig.update_layout(hovermode="x")

fig.show()



fig = px.line(df_final, 

              x='Date_',  y="Nombre de décés", color = 'Wilaya',

             title='Augmentations Cumulées des Décès par wilaya',

             labels={'Date_' : 'Date',

                     'Count': 'Total',

                    'value': 'Valeur'})

# fig.update_layout(hovermode="x")

fig.show()



fig = px.line(df_final, 

              x='Date_',  y="Nombre de patients rétablis", color = 'Wilaya',

             title='Augmentations Cumulées des Rétablissements par Wilaya',

             labels={'Date_' : 'Date',

                     'Count': 'Total',

                    'value': 'Valeur'})

# fig.update_layout(hovermode="x")

fig.show()



fig = px.line(df_final, 

              x='Date_',  y="Cas Actifs cumulé", color = 'Wilaya',

             title='Augmentations Cumulées des Cas Actifs par Wilaya',

             labels={'Date_' : 'Date',

                     'Count': 'Total',

                     'value': 'Valeur'})

# fig.update_layout(hovermode="x")

fig.show()
fig = px.bar(pd.melt(df_final, id_vars=['Date_'], value_vars=['Cas confirmés Daily', 'Nombre de décés Daily', 'Nombre de patients rétablis Daily']).groupby(['Date_', 'variable']).sum().reset_index(),

              x='Date_',  y="value", color = 'variable',

             title='Augmentations journalière des Décès, Cas et Rétablissements',

             labels={'Date_' : 'Date',

                     'Count': 'Total',

                    'value': 'Valeur'})

fig.update_layout(hovermode="x")

fig.show()
# max = df_final[df_final['Date'] == df_final['Date'].max]



fig = px.scatter(df_final[df_final['Date_'] == df_final['Date_'].max()].sort_values('Nombre de décés', ascending=False).iloc[:20, :], 

                 x='Cas confirmés (Cumulés)', y='Nombre de décés', color='Wilaya', size='Nombre de décés', height=700,

                 text='Wilaya', log_x=True, log_y=True, title="Décès vs confirmés (l'échelle est en log10)")

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
df_final.loc[df_final['Wilaya'].isin(['Alger','Blida','Bouira','TiziOuzou','Médéa','Boumerdès','Tipaza','AïnDefla']),'Region'] = 'Centre'

df_final.loc[df_final['Wilaya'].isin(['OumElBouaghi','Tébessa','Jijel','Skikda','Annaba','Guelma','Constantine','ElTarf','Khenchela','SoukAhras','Mila']),'Region'] = 'Constantine'

df_final.loc[df_final['Wilaya'].isin(['Adrar','Laghouat','Biskra','Béchar','Tamanrasset','Djelfa','Ouargla','ElBayadh','Illizi','Tindouf','ElOued','Naâma','Ghardaïa']),'Region'] = 'Grand Sud'

df_final.loc[df_final['Wilaya'].isin(['Chlef','Tiaret','Mostaganem','Mascara','Oran','Tissemsilt','Relizane']),'Region'] = 'Oran'

df_final.loc[df_final['Wilaya'].isin(['Batna','Béjaïa','Sétif',"M'Sila",'BordjBouArreridj']),'Region'] = 'Setif'

df_final.loc[df_final['Wilaya'].isin(['Tlemcen','Saïda','SidiBelAbbès','AïnTémouchent',]),'Region'] = 'Tlemcen'
fig = px.treemap(df_final[df_final['Date_'] == df_final['Date_'].max()].sort_values(by='Nombre de décés', ascending=False).reset_index(drop=True), 

                 path=["Region", "Wilaya"], values="Cas confirmés (Cumulés)", height=700,

                 title='Proportions des Nombre de Cas Par wilaya/ region',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(df_final[df_final['Date_'] == df_final['Date_'].max()].sort_values(by='Nombre de décés', ascending=False).reset_index(drop=True),

                 path=["Region", "Wilaya"], values="Nombre de décés", height=700,

                 title='Proportions des Nombre de Décès Par wilaya/ region',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(df_final[df_final['Date_'] == df_final['Date_'].max()].sort_values(by='Nombre de décés', ascending=False).reset_index(drop=True),

                 path=["Region", "Wilaya"], values="Nombre de patients rétablis", height=700,

                 title='Proportions des Nombre de Rétablissements Par wilaya/ region',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(df_final[df_final['Date_'] == df_final['Date_'].max()].sort_values(by='Nombre de décés', ascending=False).reset_index(drop=True),

                 path=["Region", "Wilaya"], values="Cas Actifs cumulé", height=700,

                 title="Proportions des Nombre d'Actifs Par wilaya/ region",

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig = px.pie(df_final.groupby(['Wilaya']).max().reset_index(), values='Cas confirmés (Cumulés)', names='Wilaya', title='Cas confirmés par Wilaya',

             labels={'deaths_cumsum_ByCountry':'COVID 19 Total Deaths',

                      'cases_cumsum' : 'Cases', 

                      'deaths_cumsum': 'Deaths',

                     'variable' : 'Eolution',

                     'dateRep_usa': 'Date',

                     'dateRep': 'Date',

                     'countriesAndTerritories' : 'Country',

                     'countryterritoryCode': 'Country code',

                     'value' : 'Count (Log10)'},)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()



fig = px.pie(df_final.groupby(['Wilaya']).max().reset_index(), values='Nombre de décés', names='Wilaya', title='Nombre de décés par Wilaya',

             labels={'deaths_cumsum_ByCountry':'COVID 19 Total Deaths',

                      'cases_cumsum' : 'Cases', 

                      'deaths_cumsum': 'Deaths',

                     'variable' : 'Eolution',

                     'dateRep_usa': 'Date',

                     'dateRep': 'Date',

                     'countriesAndTerritories' : 'Country',

                     'countryterritoryCode': 'Country code',

                     'value' : 'Count (Log10)'},)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()



fig = px.pie(df_final.groupby(['Wilaya']).max().reset_index(), values='Nombre de patients rétablis', names='Wilaya', title='Nombre de patients rétablis par Wilaya',

             labels={'deaths_cumsum_ByCountry':'COVID 19 Total Deaths',

                      'cases_cumsum' : 'Cases', 

                      'deaths_cumsum': 'Deaths',

                     'variable' : 'Eolution',

                     'dateRep_usa': 'Date',

                     'dateRep': 'Date',

                     'countriesAndTerritories' : 'Country',

                     'countryterritoryCode': 'Country code',

                     'value' : 'Count (Log10)'},)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()



fig = px.pie(df_final.groupby(['Wilaya']).max().reset_index(), values='Cas Actifs cumulé', names='Wilaya', title='Nombre de Cas Actifs par Wilaya',

             labels={'deaths_cumsum_ByCountry':'COVID 19 Total Deaths',

                      'cases_cumsum' : 'Cases', 

                      'deaths_cumsum': 'Deaths',

                     'variable' : 'Eolution',

                     'dateRep_usa': 'Date',

                     'dateRep': 'Date',

                     'countriesAndTerritories' : 'Country',

                     'countryterritoryCode': 'Country code',

                     'value' : 'Count (Log10)'},)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
ax = AxisProperties(labels = PropertySet(angle=ValueRef(value=-5)))



some_map = folium.Map(location=[df_final.groupby(['Wilaya']).max()['lat'].mean(), df_final.groupby(['Wilaya']).max()['long'].mean()], 

                      zoom_start=7)

#creating a Marker for each point in df_sample. Each point will get a popup with their zip

mc = MarkerCluster()

for index, row in df_final.groupby(['Wilaya']).max().reset_index().iterrows(): 

#     display(row._10)

#     bar = vincent.Bar({ 'Cas confirmés (Cumulés)' : row["Cas confirmés (Cumulés)"],'Nombre de décés' : row["Nombre de décés"], 'Nombre de patients rétablis' : row["Nombre de patients rétablis"]}, width=300, height=200)

    bar = vincent.GroupedBar(pd.DataFrame([{ 'Cas confirmés (Cumulés)' : row["Cas confirmés (Cumulés)"],'Nombre de décés' :row["Nombre de décés"], 'Nombre de patients rétablis' : row["Nombre de patients rétablis"]}], index = ['']), width=300, height=200)

    bar.axes[0].properties = ax

    bar.legend(title='legend')

    bar.colors(brew='Set1')

    mc.add_child(folium.Marker(location=[row["lat"],row["long"]], tooltip=row["Wilaya"],

                               popup=folium.Popup(max_width=450).add_child(folium.Vega(bar, width=500, height=250))))

#                                      popup= "Wilaya : " + str(row.Wilaya) + "<br /> Cas confirmés (Cumulés) : " + str(row._10) + "<br /> Nombre de décés : " + str(row._11)+ "<br /> Nombre de patients rétablis : " + str(row._12)+ "<br /> Cas Actifs cumulé : "+  str(row._20) ))

some_map.add_child(mc)

some_map