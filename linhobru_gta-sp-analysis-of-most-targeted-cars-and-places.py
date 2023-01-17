# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
%matplotlib inline
data = pd.read_csv("../input/all_data_in_one.csv", parse_dates=["BO_INICIADO", "BO_EMITIDO", "DATAOCORRENCIA", "DATACOMUNICACAO", "DATAELABORACAO"])
del(data['Unnamed: 0'])
data.info()
# Filter data only on auto robbery or theft -> there are many rows for other crimes related to robbery and theft.
data.loc[data.RUBRICA == "Roubo (art. 157) - VEICULO",'ATO'] = "Roubo - VEICULO"
data.loc[data.RUBRICA == "A.I.-Roubo (art. 157) - VEICULO",'ATO'] = "Roubo - VEICULO"
data.loc[data.RUBRICA == "Furto (art. 155) - VEICULO",'ATO'] = "Furto - VEICULO"
data.loc[data.RUBRICA == "A.I.-Furto (art. 155) - VEICULO",'ATO'] = "Furto - VEICULO"
data.loc[data.RUBRICA == "Furto qualificado (art. 155, §4o.) - VEICULO",'ATO'] = "Furto - VEICULO"
data.loc[data.RUBRICA == "A.I.-Furto qualificado (art. 155, §4o.) - VEICULO",'ATO'] = "Furto - VEICULO"
data.loc[data.RUBRICA == "Furto de coisa comum (art. 156) - VEICULO",'ATO'] = "Furto - VEICULO"

query = (data.ATO == "Roubo - VEICULO") | (data.ATO == "Furto - VEICULO")
data = data.loc[query, :]
data = data[data.DATAOCORRENCIA.dt.year == 2017]
data.RUBRICA.value_counts()
# Remove duplicated entries, add WEEKDAY (DIA_DA_SEMANA) and MONTH (MES) features
data = data.drop_duplicates(subset=['NUM_BO','PLACA_VEICULO','DESCR_MARCA_VEICULO'])
data['Count'] = data.groupby('NUM_BO')['NUM_BO'].transform('count')
data['DIA_DA_SEMANA'] = data.loc[:,'DATAOCORRENCIA'].dt.weekday_name
data['MES'] = data.loc[:,'DATAOCORRENCIA'].dt.month
data = data.drop_duplicates(subset=['PLACA_VEICULO','MES'])
data = data[(data['DESCR_MARCA_VEICULO'].notnull()) | 
            (data['UF_VEICULO'].notnull()) |
            (data['CIDADE_VEICULO'].notnull()) | 
            (data['PLACA_VEICULO'].notnull()) |
            (data['Count']<2)]
data.loc[data.ANO_FABRICACAO == 0] = np.nan
data.loc[data.ANO_MODELO == 0] = np.nan
data = data.loc[data.NUM_BO.notnull()]
#data = data[data.PLACA_VEICULO.notnull()]
data.info()
data[10000:10030]
data['DESCR_TIPO_VEICULO'].value_counts()
# Aggregating similar vehicle categories in a new feature: TIPO_VEICULO
data.loc[(data['DESCR_TIPO_VEICULO'] == "MOTONETA") | (data['DESCR_TIPO_VEICULO'] == "MOTOCICLO") | (data['DESCR_TIPO_VEICULO'] == "CICLOMOTO"),'TIPO_VEICULO'] = "MOTO"
data.loc[(data['DESCR_TIPO_VEICULO'] == "CAMINHONETE") | (data['DESCR_TIPO_VEICULO'] == "CAMIONETA") | (data['DESCR_TIPO_VEICULO'] == "UTILIT?RIO") | (data['DESCR_TIPO_VEICULO'] == "AUTOMOVEL"),'TIPO_VEICULO'] = "AUTOMOVEL"
data.loc[(data['DESCR_TIPO_VEICULO'] == "CAMINH?O") | (data['DESCR_TIPO_VEICULO'] == "CAMINH?O TRATOR") | (data['DESCR_TIPO_VEICULO'] == "SEMI-REBOQUE") | (data['DESCR_TIPO_VEICULO'] == "REBOQUE"),'TIPO_VEICULO'] = "CAMINHAO"
data.loc[(data['DESCR_TIPO_VEICULO'] == "MICRO-ONIBUS") | (data['DESCR_TIPO_VEICULO'] == "ONIBUS"),'TIPO_VEICULO'] = "ONIBUS"
data.loc[(data['TIPO_VEICULO'].isnull()) & (data['DESCR_TIPO_VEICULO'].notnull()),'TIPO_VEICULO'] = "OUTROS"
data['TIPO_VEICULO'].value_counts()
# CLEANING VEHICLE NAME FEATURE
data.loc[(data['DESCR_MARCA_VEICULO'].str[0:2] == "I/") | (data['DESCR_MARCA_VEICULO'].str[0:2] == "H/") | (data['DESCR_MARCA_VEICULO'].str[0:2] == "R/"),'DESCR_MARCA_VEICULO'] = data['DESCR_MARCA_VEICULO'].str[2:]
data.loc[data['DESCR_MARCA_VEICULO'].str[0:4] == "IMP/",'DESCR_MARCA_VEICULO'] = data['DESCR_MARCA_VEICULO'].str[4:]
data.loc[:,'DESCR_MARCA_VEICULO'] = data['DESCR_MARCA_VEICULO'].str.replace("NOVO ","")
data.loc[:,'DESCR_MARCA_VEICULO'] = data['DESCR_MARCA_VEICULO'].str.replace("NOVA ","")
# Creating 2 new features for car Brand (MARCA) and Model (Modelo)
data['MARCA'] = data['DESCR_MARCA_VEICULO'].str.split("/", expand=True)[0].str.split(" ", expand=True)[0]
data['MODELO'] = data['DESCR_MARCA_VEICULO'].str.split("/", expand=True)[1].str.split(" ", expand=True)[0]
data.loc[data.MODELO == 'VW','MODELO'] = data['DESCR_MARCA_VEICULO'].str.split("/", expand=True)[1].str.split(" ", expand=True)[1]
# Aggregating some brands that had variations in their names
data.loc[data['MARCA'] == "CHEV",'MARCA'] = "CHEVROLET"
data.loc[data['MARCA'] == "GM",'MARCA'] = "CHEVROLET"
data.loc[(data['MARCA'] == "VOLKS") | (data['MARCA'] == "VOLKSWAGEN") | (data['MARCA'] == "VOLKSWAGEM"),'MARCA'] = "VW"
data.loc[(data['MARCA'] == "M.B.") | (data['MARCA'] == "MBENZ") | (data['MARCA'] == "MERCEDES"),'MARCA'] = "M.BENZ"
data.loc[data['MARCA'] == "MMC",'MARCA'] = "MITSUBISHI"
data.loc[data['MARCA'] == "IVECOFIAT",'MARCA'] = "IVECO"
data.loc[(data['MARCA'] == "LR") | (data['MARCA'] == "LROVER"),'MARCA'] = "LAND ROVER"
#show brands that were targeted the most
marca = data.loc[data.TIPO_VEICULO == "AUTOMOVEL",'MARCA'].value_counts()
marca[marca>=10]
# Show models that were the most targeted
modelo = data.loc[data.TIPO_VEICULO == 'AUTOMOVEL','MODELO'].value_counts()
modelo[modelo >1000]
# MOST ROBBED MOTO BRANDS
carros_roubados = data.loc[data['TIPO_VEICULO'] == 'MOTO','MARCA']
ordem = carros_roubados.value_counts()[:10].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(carros_roubados, order=ordem)
plt.xticks(rotation=45)
plt.ylabel("Roubos de Motos")
plt.title("Marcas de motos mais roubadas / furtadas em 2017")
plt.show()
# MOST TARGETED CAR BRANDS
carros_roubados = data.loc[data['TIPO_VEICULO'] == 'AUTOMOVEL',['MARCA','ATO']]
ordem = carros_roubados['MARCA'].value_counts()[:20].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(x='MARCA', data=carros_roubados, order=ordem)
plt.xticks(rotation=45)
plt.ylabel("Roubos/Furtos de Carros")
plt.title("Marcas de carros mais roubadas / furtadas em 2017")
plt.show()
# MOST TARGETED CAR MODELS
carros_roubados = data.loc[data['TIPO_VEICULO'] == 'AUTOMOVEL',['MODELO','ATO']]
ordem = carros_roubados['MODELO'].value_counts()[:20].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(x='MODELO', data=carros_roubados, order=ordem)
plt.xticks(rotation=45)
plt.ylabel("Roubos/Furtos de Carros")
plt.title("Carros mais roubadas / furtadas em 2017")
plt.show()
# MOST TARGETED CITIES - CARS
carros_roubados = data.loc[data['TIPO_VEICULO'] == 'AUTOMOVEL',['CIDADE','ATO']]
ordem = carros_roubados['CIDADE'].value_counts()[:20].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(x='CIDADE', data=carros_roubados, order=ordem)
plt.xticks(rotation=45)
plt.ylabel("Roubos/Furtos de Automóveis")
plt.title("Cidades com maior incidência de roubos / furtos de automóveis")
plt.show()
# MOST TARGETED CITIES - MOTOS
carros_roubados = data.loc[data['TIPO_VEICULO'] == 'MOTO',['CIDADE']]
ordem = carros_roubados['CIDADE'].value_counts()[:20].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(x='CIDADE', data=carros_roubados, order=ordem)
plt.xticks(rotation=45)
plt.ylabel("Roubos/Furtos de Motos")
plt.title("Cidades com maior incidência de roubos / furtos de motos")
plt.show()
# BY VEHICLE TYPE
fig, ax = plt.subplots(1,2, figsize=(16,6))
dados1 = pd.DataFrame(data.loc[data.ATO == "Furto - VEICULO",'TIPO_VEICULO'].value_counts())
dados2 = pd.DataFrame(data.loc[data.ATO == "Roubo - VEICULO",'TIPO_VEICULO'].value_counts())
plt.subplot(1,2,1)
plt.pie(dados1,labels=dados1.index,autopct='%1.1f%%', shadow=True, startangle=45)
plt.axis('equal')
plt.title('Furtos de veículos')
#plt.tight_layout()
plt.subplot(1,2,2)
plt.pie(dados2,labels=dados2.index,autopct='%1.1f%%', shadow=True, startangle=45)
plt.axis('equal')
plt.title('Roubos de veículos')
#plt.tight_layout()
plt.show()
# BY TIME OF DAY
fig, ax = plt.subplots(1,2, figsize=(16,6))
dados1 = pd.DataFrame(data.loc[data.ATO == "Furto - VEICULO",'PERIDOOCORRENCIA'].value_counts()).sort_index()
dados2 = pd.DataFrame(data.loc[data.ATO == "Roubo - VEICULO",'PERIDOOCORRENCIA'].value_counts()).sort_index()
plt.subplot(1,2,1)
plt.pie(dados1,labels=dados1.index,autopct='%1.1f%%', shadow=True, startangle=10)
plt.axis('equal')
plt.title('Furtos de veículos')
#plt.tight_layout()
plt.subplot(1,2,2)
plt.pie(dados2,labels=dados2.index,autopct='%1.1f%%', shadow=True, startangle=10)
plt.axis('equal')
plt.title('Roubos de veículos')
#plt.tight_layout()
plt.show()
# MOST TARGETED PLACES (street, house, others)
fig, ax = plt.subplots(1,2, figsize=(16,6))
dados1 = pd.DataFrame(data.loc[data.ATO == "Furto - VEICULO",'DESCRICAOLOCAL'].value_counts())
dados2 = pd.DataFrame(data.loc[data.ATO == "Roubo - VEICULO",'DESCRICAOLOCAL'].value_counts())
plt.subplot(1,2,1)
plt.pie(dados1,labels=dados1.index,autopct='%1.1f%%', shadow=True, startangle=50)
plt.axis('equal')
plt.title('Furtos de veículos')
#plt.tight_layout()
plt.subplot(1,2,2)
plt.pie(dados2,labels=dados2.index,autopct='%1.1f%%', shadow=True, startangle=50)
plt.axis('equal')
plt.title('Roubos de veículos')
#plt.tight_layout()
plt.show()
# BY WEEKDAY
fig, ax = plt.subplots(1,2, figsize=(16,6))
dados1 = pd.DataFrame(data.loc[data.ATO == "Furto - VEICULO",'DIA_DA_SEMANA'].value_counts()).sort_index()
dados2 = pd.DataFrame(data.loc[data.ATO == "Roubo - VEICULO",'DIA_DA_SEMANA'].value_counts()).sort_index()
plt.subplot(1,2,1)
plt.pie(dados1,labels=dados1.index,autopct='%1.1f%%', shadow=True, startangle=0)
plt.axis('equal')
plt.title('Furtos de veículos')
#plt.tight_layout()
plt.subplot(1,2,2)
plt.pie(dados2,labels=dados2.index,autopct='%1.1f%%', shadow=True, startangle=0)
plt.axis('equal')
plt.title('Roubos de veículos')
#plt.tight_layout()
plt.show()
# BY THIEF SEX
dados = data.loc[data.SEXO.notnull(),'SEXO']
plt.pie(dados.value_counts(),labels=dados.value_counts().index,autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.tight_layout()
plt.show()
# MOST TARGETED NEIGHBORHOODS IN SAO PAULO CAPITAL
carros_roubados = data.loc[data['CIDADE'] == 'S.PAULO','BAIRRO']
ordem = carros_roubados.value_counts()[:20].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(carros_roubados, order=ordem)
plt.xticks(rotation=45)
plt.ylabel("Roubos/Furtos")
plt.title("Bairros com maior incidência de roubos e furtos de veículos / motos")
plt.show()
# MOST TARGETED STREETS IN SAO PAULO CAPITAL
carros_roubados = data.loc[data['CIDADE'] == 'S.PAULO','LOGRADOURO']
ordem = carros_roubados.value_counts()[:20].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(carros_roubados, order=ordem)
plt.xticks(rotation=65)
plt.ylabel("Roubos/Furtos")
plt.title("Ruas com maior incidência de roubos e furtos de veículos / motos")
plt.show()
# MOST TARGET CAR MODELS IN ITAIM BIBI NEIGHBORHOOD
carros_roubados = data.loc[(data['TIPO_VEICULO'] == 'AUTOMOVEL') & (data['BAIRRO'] == 'ITAIM BIBI'),'MODELO']
ordem = carros_roubados.value_counts()[:20].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(carros_roubados, order=ordem)
plt.xticks(rotation=45)
plt.ylabel("Roubos/Furtos de Carros")
plt.title('CARROS MAIS ROUBADOS NO ITAIM BIBI')
plt.show()
# NEIGHBORHOODS WHERE CIVIC IS ROBBED THE MOST
carros_roubados = data.loc[(data['MODELO'] == 'CIVIC')& (data['CIDADE'] == 'S.PAULO'),'BAIRRO']
ordem = carros_roubados.value_counts()[:20].index
fig, ax = plt.subplots(1,1, figsize=(16,6))
sns.countplot(carros_roubados, order=ordem)
plt.xticks(rotation=45)
plt.ylabel("Roubos/Furtos")
plt.title('BAIRROS ONDE MAIS ROUBAM CIVIC')
plt.show()
# BY MONTH OF 2017
carros_roubados = data.loc[:,'MES']
fig, ax = plt.subplots(1,1, figsize=(16,10))
sns.countplot(carros_roubados)
plt.xticks(rotation=45)
plt.ylabel("Roubos/Furtos de Carros e Motos")
plt.title('ROUBOS E FURTOS POR MES')
plt.show()
# Correct LAT and LONG features and create a new dataframe to use these data to plot in a heatmap all crimes in SP Capital
data['LATITUDE'] = data.LATITUDE.str.replace(',','.').astype(float)
data['LONGITUDE'] = data.LONGITUDE.str.replace(',','.').astype(float)
heatmap = data.loc[(data.LATITUDE.notnull())&(data.LONGITUDE.notnull())&(data.CIDADE=="S.PAULO"),['LATITUDE','LONGITUDE']]
#import gmplot
#gmap = gmplot.GoogleMapPlotter.from_geocode("Sao Paulo")
#gmap.heatmap(heatmap['LATITUDE'], heatmap['LONGITUDE'])
#gmap.draw("mymap.html")
# VISUALIZING MOTO AND CAR THEFTS IN SAO PAULO CAPITAL
heatmap1 = data.loc[(data.LATITUDE.notnull())&(data.LONGITUDE.notnull())&(data.CIDADE == 'S.PAULO')&(data.TIPO_VEICULO == 'AUTOMOVEL'),['LATITUDE','LONGITUDE']]
heatmap2 = data.loc[(data.LATITUDE.notnull())&(data.LONGITUDE.notnull())&(data.CIDADE == 'S.PAULO')&(data.TIPO_VEICULO == 'MOTO'),['LATITUDE','LONGITUDE']]
#import gmplot
#gmap = gmplot.GoogleMapPlotter.from_geocode("Sao Paulo")
#gmap.scatter(heatmap1['LATITUDE'], heatmap1['LONGITUDE'], '#FFA07A', size=30, marker=False)
#gmap.scatter(heatmap2['LATITUDE'], heatmap2['LONGITUDE'], '#191970', size=30, marker=False)
#gmap.draw("mymap2.html")