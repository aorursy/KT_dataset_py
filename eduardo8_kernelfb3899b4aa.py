#importando as bibliotecas

import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 1000)

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))



plt.rcParams['figure.dpi'] = 90



from sklearn.preprocessing import scale



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Carregando arquivo csv usando Pandas

arquivo = '../input/dataset_treino.csv'

dados = pd.read_csv(arquivo)

print(dados.shape)
# Tipo de dados de cada atributo

dados.dtypes
dados['ALVO']=dados['ENERGY STAR Score']

dados.drop('ENERGY STAR Score', inplace=True, axis=1)
dados.isnull().any()
dados.info()
coluna_numericas=['Order',

'Property Id',

'DOF Gross Floor Area',

'Largest Property Use Type - Gross Floor Area (ft²)',

'Year Built',

'Number of Buildings - Self-reported',

'Occupancy',

'Site EUI (kBtu/ft²)',

'Property GFA - Self-Reported (ft²)',

'Source EUI (kBtu/ft²)',

'Latitude',

'Longitude',

'Community Board',

'Council District',

'Census Tract']

dados_numéricos=dados[coluna_numericas]
dados_numéricos.describe()
dados_numéricos.head()
dados.groupby('Order').size().sort_values(ascending=False)
dados_1=dados.copy()
colunas=['Order'

,'Property Id']

dados_1.drop(colunas, inplace=True, axis=1)
dados.describe(include='object')
############

dados.groupby('Borough').size().sort_values(ascending=False)
dados.head()
dados_1.info()
# apaga colunas 

colunas=['Property Name'

,'Parent Property Id'

,'Parent Property Name'

,'BBL - 10 digits'

,'NYC Borough, Block and Lot (BBL) self-reported'

,'NYC Building Identification Number (BIN)'

,'Address 1 (self-reported)'

,'Address 2'

,'Street Number'

,'Street Name'

,'Postal Code'

,'List of All Property Use Types at Property'

,'2nd Largest Property Use Type'

,'2nd Largest Property Use - Gross Floor Area (ft²)'

,'3rd Largest Property Use Type'

,'3rd Largest Property Use Type - Gross Floor Area (ft²)'

,'Metered Areas (Energy)'

,'Metered Areas  (Water)'

,'Fuel Oil #1 Use (kBtu)'

,'Fuel Oil #2 Use (kBtu)'

,'Fuel Oil #4 Use (kBtu)'

,'Fuel Oil #5 & 6 Use (kBtu)'

,'Diesel #2 Use (kBtu)'

,'District Steam Use (kBtu)'

,'Release Date'

,'DOF Benchmarking Submission Status'

,'NTA']

dados_1.drop(colunas, inplace=True, axis=1)
def converter_numero(valor):

    try:

        return float(valor)

    except:

        return np.nan





dados_1['Weather Normalized Site EUI (kBtu/ft²)'] = dados_1['Weather Normalized Site EUI (kBtu/ft²)'].apply(converter_numero)



dados_1['Weather Normalized Site Electricity Intensity (kWh/ft²)'] = dados_1['Weather Normalized Site Electricity Intensity (kWh/ft²)'].apply(converter_numero)

dados_1['Weather Normalized Site Natural Gas Intensity (therms/ft²)'] = dados_1['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].apply(converter_numero)

dados_1['Weather Normalized Source EUI (kBtu/ft²)'] = dados_1['Weather Normalized Source EUI (kBtu/ft²)'].apply(converter_numero)

dados_1['Natural Gas Use (kBtu)'] = dados_1['Natural Gas Use (kBtu)'].apply(converter_numero)

dados_1['Weather Normalized Site Natural Gas Use (therms)'] = dados_1['Weather Normalized Site Natural Gas Use (therms)'].apply(converter_numero)

dados_1['Electricity Use - Grid Purchase (kBtu)'] = dados_1['Electricity Use - Grid Purchase (kBtu)'].apply(converter_numero)

dados_1['Weather Normalized Site Electricity (kWh)'] = dados_1['Weather Normalized Site Electricity (kWh)'].apply(converter_numero)

dados_1['Total GHG Emissions (Metric Tons CO2e)'] = dados_1['Total GHG Emissions (Metric Tons CO2e)'].apply(converter_numero)

dados_1['Direct GHG Emissions (Metric Tons CO2e)'] = dados_1['Direct GHG Emissions (Metric Tons CO2e)'].apply(converter_numero)

dados_1['Indirect GHG Emissions (Metric Tons CO2e)'] = dados_1['Indirect GHG Emissions (Metric Tons CO2e)'].apply(converter_numero)

dados_1['Water Use (All Water Sources) (kgal)'] = dados_1['Water Use (All Water Sources) (kgal)'].apply(converter_numero)

dados_1['Water Intensity (All Water Sources) (gal/ft²)'] = dados_1['Water Intensity (All Water Sources) (gal/ft²)'].apply(converter_numero)
dados_1.shape
# Visualizando as primeiras linhas

dados_1.info()
dados_1.head()
dados_1.shape
"""

Função para fazer a conversão do dado de valores textuais para valores inteiros 

recebe como parâmetro o dataframe, um dicionario para fazer a conversão e o campo (a ser tratado)

"""

def alteraValores(df,dicionario,campo):

    for i in dicionario:

        df[campo]=np.where(df[campo]==i, dicionario[i], df[campo])

    df[campo]=pd.to_numeric(df[campo], errors='ignore')

    return df

    
dados_1.groupby('Borough').size()
#Quantidade de registros nulos para o campo em questão

len(dados_1[dados_1['Borough'].isnull()])
#preenchendo os valores nulos com o valor "Sem Infor"

dados_1['Borough']=np.where(dados_1['Borough'].isnull(), 'Sem Infor', dados_1['Borough'])
dados_1.groupby('Borough').size()
#prepara um dicionário para utilizar na função alteraValores

dict_translate_Borough={'Sem Infor':0,'Bronx':1,'Brooklyn':2,'Manhattan':3,'Queens':4,'Staten Island':5}
#chama a função para alterar alterar os valores do campo 'Borough'

dados_1=alteraValores(dados_1,dict_translate_Borough,'Borough')
sns.distplot(dados_1['Borough'])
dados_1.groupby('DOF Gross Floor Area').size()
#Quantidade de registros nulos para o campo em questão

len(dados_1[dados_1['DOF Gross Floor Area'].isnull()])
# Atualiza os valores nulos com a média

media_DOF_Gross_Floor_Area=round(dados_1['DOF Gross Floor Area'].mean(),2)

media_DOF_Gross_Floor_Area
#atualiza os valores nulos com a média

dados_1['DOF Gross Floor Area'].fillna(media_DOF_Gross_Floor_Area, inplace = True) 
sns.distplot(dados_1['DOF Gross Floor Area'])
dados_1['Primary Property Type - Self Selected'].describe()
dados_1.groupby('Primary Property Type - Self Selected').size()
dict_tanslate_Primary_Property_Type_Self_Selected={'Bank Branch':0

,'College/University':1

,'Courthouse':2

,'Distribution Center':3

,'Financial Office':4

,'Hospital (General Medical & Surgical)':5

,'Hotel':6

,'K-12 School':7

,'Manufacturing/Industrial Plant':8

,'Medical Office':9

,'Mixed Use Property':10

,'Multifamily Housing':11

,'Non-Refrigerated Warehouse':12

,'Office':13

,'Other':14

,'Refrigerated Warehouse':15

,'Residence Hall/Dormitory':16

,'Residential Care Facility':17

,'Retail Store':18

,'Self-Storage Facility':19

,'Senior Care Community':20

,'Supermarket/Grocery Store':21

,'Wholesale Club/Supercenter':22

,'Worship Facility':23

,'Fitness Center/Health Club/Gym':24}
dados_1=alteraValores(dados_1,dict_tanslate_Primary_Property_Type_Self_Selected,'Primary Property Type - Self Selected')
#histograma

sns.distplot(dados_1['Primary Property Type - Self Selected'])
dados_1['Largest Property Use Type'].describe()
dados_1.groupby('Largest Property Use Type').size()
dict_Largest_Property_Use_Type={'Bank Branch':0

,'Courthouse':1

,'Distribution Center':2

,'Financial Office':3

,'Hospital (General Medical & Surgical)':4

,'Hotel':5

,'K-12 School':6

,'Medical Office':7

,'Multifamily Housing':8

,'Non-Refrigerated Warehouse':9

,'Office':10

,'Parking':11

,'Refrigerated Warehouse':12

,'Residence Hall/Dormitory':13

,'Retail Store':14

,'Senior Care Community':15

,'Supermarket/Grocery Store':16

,'Wholesale Club/Supercenter':17

,'Worship Facility':18}
dados_1=alteraValores(dados_1,dict_Largest_Property_Use_Type,'Largest Property Use Type')
sns.distplot(dados_1['Largest Property Use Type'])
#Quantidade de registros nulos para o campo em questão

len(dados_1[dados_1['Weather Normalized Site EUI (kBtu/ft²)'].isnull()])
dados_1.groupby('Weather Normalized Site EUI (kBtu/ft²)').size()
# Atualiza os valores nulos com a média

media_Weather_Normalized_Site_EUI=round(dados_1['Weather Normalized Site EUI (kBtu/ft²)'].mean(),2)

media_Weather_Normalized_Site_EUI



dados_1['Weather Normalized Site EUI (kBtu/ft²)'].fillna(media_Weather_Normalized_Site_EUI, inplace = True) 
#Quantidade de registros nulos para o campo em questão

len(dados_1[dados_1['Weather Normalized Site Electricity Intensity (kWh/ft²)'].isnull()])
# Atualiza os valores nulos com a média

media_Weather_Normalized_Site_Electricity_Intensity=round(dados_1['Weather Normalized Site Electricity Intensity (kWh/ft²)'].mean(),2)



dados_1['Weather Normalized Site Electricity Intensity (kWh/ft²)'].fillna(media_Weather_Normalized_Site_Electricity_Intensity, inplace = True) 
#Quantidade de registros nulos para o campo em questão

len(dados_1[dados_1['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].isnull()])
# Atualiza os valores nulos com a média

media_Weather_Normalized_Site_Natural_Gas_Intensity=round(dados_1['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].mean(),2)



dados_1['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].fillna(media_Weather_Normalized_Site_Natural_Gas_Intensity, inplace = True) 
#Quantidade de registros nulos para o campo em questão

len(dados_1[dados_1['Weather Normalized Source EUI (kBtu/ft²)'].isnull()])
# Atualiza os valores nulos com a média

media_Weather_Normalized_Source_EUI=round(dados_1['Weather Normalized Source EUI (kBtu/ft²)'].mean(),2)



dados_1['Weather Normalized Source EUI (kBtu/ft²)'].fillna(media_Weather_Normalized_Source_EUI, inplace = True) 
# Atualiza os valores nulos com a média

media_Natural_Gas_Use=round(dados_1['Natural Gas Use (kBtu)'].mean(),2)



dados_1['Natural Gas Use (kBtu)'].fillna(media_Natural_Gas_Use, inplace = True)
# Atualiza os valores nulos com a média

media_Weather_Normalized_Site_Natural_Gas_Use=round(dados_1['Weather Normalized Site Natural Gas Use (therms)'].mean(),2)



dados_1['Weather Normalized Site Natural Gas Use (therms)'].fillna(media_Weather_Normalized_Site_Natural_Gas_Use, inplace = True)
# Atualiza os valores nulos com a média

media_Electricity_Use_Grid_Purchase=round(dados_1['Electricity Use - Grid Purchase (kBtu)'].mean(),2)



dados_1['Electricity Use - Grid Purchase (kBtu)'].fillna(media_Electricity_Use_Grid_Purchase, inplace = True)
# Atualiza os valores nulos com a média

media_Weather_Normalized_Site_Electricity=round(dados_1['Weather Normalized Site Electricity (kWh)'].mean(),2)



dados_1['Weather Normalized Site Electricity (kWh)'].fillna(media_Weather_Normalized_Site_Electricity, inplace = True)
# Atualiza os valores nulos com a média

media_Total_GHG_Emissions=round(dados_1['Total GHG Emissions (Metric Tons CO2e)'].mean(),2)



dados_1['Total GHG Emissions (Metric Tons CO2e)'].fillna(media_Total_GHG_Emissions, inplace = True)
# Atualiza os valores nulos com a média

media_Direct_GHG_Emissions=round(dados_1['Direct GHG Emissions (Metric Tons CO2e)'].mean(),2)



dados_1['Direct GHG Emissions (Metric Tons CO2e)'].fillna(media_Direct_GHG_Emissions, inplace = True)
# Atualiza os valores nulos com a média

media_Indirect_GHG_Emissions=round(dados_1['Indirect GHG Emissions (Metric Tons CO2e)'].mean(),2)



dados_1['Indirect GHG Emissions (Metric Tons CO2e)'].fillna(media_Indirect_GHG_Emissions, inplace = True)
# Atualiza os valores nulos com a média

media_Water_Use=round(dados_1['Water Use (All Water Sources) (kgal)'].mean(),2)



dados_1['Water Use (All Water Sources) (kgal)'].fillna(media_Water_Use, inplace = True)
# Atualiza os valores nulos com a média

media_Water_Intensity=round(dados_1['Water Intensity (All Water Sources) (gal/ft²)'].mean(),2)



dados_1['Water Intensity (All Water Sources) (gal/ft²)'].fillna(media_Water_Intensity, inplace = True)
# Atualiza os valores nulos com a média

media_Source_EUI=round(dados_1['Source EUI (kBtu/ft²)'].mean(),2)



dados_1['Source EUI (kBtu/ft²)'].fillna(media_Source_EUI, inplace = True)
dados_1['Water Required?'].describe()
dados_1.groupby('Water Required?').size()
#preenchendo os valores nulos com o valor "Sem Infor"

dados_1['Water Required?']=np.where(dados_1['Water Required?'].isnull(), 'Sem Infor', dados_1['Water Required?'])
dados_1.groupby('Water Required?').size()
dict_Water_Required={'Sem Infor':0,'No':1,'Yes':2}
dados_1=alteraValores(dados_1,dict_Water_Required,'Water Required?')
# Atualiza os valores nulos com a média

media_Latitude=round(dados_1['Latitude'].mean(),2)



dados_1['Latitude'].fillna(media_Latitude, inplace = True)
# Atualiza os valores nulos com a média

media_Longitude =round(dados_1['Longitude'].mean(),2)



dados_1['Longitude'].fillna(media_Longitude, inplace = True)
# Atualiza os valores nulos com a média

media_Community_Board =round(dados_1['Community Board'].mean(),2)



dados_1['Community Board'].fillna(media_Community_Board, inplace = True)
# Atualiza os valores nulos com a média

media_Council_District =round(dados_1['Council District'].mean(),2)



dados_1['Council District'].fillna(media_Council_District, inplace = True)
# Atualiza os valores nulos com a média

media_Census_Tract =round(dados_1['Census Tract'].mean(),2)



dados_1['Census Tract'].fillna(media_Census_Tract, inplace = True)
dados_1.isnull().any()
dados_1.info()
# Find most important features relative to target

print("Find most important features relative to target")

corr = dados_1.corr()

corr.sort_values(["ALVO"], ascending = False, inplace = True)

print(corr.ALVO)
# Correlação de Pearson

dados_1.corr(method = 'pearson')
corr = dados_1.corr()

_ , ax = plt.subplots( figsize =( 30 , 30 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })


colunas=[



'Primary Property Type - Self Selected'

,'Latitude'

,'Longitude'

,'Largest Property Use Type - Gross Floor Area (ft²)'

,'DOF Gross Floor Area'



,'Weather Normalized Site Natural Gas Use (therms)'

,'Total GHG Emissions (Metric Tons CO2e)'

,'Property GFA - Self-Reported (ft²)'

    

,'Weather Normalized Site Electricity Intensity (kWh/ft²)'

]





dados_1.drop(colunas, inplace=True, axis=1)

from sklearn.model_selection import train_test_split

X=dados_1.iloc[:,:-1]

y=dados_1['ALVO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression



# Definir o modelo (Regressão Linear)

modelo_1 = LinearRegression()



# Ajustar o modelo (treinamento)

modelo_1.fit(X_train, y_train)
y_predicted=modelo_1.predict(X_test)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_predicted)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV



ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Melhor alpha :", alpha)


ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 

                cv = 10)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Melhor alpha :", alpha)
y_predicted=ridge.predict(X_test)

mean_absolute_error(y_test, y_predicted)
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Melhor alpha :", alpha)
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Melhor alpha :", alpha)
y_predicted=lasso.predict(X_test)

mean_absolute_error(y_test, y_predicted)
import xgboost as xgb



dtrain = xgb.DMatrix(X_train, label = y_train)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=1000, early_stopping_rounds=500)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=490, max_depth=3, learning_rate=0.075) 

model_xgb.fit(X_train, y_train)
xgb_preds = model_xgb.predict(X_test)

#mean_absolute_error(y_test, xgb_preds)
mean_absolute_error(y_test, xgb_preds)
for i in range(0,len(xgb_preds)):

    if xgb_preds[i]>100:

        xgb_preds[i]=100

    if xgb_preds[i]<1:

        xgb_preds[i]=1
mean_absolute_error(y_test, xgb_preds)
# Carregando arquivo csv usando Pandas

arquivo = '../input/dataset_teste.csv'

dados_sub = pd.read_csv(arquivo)

dados_sub_original = pd.read_csv(arquivo)

print(dados_sub.shape)
dados_sub.info()
colunas=['OrderId'

,'Property Id'

,'Property Name'

,'Parent Property Id'

,'Parent Property Name'

,'BBL - 10 digits'

,'NYC Borough, Block and Lot (BBL) self-reported'

,'NYC Building Identification Number (BIN)'

,'Address 1 (self-reported)'

,'Address 2'

,'Street Number'

,'Street Name'

,'Postal Code'

,'List of All Property Use Types at Property'

,'2nd Largest Property Use Type'

,'2nd Largest Property Use - Gross Floor Area (ft²)'

,'3rd Largest Property Use Type'

,'3rd Largest Property Use Type - Gross Floor Area (ft²)'

,'Metered Areas (Energy)'

,'Metered Areas  (Water)'

,'Fuel Oil #1 Use (kBtu)'

,'Fuel Oil #2 Use (kBtu)'

,'Fuel Oil #4 Use (kBtu)'

,'Fuel Oil #5 & 6 Use (kBtu)'

,'Diesel #2 Use (kBtu)'

,'District Steam Use (kBtu)'

,'Release Date'

,'DOF Benchmarking Submission Status'

,'NTA']

dados_sub.drop(colunas, inplace=True, axis=1)
dados_sub['Weather Normalized Site EUI (kBtu/ft²)'] = dados_sub['Weather Normalized Site EUI (kBtu/ft²)'].apply(converter_numero)

dados_sub['Weather Normalized Site Electricity Intensity (kWh/ft²)'] = dados_sub['Weather Normalized Site Electricity Intensity (kWh/ft²)'].apply(converter_numero)

dados_sub['Weather Normalized Site Natural Gas Intensity (therms/ft²)'] = dados_sub['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].apply(converter_numero)

dados_sub['Weather Normalized Source EUI (kBtu/ft²)'] = dados_sub['Weather Normalized Source EUI (kBtu/ft²)'].apply(converter_numero)

dados_sub['Natural Gas Use (kBtu)'] = dados_sub['Natural Gas Use (kBtu)'].apply(converter_numero)

dados_sub['Weather Normalized Site Natural Gas Use (therms)'] = dados_sub['Weather Normalized Site Natural Gas Use (therms)'].apply(converter_numero)

dados_sub['Electricity Use - Grid Purchase (kBtu)'] = dados_sub['Electricity Use - Grid Purchase (kBtu)'].apply(converter_numero)

dados_sub['Weather Normalized Site Electricity (kWh)'] = dados_sub['Weather Normalized Site Electricity (kWh)'].apply(converter_numero)

dados_sub['Total GHG Emissions (Metric Tons CO2e)'] = dados_sub['Total GHG Emissions (Metric Tons CO2e)'].apply(converter_numero)

dados_sub['Direct GHG Emissions (Metric Tons CO2e)'] = dados_sub['Direct GHG Emissions (Metric Tons CO2e)'].apply(converter_numero)

dados_sub['Indirect GHG Emissions (Metric Tons CO2e)'] = dados_sub['Indirect GHG Emissions (Metric Tons CO2e)'].apply(converter_numero)

dados_sub['Water Use (All Water Sources) (kgal)'] = dados_sub['Water Use (All Water Sources) (kgal)'].apply(converter_numero)

dados_sub['Water Intensity (All Water Sources) (gal/ft²)'] = dados_sub['Water Intensity (All Water Sources) (gal/ft²)'].apply(converter_numero)
#Borough

dados_sub['Borough']=np.where(dados_sub['Borough'].isnull(), 'Sem Infor', dados_sub['Borough'])

dict_translate_Borough={'Sem Infor':0,'Bronx':1,'Brooklyn':2,'Manhattan':3,'Queens':4,'Staten Island':5}

dados_sub=alteraValores(dados_sub,dict_translate_Borough,'Borough')



#DOF Gross Floor Area

media_DOF_Gross_Floor_Area=round(dados_sub['DOF Gross Floor Area'].mean(),2)

dados_sub['DOF Gross Floor Area'].fillna(media_DOF_Gross_Floor_Area, inplace = True)



#Primary Property Type - Self Selected

dados_sub=alteraValores(dados_sub,dict_tanslate_Primary_Property_Type_Self_Selected,'Primary Property Type - Self Selected')



#'Largest Property Use Type'

dados_sub=alteraValores(dados_sub,dict_Largest_Property_Use_Type,'Largest Property Use Type')



# Weather Normalized Site EUI (kBtu/ft²)

dados_sub['Weather Normalized Site EUI (kBtu/ft²)'].fillna(media_Weather_Normalized_Site_EUI, inplace = True) 



# Weather Normalized Site Electricity Intensity (kWh/ft²)

dados_sub['Weather Normalized Site Electricity Intensity (kWh/ft²)'].fillna(media_Weather_Normalized_Site_Electricity_Intensity, inplace = True) 



# Weather Normalized Site Natural Gas Intensity (therms/ft²)

dados_sub['Weather Normalized Site Natural Gas Intensity (therms/ft²)'].fillna(media_Weather_Normalized_Site_Natural_Gas_Intensity, inplace = True) 



# Weather Normalized Source EUI (kBtu/ft²)

dados_sub['Weather Normalized Source EUI (kBtu/ft²)'].fillna(media_Weather_Normalized_Source_EUI, inplace = True) 



# Natural Gas Use (kBtu)

dados_sub['Natural Gas Use (kBtu)'].fillna(media_Natural_Gas_Use, inplace = True)



# Weather Normalized Site Natural Gas Use (therms)

dados_sub['Weather Normalized Site Natural Gas Use (therms)'].fillna(media_Weather_Normalized_Site_Natural_Gas_Use, inplace = True)



# Electricity Use - Grid Purchase (kBtu)

dados_sub['Electricity Use - Grid Purchase (kBtu)'].fillna(media_Electricity_Use_Grid_Purchase, inplace = True)



# Weather Normalized Site Electricity (kWh)

dados_sub['Weather Normalized Site Electricity (kWh)'].fillna(media_Weather_Normalized_Site_Electricity, inplace = True)



# Total GHG Emissions (Metric Tons CO2e)

dados_sub['Total GHG Emissions (Metric Tons CO2e)'].fillna(media_Total_GHG_Emissions, inplace = True)



# Direct GHG Emissions (Metric Tons CO2e)

dados_sub['Direct GHG Emissions (Metric Tons CO2e)'].fillna(media_Direct_GHG_Emissions, inplace = True)



# Indirect GHG Emissions (Metric Tons CO2e)

dados_sub['Indirect GHG Emissions (Metric Tons CO2e)'].fillna(media_Indirect_GHG_Emissions, inplace = True)



# Water Use (All Water Sources) (kgal)

dados_sub['Water Use (All Water Sources) (kgal)'].fillna(media_Water_Use, inplace = True)



# Water Intensity (All Water Sources) (gal/ft²)

dados_sub['Water Intensity (All Water Sources) (gal/ft²)'].fillna(media_Water_Intensity, inplace = True)



# Source EUI (kBtu/ft²)

dados_sub['Source EUI (kBtu/ft²)'].fillna(media_Source_EUI, inplace = True)



# Water Required?

dados_sub['Water Required?']=np.where(dados_sub['Water Required?'].isnull(), 'Sem Infor', dados_sub['Water Required?'])

dados_sub=alteraValores(dados_sub,dict_Water_Required,'Water Required?')



# Latitude

dados_sub['Latitude'].fillna(media_Latitude, inplace = True)



# Longitude

dados_sub['Longitude'].fillna(media_Longitude, inplace = True)



# Community Board

dados_sub['Community Board'].fillna(media_Community_Board, inplace = True)



# Council District

dados_sub['Council District'].fillna(media_Council_District, inplace = True)



# Census Tract

dados_sub['Census Tract'].fillna(media_Census_Tract, inplace = True)
dados_sub.info()
dados_sub.isnull().any()
colunas=[

'Primary Property Type - Self Selected'

,'Latitude'

,'Longitude'

,'Largest Property Use Type - Gross Floor Area (ft²)'

,'DOF Gross Floor Area'

,'Weather Normalized Site Natural Gas Use (therms)'

,'Total GHG Emissions (Metric Tons CO2e)'

,'Property GFA - Self-Reported (ft²)'

,'Weather Normalized Site Electricity Intensity (kWh/ft²)'

]



dados_sub.drop(colunas, inplace=True, axis=1)
y_sub=model_xgb.predict(dados_sub)
for i in range(0,len(y_sub)):

    if y_sub[i]>100:

        y_sub[i]=100

    if y_sub[i]<1:

        y_sub[i]=1
y_sub[25]
dfResult=pd.DataFrame(y_sub, columns=['score'], index=dados_sub_original.index)



frames=[dados_sub_original.iloc[:,1:2], dfResult]

df_final = pd.concat(frames, axis=1, join_axes=[dados_sub_original.index])



df_final
df_final['score']=df_final['score'].round().values.astype(np.int64)
df_final['score']=df_final['score'].abs()  ## Só fiz isso para submeter ....
df_final.to_csv('arquivoFinal12.csv', index=False, sep=',', encoding='utf-8')