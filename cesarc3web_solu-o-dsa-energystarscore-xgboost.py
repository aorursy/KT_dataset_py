import pandas as pd

import numpy as np

np.random.seed(88)

#

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 100)

import warnings

warnings.filterwarnings('ignore')

#

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
nomes_colunas_treino = ['Order', 'PropertyId', 'PropertyName', 'ParentPropertyId', 'ParentPropertyName', 'BBL10digits', 

       'NYCBoroughBlockandLotBBLSelfReported', 'NYCBuildingIdentificationNumberBIN', 'Address1SelfReported', 'Address2', 

       'PostalCode', 'StreetNumber', 'StreetName','Borough', 'DOFGrossFloorArea', 'PrimaryPropertyTypeSelfSelected', 

       'ListofAllPropertyUseTypesatProperty', 'LargestPropertyUseType', 'LargestPropertyUseTypeGrossFloorArea', 

       '2ndLargestPropertyUseType', '2ndLargestPropertyUseGrossFloorArea', '3rdLargestPropertyUseType', 

       '3rdLargestPropertyUseTypeGrossFloorArea', 'YearBuilt', 'NumberofBuildingsSelfReported', 'Occupancy', 

       'MeteredAreasEnergy', 'MeteredAreasWater', 'ENERGYSTARScore', 'SiteEUI', 'WeatherNormalizedSiteEUI', 

       'WeatherNormalizedSiteElectricityIntensity', 'WeatherNormalizedSiteNaturalGasIntensity', 'WeatherNormalizedSourceEUI', 

       'FuelOil1Use', 'FuelOil2Use', 'FuelOil4Use', 'FuelOil56Use', 'Diesel2Use', 'DistrictSteamUse', 'NaturalGasUse', 

       'WeatherNormalizedSiteNaturalGasUseTherms', 'ElectricityUseGridPurchase', 'WeatherNormalizedSiteElectricity', 

       'TotalGHGEmissionsMetricTonsCO2e', 'DirectGHGEmissionsMetricTonsCO2e', 'IndirectGHGEmissionsMetricTonsCO2e', 

       'PropertyGFASelfReported', 'WaterUseAllWaterSources', 'WaterIntensityAllWaterSources', 'SourceEUI', 'ReleaseDate', 

       'WaterRequired', 'DOFBenchmarkingSubmissionStatus', 'Latitude', 'Longitude', 'CommunityBoard', 'CouncilDistrict', 

       'CensusTract', 'NTA']
dados_treino = pd.read_csv("../input/dataset_treino.csv", header=0, names=nomes_colunas_treino)
dados_treino.head()
dados_treino.shape
dados_treino.info()
corr = dados_treino.corr()

_ , ax = plt.subplots(figsize=(20, 10))

cmap = sns.diverging_palette(220, 10, as_cmap = True)

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax = ax, annot = True, annot_kws = {'fontsize' : 12 })
dados_treino['ENERGYSTARScore'].value_counts().plot(kind='bar', figsize=(20,6))

plt.title('ENERGY STAR Score')

plt.xlabel('Energy Star Score')

plt.ylabel('Frequência')

plt.show()
dados_treino.boxplot(column = 'SiteEUI', by = 'ENERGYSTARScore', figsize=(20,10))

plt.show()
dados_treino.boxplot(column = 'SourceEUI', by = 'ENERGYSTARScore', figsize=(20,10))

plt.show()
dados_treino.boxplot(column = 'DOFGrossFloorArea', by = 'ENERGYSTARScore', figsize=(20,10))

plt.show()
dados_treino.boxplot(column = 'LargestPropertyUseTypeGrossFloorArea', by = 'ENERGYSTARScore', figsize=(20,10))

plt.show()
dados_treino.describe()
dados_treino.isnull().any()
def getBoroughDict():

    borough = {

        1 : "Manhattan",

        2 : "Bronx",

        3 : "Brooklyn",

        4 : "Queens",

        5 : "Staten Island"

    }

    return borough
def trataAtributoBorough(dados):

    print("->", "Tratamento do atributo Borough")

    dados['BBL'] = dados['BBL10digits'].str.replace("\D", "")

    dados['Borough'].fillna(np.NaN)

    boroughDict = getBoroughDict()

    for i in range(len(dados)):

        if(pd.isna(dados['Borough'][i])):

            borough = boroughDict.get(eval(dados['BBL'][i][0]), "Argumento inválido")

            dados['Borough'][i] = borough

    dados.drop(['BBL'], axis=1, inplace=True)

    return dados
def trataAtributoDOFGrossFloorAreaByLargestProperty(dados):

    print("->", "Tratamento do atributo DOF Gross Floor Area")

    unicos = dados['LargestPropertyUseType'].unique()

    for name in unicos:

        if(dados.loc[dados['LargestPropertyUseType'] == name]['DOFGrossFloorArea'].isnull().any()):

            nulos = dados.loc[dados['LargestPropertyUseType'] == name]['DOFGrossFloorArea'].isnull()

            media = dados.loc[dados['LargestPropertyUseType'] == name]['DOFGrossFloorArea'].astype(float).mean(skipna=True)

            dados.loc[dados['DOFGrossFloorArea'].isnull() & 

                      (dados['LargestPropertyUseType'] == name), 'DOFGrossFloorArea'] = media

            print("   Atulizando valores missing para", name)

    return dados
def trataAtributosCategoricosMissing(dados):

    print("->", "Tratamento do atributo com valores missing")

    for name in dados.columns:

        if(dados[name].dtype == 'object'):

            

            valor = dados[name].mode()[0]

            

            lista = ['PropertyName', 'ParentPropertyName', 'ParentPropertyId', 'StreetName', 'StreetNumber', 

                     'BBL10digits', 'NYCBoroughBlockandLotBBLSelfReported', 'Address1SelfReported', 

                     'PrimaryPropertyTypeSelfSelected', 'ListofAllPropertyUseTypesatProperty', 

                     'NYCBuildingIdentificationNumberBIN', 'Address2', '2ndLargestPropertyUseType', 

                     '2ndLargestPropertyUseGrossFloorArea', '3rdLargestPropertyUseType', 

                     '3rdLargestPropertyUseTypeGrossFloorArea', 'FuelOil1Use', 'FuelOil2Use', 'FuelOil4Use', 'FuelOil56Use', 

                     'Diesel2Use', 'DistrictSteamUse', 'ReleaseDate', 'DOFBenchmarkingSubmissionStatus', 'NTA']

            if(name in lista):

                dados.drop([name], axis=1, inplace=True)

                continue

            

            if(name == 'WaterRequired'):

                dados.loc[dados[name].isnull(), name] = 'No'

                continue

                

            if(name == 'MeteredAreasWater'):

                dados.loc[dados[name].isin(['0', 'Not Available']), name] = np.nan

                dados.loc[dados[name].isnull(), name] = 'Another configuration'

                continue

            

            if(name == 'PostalCode'):

                dados['PostalCode'] = dados['PostalCode'].str.replace("\D", "")



            lista = ['WeatherNormalizedSiteEUI', 'WeatherNormalizedSiteElectricityIntensity', 

                     'WeatherNormalizedSiteNaturalGasIntensity', 'WeatherNormalizedSourceEUI', 'NaturalGasUse',

                     'WeatherNormalizedSiteNaturalGasUseTherms', 'ElectricityUseGridPurchase', 

                     'WeatherNormalizedSiteElectricity', 'TotalGHGEmissionsMetricTonsCO2e', 'DirectGHGEmissionsMetricTonsCO2e',

                     'IndirectGHGEmissionsMetricTonsCO2e', 'PropertyGFASelfReported', 'WaterUseAllWaterSources', 

                     'WaterIntensityAllWaterSources']

            if(name in lista):

                dados.loc[dados[name].isin(['0', 'Not Available']), name] = np.nan

                valor = dados[name].astype(float).mean(skipna=True)

                dados.loc[dados[name].isnull(), name] = valor

                print("   Valores missing atualizados para", name)

            

    return dados
def trataAtributosNumericosMissing(dados):

    print("->", "Tratamento do atributo com zeros")

    names = dados._get_numeric_data().columns

    for name in names:

        

        lista = ['Order', 'Latitude', 'Longitude']

        if(name in lista):

            dados.drop([name], axis=1, inplace=True)

            continue

        if(name == 'PostalCode'):

            dados['PostalCode'] = dados['PostalCode'].astype(str).str.slice(stop=5)

        

        if(dados.loc[dados[name] == 0].shape[0] > 0):

            dados.loc[dados[name] == 0, name] = np.nan

            valor = dados[name].astype(float).mean(skipna=True)

            dados.loc[dados[name].isnull(), name] = valor

            print("   Atualizando valores com zeros para", name)

    return dados
def trataAtributosNaN(dados):

    print("->", "Tratamento do atributo com valores nulos")

    for name in dados.columns:

        if(dados[name].isnull().any()):

            if(dados[name].dtype == np.number):

                media = dados[name].astype(float).mean(skipna=True)

                dados.loc[dados[name].isnull(), name] = media

            print("   Atulizando valores nulos para", name)

    return dados
def converteParaNumerico(dados):

    print("->", "Conversão dos atributos para numéricos")

    for name in dados.columns:



        # Converte atributo para numerico

        try:

            dados[name] = pd.to_numeric(dados[name])

            print("Atributo convertido:", name)

        except Exception as error:

            #print("-->", str(error))

            print("   Atributo não convertido", "->", name)

    return dados
def tratarDados(dados):

    print("Entrada:", dados.shape)

    

    # Tratamento para o atributo Bairro

    dados = trataAtributoBorough(dados)

    print("- Trata Bairros:", dados.shape)

    

    # Tratamento de valores missing para DOF Gross Floor Area

    dados = trataAtributoDOFGrossFloorAreaByLargestProperty(dados)

    print("- Trata DOF - Gross Floor Area por Largest Property:", dados.shape)

    

    # Tratamento dos atributos categoricos com valores missing

    dados = trataAtributosCategoricosMissing(dados)

    print("- Trata dados missing para atributos categóricos:", dados.shape)

    

    # Tratamento das colunas numéricas com valores 0 preenchendo com a Moda

    dados = trataAtributosNumericosMissing(dados)

    print("- Trata dados missing para atributos numéricos:", dados.shape)

    

    # Tratamento dos atributos com valores NULOS

    dados = trataAtributosNaN(dados)

    print("Trata atributos nulos:", dados.shape)

    

    # Conversão dos atributos para numéricos

    dados = converteParaNumerico(dados)

    print("Conversão dos atributos para numérico")

    

    print("Saída:", dados.shape)

    return dados
df = dados_treino.copy()

df = tratarDados(df)
df.head()
df.ENERGYSTARScore.value_counts()
df['ENERGYSTARScore'].value_counts().plot(kind='bar', figsize=(20,6))

plt.title('ENERGY STAR Score')

plt.xlabel('Energy Star Score')

plt.ylabel('Frequência')

plt.show()
smote_data = df.drop('ENERGYSTARScore', axis=1)

smote_score = df['ENERGYSTARScore']
categorical_features = smote_data.select_dtypes(exclude=[np.number]).columns

categorical_features
categorical_features_idx = [smote_data.columns.get_loc(c) for c in categorical_features if c in smote_data]
import imblearn

from imblearn.over_sampling import SMOTENC, ADASYN
data_o, target_o = SMOTENC(categorical_features=categorical_features_idx).fit_sample(smote_data, smote_score)
data_o.shape
target_o.shape
import collections

collections.Counter(target_o)
data_o[0]
smote_data.columns
nomes_colunas_treino = smote_data.columns
dados_smoted = pd.DataFrame(data_o, columns=nomes_colunas_treino)
score_smoted = pd.Series(target_o, dtype=np.int64)
df = pd.concat([dados_smoted, score_smoted], axis=1, sort=False)
df.rename(columns={0:'ENERGYSTARScore'}, inplace=True)
df.head()
df['ENERGYSTARScore'].value_counts().plot(kind='bar', figsize=(20,6))

plt.title('ENERGY STAR Score')

plt.xlabel('Energy Star Score')

plt.ylabel('Frequência')

plt.show()
df
X = df.drop(['PropertyId', 'ENERGYSTARScore'], axis=1).copy()

y = df['ENERGYSTARScore'].copy().values
X
y
categorical_features = X.select_dtypes(exclude=[np.number]).columns

categorical_features_idx = [X.columns.get_loc(c) for c in categorical_features if c in X]
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
X = X.values

for idx in categorical_features_idx:

    X[:, idx] = labelencoder.fit_transform(X[:, idx])
X[0]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_scaled[0]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled[0]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=88)
X_train.shape
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.utils import shuffle

from sklearn.metrics import mean_absolute_error
params = {'n_estimators': 500, 'max_depth': 8, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'huber'}

regr = GradientBoostingRegressor(**params)



regr.fit(X_train, y_train)

mae = mean_absolute_error(y_test, regr.predict(X_test))

print("MAE: %.4f" % mae)
# compute test set deviance

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(regr.staged_predict(X_test)):

    test_score[i] = regr.loss_(y_test, y_pred)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, regr.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')
import xgboost as xgb

from sklearn.metrics import explained_variance_score, mean_squared_error

from sklearn.metrics import mean_absolute_error, r2_score, f1_score
xgb_regr = xgb.XGBRegressor(n_estimators=100, 

                            learning_rate=0.1, 

                            gamma=0, 

                            subsample=0.50,

                            colsample_bytree=1, 

                            max_depth=5)
xgb_regr.fit(X_train, y_train)
# MinMaxScaler: 0.7569492977332158 - RMSE: 12.6032

predictions = xgb_regr.predict(X_test)

print(explained_variance_score(predictions, y_test))
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("RMSE: %.4f" % (rmse))
xgb_regr = xgb.XGBRegressor(n_estimators=1000, 

                            learning_rate=0.08, 

                            gamma=0, 

                            subsample=0.75,

                            colsample_bytree=1, 

                            max_depth=10)
xgb_regr.fit(X_train, y_train)
# MinMaxScaler:   0.8891617226022565 - RMSE: 8.9055 - MAE: 5.7404 - R^2: 0.9039

predictions = xgb_regr.predict(X_test)

print(explained_variance_score(predictions, y_test))
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("RMSE: %.4f" % (rmse))
mae = mean_absolute_error(y_test, predictions)

print("MAE: %.4f" % (mae))
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')

print("R^2: %.4f" % (r2))
from sklearn.model_selection import cross_val_predict

from sklearn import linear_model

import matplotlib.pyplot as plt



lr = linear_model.LinearRegression()

#X = dados_smoted

#y = score_smoted



# cross_val_predict returns an array of the same size as `y` where each entry

# is a prediction obtained by cross validation:

predicted = cross_val_predict(lr, X_train, y_train, cv=10)

fig, ax = plt.subplots()

ax.scatter(y_train, predicted, edgecolors=(0, 0, 0))

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
nomes_colunas_teste = ['Order', 'PropertyId', 'PropertyName', 'ParentPropertyId', 'ParentPropertyName', 'BBL10digits', 

       'NYCBoroughBlockandLotBBLSelfReported', 'NYCBuildingIdentificationNumberBIN', 'Address1SelfReported', 'Address2', 

       'PostalCode', 'StreetNumber', 'StreetName','Borough', 'DOFGrossFloorArea', 'PrimaryPropertyTypeSelfSelected', 

       'ListofAllPropertyUseTypesatProperty', 'LargestPropertyUseType', 'LargestPropertyUseTypeGrossFloorArea', 

       '2ndLargestPropertyUseType', '2ndLargestPropertyUseGrossFloorArea', '3rdLargestPropertyUseType', 

       '3rdLargestPropertyUseTypeGrossFloorArea', 'YearBuilt', 'NumberofBuildingsSelfReported', 'Occupancy', 

       'MeteredAreasEnergy', 'MeteredAreasWater', 'SiteEUI', 'WeatherNormalizedSiteEUI', 

       'WeatherNormalizedSiteElectricityIntensity', 'WeatherNormalizedSiteNaturalGasIntensity', 'WeatherNormalizedSourceEUI', 

       'FuelOil1Use', 'FuelOil2Use', 'FuelOil4Use', 'FuelOil56Use', 'Diesel2Use', 'DistrictSteamUse', 'NaturalGasUse', 

       'WeatherNormalizedSiteNaturalGasUseTherms', 'ElectricityUseGridPurchase', 'WeatherNormalizedSiteElectricity', 

       'TotalGHGEmissionsMetricTonsCO2e', 'DirectGHGEmissionsMetricTonsCO2e', 'IndirectGHGEmissionsMetricTonsCO2e', 

       'PropertyGFASelfReported', 'WaterUseAllWaterSources', 'WaterIntensityAllWaterSources', 'SourceEUI', 'ReleaseDate', 

       'WaterRequired', 'DOFBenchmarkingSubmissionStatus', 'Latitude', 'Longitude', 'CommunityBoard', 'CouncilDistrict', 

       'CensusTract', 'NTA']
dados_teste = pd.read_csv("../input/dataset_teste.csv", header=0, names = nomes_colunas_teste)
df_teste = dados_teste.copy()
df_teste.loc[df_teste['NYCBoroughBlockandLotBBLSelfReported'] == 'Not Available']
df_teste.loc[df_teste['PropertyId'] == 5835940, 'NYCBoroughBlockandLotBBLSelfReported'] = '2-03642-00001'

df_teste.loc[df_teste['PropertyId'] == 5835940, 'BBL10digits'] = '2-03642-00001'

#

df_teste.loc[df_teste['PropertyId'] == 5810794, 'NYCBoroughBlockandLotBBLSelfReported'] = '3-06713-00006'

df_teste.loc[df_teste['PropertyId'] == 5810794, 'BBL10digits'] = '3-06713-00006'
df_teste_tratados = tratarDados(df_teste)
df_teste_tratados.isnull().any()
X_teste = df_teste_tratados.drop(['PropertyId'], axis=1).copy()
X_teste.head()
categorical_features = X.select_dtypes(exclude=[np.number]).columns

categorical_features_idx = [X.columns.get_loc(c) for c in categorical_features if c in X]
X = X.values

for idx in categorical_features_idx:

    X[:, idx] = labelencoder.fit_transform(X[:, idx])
X[0]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_teste_scaled = scaler.fit_transform(X)

X_teste_scaled[0]
y_pred = regr.predict(X)
print(y_pred)
y_pred = [1 if num < 1 else 100 if num > 100 else round(num, 0) for num in y_pred]
y_pred = list(map(int, y_pred))
_id = np.array(df_teste_tratados['PropertyId']).astype(int)

solucao_dsa_02_xgboost_03 = pd.DataFrame(y_pred, _id, columns = ["score"])

solucao_dsa_02_xgboost_03.to_csv("solucao_dsa_02_xgboost_03.csv", index_label = ['Property Id'])