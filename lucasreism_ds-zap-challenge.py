# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.

import json

from pandas.io.json import json_normalize



# Import visualization libs

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
#List output

with open('../input/source-4-ds-train.json', encoding="utf8") as file_data:

    data = file_data.readlines()

    

for i, item in enumerate(data):

    data[i] = json.loads(item)    

    

data_3 = json_normalize(data)    



#Split de treino / teste para validação, falaremos sobre isso mais tarde

data_3, data_3_test = train_test_split(data_3, test_size=0.2, random_state=42)
data_3.head()
data_3.info()
data_3.describe()
# Argumentos: Dataset - o dataset em questão

#             nome_col - a coluna

#             k - variável que controla o peso

def remove_outlier(dataset, nome_col, k):

    q1 = dataset[nome_col].quantile(0.25)

    q3 = dataset[nome_col].quantile(0.75)

    iqr = q3-q1  # Interquartile range

    qbaixo  = q1-k*iqr

    qcima = q3+k*iqr

    dataset_final = dataset.loc[(dataset[nome_col] > qbaixo) & (dataset[nome_col] < qcima)]

    print(qbaixo)

    print(qcima)

    return dataset_final
data_3.boxplot (column='pricingInfos.price', figsize = (12,8))
# Usando a função criada acima

k = 10

for coluna_out in [

                    'parkingSpaces', 'usableAreas', 'suites', 'bathrooms', 'bedrooms', 'totalAreas'

                    ,'pricingInfos.price', 'pricingInfos.monthlyCondoFee', 'pricingInfos.yearlyIptu']:

    data_3 = remove_outlier(data_3, coluna_out, k)

data_3.boxplot (column='pricingInfos.price', figsize = (12,8))
# Chaging some data types

for coluna_type in [

                    'pricingInfos.price', 'pricingInfos.yearlyIptu', 'pricingInfos.rentalTotalPrice'

                    , 'pricingInfos.monthlyCondoFee', 'parkingSpaces'

                    , 'suites', 'bathrooms', 'totalAreas', 'bedrooms'

                    , 'address.geoLocation.location.lon', 'address.geoLocation.location.lat']:

    data_3[coluna_type].astype(float)
data_3 = data_3.drop([

             'address.city', 'address.country',

             'address.district', 'address.geoLocation.precision', 'address.locationId'

            , 'address.neighborhood', 'address.state', 'address.street'

            , 'address.streetNumber'

            , 'address.unitNumber'

            , 'address.zipCode'

            , 'address.zone'

            , 'createdAt'

            , 'description'

            , 'id'

            , 'images'

            , 'listingStatus'

            , 'owner'

            , 'updatedAt'

            , 'title'],axis = 1)
data_3.isnull().sum()
data_3 = data_3.dropna(subset=['address.geoLocation.location.lat'])
data_3.describe()
data_3['usableAreas'].hist(bins=50, figsize=(10,10))
data_3['pricingInfos.price'].hist(bins=50, figsize=(10,10))
#Preço/geolocalização



data_3.plot(kind='scatter', x='address.geoLocation.location.lon', y='address.geoLocation.location.lat', alpha=0.8, 

    s=data_3['pricingInfos.price']/1000000, label='pricingInfos.price', figsize=(12,9), 

    c='pricingInfos.price', cmap=plt.get_cmap('jet'), colorbar=True)
corr_matrix = data_3.corr()
corr_matrix['pricingInfos.price'].sort_values(ascending=False)
# Rapida limpeza do nosso df de teste 

teste_sale = data_3_test[data_3_test['pricingInfos.businessType'].apply(lambda x:x in ['SALE'])]



teste_sale=teste_sale.drop(['pricingInfos.businessType',

                                        'pricingInfos.period','pricingInfos.rentalTotalPrice',

                                        'publicationType','publisherId'], axis=1)



for coluna_med in [

                'bedrooms', 'address.geoLocation.location.lat', 'address.geoLocation.location.lon'

                , 'bathrooms', 'suites', 'parkingSpaces', 'pricingInfos.monthlyCondoFee'

                , 'pricingInfos.monthlyCondoFee', 'pricingInfos.yearlyIptu', 'totalAreas', 'usableAreas']:

    teste_sale[coluna_med] = teste_sale[coluna_med].fillna(teste_sale[coluna_med].median())
teste_rental = data_3_test[data_3_test['pricingInfos.businessType'].apply(lambda x:x in ['RENTAL'])]



teste_rental = teste_rental.drop(['pricingInfos.businessType','pricingInfos.period'

                                  ,'pricingInfos.monthlyCondoFee','pricingInfos.price'

                                  ,'publicationType','publisherId','totalAreas'], axis=1)



for coluna_med in [

                'bedrooms', 'address.geoLocation.location.lat', 'address.geoLocation.location.lon'

                , 'bathrooms', 'suites', 'parkingSpaces', 'pricingInfos.yearlyIptu'

                ,'usableAreas']:

    teste_rental[coluna_med] = teste_rental[coluna_med].fillna(teste_rental[coluna_med].median())
data_3['pricingInfos.period'].value_counts()
#Drop yearly/daily 

data_3 = data_3[~data_3['pricingInfos.period'].str.contains('DAILY', na = False)]

data_3 = data_3[~data_3['pricingInfos.period'].str.contains('YEARLY', na = False)]
# Pegar apenas os registros de venda

data_3_sale = data_3[data_3['pricingInfos.businessType'].apply(lambda x:x in ['SALE'])]
le = LabelEncoder()

le.fit(teste_sale['unitTypes'])

teste_sale['unitTypes'] = le.transform(teste_sale['unitTypes'])

le.fit(data_3_sale['unitTypes'])

data_3_sale['unitTypes'] = le.transform(data_3_sale['unitTypes'])
x_treino = data_3_sale[['address.geoLocation.location.lat','address.geoLocation.location.lon','bathrooms'

                         ,'bedrooms','parkingSpaces','pricingInfos.monthlyCondoFee'

                     ,'pricingInfos.yearlyIptu','suites','unitTypes','usableAreas']]

y_treino = data_3_sale[['pricingInfos.price']]





x_teste = teste_sale[['address.geoLocation.location.lat','address.geoLocation.location.lon','bathrooms'

                         ,'bedrooms','parkingSpaces','pricingInfos.monthlyCondoFee'

                     ,'pricingInfos.yearlyIptu','suites','unitTypes','usableAreas''']]

y_teste = teste_sale[['pricingInfos.price']]
dt = DecisionTreeRegressor()

dt.fit(x_treino,y_treino)

y_pred = dt.predict(x_treino)

mse = mean_squared_error(y_treino, y_pred)

final_mse = np.sqrt(mse)

print(final_mse)



# Results:

# 8.942,45   / DF Treino

# 455.017,95 / DF Teste
rf = RandomForestRegressor()

rf.fit(x_treino,y_treino)

y_pred = rf.predict(x_treino)

mse = mean_squared_error(y_treino, y_pred)

final_mse = np.sqrt(mse)

print(final_mse)



# Results:

# 134.991,01  / DF Treino

# 345.346,02  / DF Teste
parametros = [{'n_estimators':[20, 40, 50], 'max_features': [7, 9]}]

grid_search = GridSearchCV(rf, parametros, cv=5, scoring='neg_mean_squared_error')



grid_search.fit(x_treino,y_treino)

print(grid_search.best_params_)

print(grid_search.best_estimator_)



feature_importances = grid_search.best_estimator_.feature_importances_

print(feature_importances)



# Results 1:

'''rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

                      max_features=7, max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=50,

                      n_jobs=-1, oob_score=False, random_state=None,

                      verbose=0, warm_start=False)'''

                      

# Results 2:

# Top 3 features: 

# 'usableAreas': 0.48659525 

# 'pricingInfos.monthlyCondoFee': 0.17287329

# 'address.geoLocation.location.lon': 0.10273496
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

                      max_features=9, max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=40,

                      n_jobs=-1, oob_score=False, random_state=None,

                      verbose=0, warm_start=False)





rf.fit(x_treino,y_treino)

y_pred = rf.predict(x_treino)

mse = mean_squared_error(y_treino, y_pred)

final_mse = np.sqrt(mse)

print(final_mse)



# Results:

# 330.029,01