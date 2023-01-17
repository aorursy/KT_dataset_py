# Importar as bibliotecas necessárias para este projeto

import numpy as np

import pandas as pd

from IPython.display import display 



%matplotlib inline

data = pd.read_csv('dataset_treino.csv')

# Êxito

print ('O conjunto de dados possui {} pontos com {} variáveis em cada.'.format(data.shape[0], data.shape[1]))
display(data.head(n = 100))
print(data.columns)
target = data['ENERGY STAR Score']
columns_to_clean = ['ENERGY STAR Score', 'Order', 'Parent Property Id', 'Parent Property Name', 'Property Name', 

                    'NYC Borough, Block and Lot (BBL) self-reported', 'NYC Building Identification Number (BIN)', 

                    'Address 1 (self-reported)', 'Address 2', 'Postal Code', 'Street Number', 'Street Name', 

                    'Latitude', 'Longitude', 'Release Date', 'Property Id',

                    'BBL - 10 digits', 'List of All Property Use Types at Property']
print(data.shape)

data.drop(columns_to_clean, axis = 1, inplace = True)

print(data.shape)
data.replace({'Not Available': 0}, inplace=True)

data = data.fillna(0)
columns_to_encode = ['Borough', 'Primary Property Type - Self Selected',

                     'Metered Areas (Energy)', 'DOF Benchmarking Submission Status',

                     'Water Required?', 'NTA']
def join_columns(data):

    for i in data.index:

        col = 'Metered Areas (Energy)_' + str(data.at[i, 'Metered Areas  (Water)'])

        if col in data.columns:

            data.at[i, col] += 1



    for i in data.index:

        col = 'Primary Property Type - Self Selected_' + str(data.at[i, 'Largest Property Use Type'])

        if col in data.columns:

            data.at[i, col] += 1





    for i in data.index:

        col = 'Primary Property Type - Self Selected_' + str(data.at[i, '2nd Largest Property Use Type'])

        if col in data.columns:

            data.at[i, col] += 1



    for i in data.index:

        col = 'Primary Property Type - Self Selected_' + str(data.at[i, '3rd Largest Property Use Type'])

        if col in data.columns:

            data.at[i, col] += 1

    data.drop(['Metered Areas  (Water)', 'Largest Property Use Type', '2nd Largest Property Use Type', '3rd Largest Property Use Type'], axis = 1, inplace = True)

    return data

data = data.join(pd.get_dummies(data[columns_to_encode]))

data.drop(columns_to_encode, axis = 1, inplace = True)

data.shape
data = join_columns(data)

data = data.astype('float32')

data.shape
display(data.head())
features = data.values

target = target.values / 100
# TODO: Importar 'train_test_split'

from sklearn.model_selection import train_test_split



# TODO: Misturar e separar os dados em conjuntos de treinamento e teste

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=3)



# Êxito

print ("Separação entre treino e teste feita com êxito.")
print('X_train possui {} amostras e {} features.'.format(X_train.shape[0], X_train.shape[1]))

print('y_train possui {} amostras.'.format(y_train.shape[0]))

print('X_test possui {} amostras e {} features.'.format(X_test.shape[0], X_test.shape[1]))

print('y_test possui {} amostras.'.format(y_test.shape[0]))
from sklearn.tree import DecisionTreeRegressor



clf = DecisionTreeRegressor(criterion='mae')

clf.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error as MAE



print(MAE(clf.predict(X_test), y_test))
clf2 = DecisionTreeRegressor(criterion='mae', max_depth=6)

clf2.fit(X_train, y_train)

print(MAE(clf2.predict(X_test), y_test))
data_test = pd.read_csv('dataset_teste.csv')

property_id = data_test['Property Id'].values
columns_to_remove_test = ['OrderId', 'Parent Property Id', 'Parent Property Name', 'Property Name', 

                    'NYC Borough, Block and Lot (BBL) self-reported', 'NYC Building Identification Number (BIN)', 

                    'Address 1 (self-reported)', 'Address 2', 'Postal Code', 'Street Number', 'Street Name', 

                    'Latitude', 'Longitude', 'Release Date', 'Property Id',

                    'BBL - 10 digits', 'List of All Property Use Types at Property']
print(data_test.shape)

data_test.drop(columns_to_remove_test, axis = 1, inplace = True)

print(data_test.shape)

data_test.replace({'Not Available': 0}, inplace=True)

data_test = data_test.fillna(0)
data_test = data_test.join(pd.get_dummies(data_test[columns_to_encode]))

data_test.drop(columns_to_encode, axis = 1, inplace = True)

data_test.shape
data_test = join_columns(data_test)
for column in data.columns:

    if column not in data_test.columns:

        data_test[column] = np.zeros(data_test.shape[0])

for column in data_test.columns:

    if column not in data.columns:

        data_test.drop([column], axis=1, inplace=True)

print(data_test.shape)
data_test = data_test.astype('float32')
#reordena as colunas dos dados de teste com base nas colunas dos dados de treinamento

data_test = data_test[data.columns]
x_test = data_test.values

scores = clf2.predict(x_test)
scores = np.round(np.multiply(scores, 100)).astype(int)
scores = scores.tolist()
len(scores)
preds = pd.DataFrame({'Property Id': property_id, 'Score': scores})

preds.to_csv('submission3.csv', index=False)