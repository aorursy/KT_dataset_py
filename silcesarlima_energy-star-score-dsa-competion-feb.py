# Pacotes utilizados

import pandas as pd

import numpy as np



# Leitura do dataset de treino

df=pd.read_csv("../input/dataset_treino.csv")
# Primeiras linhas do dataset

df.head()
# Dataset possue 60 atributos ou colunas e 6622 linhas ou observações

df.info()
# Total de linhas e colunas

df.shape
df.columns
# Atributo de interesse é ENERGY STAR Score que é explicado pelos demais atributos no dataset

# Porém nem todos atributos são necessários ou relevantes e devem ser eliminados na Análise de Regressão que será feita.

colunas=list(df.columns.values)

colunas
colunas_usadas= ['Property Id',

                 'Site EUI (kBtu/ft²)', 

                 'Weather Normalized Site EUI (kBtu/ft²)',

                 'Weather Normalized Site Electricity Intensity (kWh/ft²)',

                 'Weather Normalized Source EUI (kBtu/ft²)',

                 'Weather Normalized Site Natural Gas Use (therms)', 

                 'Weather Normalized Site Electricity (kWh)',

                 'Total GHG Emissions (Metric Tons CO2e)',

                 'Direct GHG Emissions (Metric Tons CO2e)',

                 'ENERGY STAR Score']
colunas_usadas
df = pd.read_csv("../input/dataset_treino.csv", usecols = colunas_usadas)
df=df[['Property Id',

 'Site EUI (kBtu/ft²)',

 'Weather Normalized Site EUI (kBtu/ft²)',

 'Weather Normalized Site Electricity Intensity (kWh/ft²)',

 'Weather Normalized Source EUI (kBtu/ft²)',

 'Weather Normalized Site Natural Gas Use (therms)',

 'Weather Normalized Site Electricity (kWh)',

 'Total GHG Emissions (Metric Tons CO2e)',

 'Direct GHG Emissions (Metric Tons CO2e)',

 'ENERGY STAR Score']]

df
df=df.replace('Not Available',np.NaN, regex=True)
df
y = df.iloc[:,9].values

X = df.iloc[:, 0:9].values

y
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')

imputer = imputer.fit(X[:,:])

X[:,:] = imputer.transform(X[:,:])
X
type(X)
from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()

X_scaler= scaler_x.fit_transform(X)
X_scaler.shape
previsores_colunas=colunas_usadas[0:9]
previsores_colunas

scaler_y = MinMaxScaler()

y_scaler= scaler_y.fit_transform(y.reshape(-1, 1))
y_scaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaler, y_scaler, test_size = 0.3)
print('X Training Shape:', X_train.shape)

print('y Training Shape:', y_train.shape)

print('X Testing Shape:', X_test.shape)

print('y Testing Shape:', y_test.shape)
from sklearn.ensemble import ExtraTreesRegressor

et=ExtraTreesRegressor(n_estimators=10,random_state=42)

et.fit(X_train,y_train)
et
# Use the et's predict method on the test data

predictions = et.predict(X_test)

# Calculate the absolute errors

errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Dataset Teste

# Leitura apenas dos atributos usados no treinamento

colunas_usadas= ['Property Id',

                 'Site EUI (kBtu/ft²)', 

                 'Weather Normalized Site EUI (kBtu/ft²)',

                 'Weather Normalized Site Electricity Intensity (kWh/ft²)',

                 'Weather Normalized Source EUI (kBtu/ft²)',

                 'Weather Normalized Site Natural Gas Use (therms)', 

                 'Weather Normalized Site Electricity (kWh)',

                 'Total GHG Emissions (Metric Tons CO2e)',

                 'Direct GHG Emissions (Metric Tons CO2e)']



df_teste = pd.read_csv("../input/dataset_teste.csv", usecols = colunas_usadas)
# Total de linhas e colunas

df_teste.shape
# String 'Not Available' é trocada por NaN

df_teste=df_teste.replace('Not Available',np.NaN, regex=True)
X_teste = df_teste.values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')

imputer = imputer.fit(X_teste[:,:])

X_teste[:,:] = imputer.transform(X_teste[:,:])
X_teste
X_teste_scaler= scaler_x.fit_transform(X_teste)
X_teste_scaler.shape

pred_scaler=et.predict(X_teste_scaler)
pred_scaler

pred_scaler=pred_scaler.reshape(-1,1)
real_pred=scaler_y.inverse_transform(pred_scaler)

type(real_pred)


df_final=pd.DataFrame()

df_final= pd.DataFrame.from_records(real_pred.astype(int))
df_final.rename(columns={0: 'score'}, inplace=True)
df_final['Property Id']= df_teste[['Property Id']]
df_final=df_final[['Property Id','score']]
df_final.to_dense().to_csv("submission.csv", index = False, sep=',', encoding='utf-8')