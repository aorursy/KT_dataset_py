import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv("../input/atividade-regressao-PMR3508/train.csv", na_values = "NaN")

df_test = pd.read_csv("../input/atividade-regressao-PMR3508/test.csv", na_values = "NaN")
df_train.head()
df_test.head()
train_shape = df_train.shape

print("Número de Linhas do train.csv:", train_shape[0])

print("Número de colunas do train.csv:", train_shape[1])



print("\n")



test_shape = df_test.shape

print("Número de Linhas do test.csv:", test_shape[0])

print("Número de colunas do test.csv:", test_shape[1])
df_train.describe()
df_test.describe()
"""

Função para descobrir a quantidade de dados faltantes em cada coluna

Recebe pandas.dataFrame

Retorna pd.dataFrame com quantidade de dados faltantes por coluna

"""

def missing_data(dataFrame):

    quantity = pd.Series(dataFrame.isnull().sum(), name='qty')

    frequency = pd.Series(100*quantity/(dataFrame.count() + quantity), name = 'freq', dtype='float16')

    missingData = pd.concat([quantity, frequency], axis=1)

    return missingData
missing_data(df_train)
missing_data(df_test)
correlation = df_train.corr()



plt.figure(figsize=(16,16))

matrix = np.triu(correlation)

sns.heatmap(correlation, annot=True, mask = matrix, vmin = -0.5, vmax = 0.5, center = 0, cmap= 'coolwarm')
# Importar package

from sklearn.linear_model import LinearRegression
# Obter entradas

entradas = ["longitude", "latitude", "total_rooms", "population", "households", "median_income"]

df_x = df_train[entradas]



# Obter saída

df_y = df_train["median_house_value"]
# Criar modelo

model_simple = LinearRegression()



# Alocar dados no modelo

model_simple.fit(df_x, df_y)
# Obtenção do coeficiente R**2

r_sq = model_simple.score(df_x, df_y)

print('coefficient of determination:', r_sq)
# Importar package

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
# Obter entradas

entradas = ["longitude", "latitude", "total_rooms", "population", "households", "median_income"]

df_x = df_train[entradas]

x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(df_x)



# Obter saída

df_y = df_train["median_house_value"]
# Criar modelo

model_multiple = LinearRegression().fit(x_, df_y)

r_sq = model_multiple.score(x_, df_y)

intercept, coefficients = model_multiple.intercept_, model_multiple.coef_
print('coefficient of determination:', r_sq)

print('intercept:', intercept)

print('coefficients:', coefficients, sep='\n')
from geopy.geocoders import Nominatim
def get_county(coordinates):

    geolocator = Nominatim(user_agent="California Housing EP")

    location = geolocator.reverse(coordinates)

    address = location.raw['address']

    county = address.get('county', '')

    # print(county)

    return county
# Como é um processo demorado a transformação das coordenadas em endereços, foi utilizado um arquivo .xlsx processado no meu computador

# com os dados já processados



# df_trainn = pd.read_csv("../input/atividade-regressao-PMR3508/train.csv", na_values = "NaN")

# df_trainn["county"] = df_trainn[["latitude","longitude"]].apply(get_county,axis=1)



df_trainn = pd.read_csv("../input/file-name/File_name.csv", na_values = "NaN")
def string2int(columns_name, df_analyse):

    for i in range(len(columns_name)):

        curr_column = df_analyse[columns_name[i]].unique().tolist()

        mapping = dict(zip(curr_column, range(len(curr_column))))

        df_analyse.replace({columns_name[i]: mapping}, inplace = True)

    return df_analyse



# Trocar String county por inteiro

columns_name = ["county"]

df_analyse = df_trainn.copy()

df_analyse = string2int(columns_name, df_analyse)

df_trainn = string2int(columns_name, df_trainn)
# Guardar resultados

# df_trainn.to_excel(r'./County_results.xlsx', index = False)
def rooms_per_house(house):

    total_rooms = house[0]

    households = house[1]

    return total_rooms/households



def house_per_population(house):

    households = house[0]

    population = house[1]

    return households/population



def bedroom_per_room(house):

    total_bedrooms = house[0]

    total_rooms = house[1]

    return total_bedrooms/total_rooms



def extra_rooms(house):

    total_bedrooms = house[0]

    total_rooms = house[1]

    return total_rooms - total_bedrooms



df_trainn["rooms_per_house"] = df_trainn[["total_rooms", "households"]].apply(rooms_per_house,axis=1)

df_trainn["house_per_population"] = df_trainn[["households", "population"]].apply(house_per_population,axis=1)

df_trainn["bedroom_per_room"] = df_trainn[["total_bedrooms", "total_rooms"]].apply(bedroom_per_room,axis=1)

df_trainn["extra_rooms"] = df_trainn[["total_bedrooms", "total_rooms"]].apply(extra_rooms,axis=1)
correlation = df_analyse.corr()



plt.figure(figsize=(16,16))

matrix = np.triu(correlation)

sns.heatmap(correlation, annot=True, mask = matrix, vmin = -0.5, vmax = 0.5, center = 0, cmap= 'coolwarm')
import seaborn

plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="median_income",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="median_age",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="total_rooms",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="total_bedrooms",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="population",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="households",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="rooms_per_house",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="house_per_population",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="bedroom_per_room",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_trainn,x="extra_rooms",y="median_house_value")
plt.figure(figsize=(15, 7))

seaborn.scatterplot(data=df_analyse,x="county",y="median_house_value")
# Trocar String county por inteiro

columns_name = ["county"]

#df_trainn = string2int(columns_name, df_trainn)



# Obter entradas

entradas = ["county", "total_rooms", "population", "households", "median_income", "median_age", "bedroom_per_room"]

df_xn = df_trainn[entradas]



# Obter saída

df_y = df_trainn["median_house_value"]
df_trainn
# Criar modelo

model_simple = LinearRegression()



# Alocar dados no modelo

model_simple.fit(df_xn, df_y)
# Obtenção do coeficiente R**2

r_sq = model_simple.score(df_xn, df_y)

print('coefficient of determination:', r_sq)
xn_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(df_xn)



# Obter saída

df_y = df_train["median_house_value"]



# Criar modelo

model_multiple = LinearRegression().fit(xn_, df_y)

r_sq = model_multiple.score(xn_, df_y)

intercept, coefficients = model_multiple.intercept_, model_multiple.coef_



print('coefficient of determination:', r_sq)

print('intercept:', intercept)

print('coefficients:', coefficients, sep='\n')
# Como é um processo demorado a transformação das coordenadas em endereços, foi utilizado um arquivo .xlsx processado no meu computador

# com os dados já processados



# df_test["county"] = df_test[["latitude","longitude"]].apply(get_county,axis=1)



df_test = pd.read_csv("../input/test-name/test_name.csv", na_values = "NaN")
columns_name = ["county"]

df_test = string2int(columns_name, df_test)
df_test["rooms_per_house"] = df_test[["total_rooms", "households"]].apply(rooms_per_house,axis=1)

df_test["house_per_population"] = df_test[["households", "population"]].apply(house_per_population,axis=1)

df_test["bedroom_per_room"] = df_test[["total_bedrooms", "total_rooms"]].apply(bedroom_per_room,axis=1)

df_test["extra_rooms"] = df_test[["total_bedrooms", "total_rooms"]].apply(extra_rooms,axis=1)
# df_test.to_excel(r'Test_County_results.xlsx', index = False)
entradas = ["county", "total_rooms", "population", "households", "median_income", "median_age", "bedroom_per_room"]

y_pred = model_simple.predict(df_test[entradas])



df_pred = pd.DataFrame({'Id': df_test["Id"], 'median_house_value': y_pred})



# Salvar resultados

df_pred.to_csv("submission.csv", index = False)

#print('predicted response:', y_pred, sep='\n')