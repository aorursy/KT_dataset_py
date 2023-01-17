from PIL import Image , ImageFilter
im = Image . open ( 'etapas.jpeg' )

im
import numpy as np

import pandas as pd 

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

import matplotlib

import sys



import matplotlib.pyplot as plt

from scipy.stats import norm, skew

from scipy.stats.stats import pearsonr
# Utilizando a biblioteca pandas para importação dos dados

train_sf = pd.read_csv('crime_train.csv')
train_sf
test_sf = pd.read_csv('crime_test.csv')
test_sf
train_sf.groupby("Category")["Category"].count().sort_values(ascending=False)
top_crimes = train_sf.Category.value_counts()[:10]

plt.figure(figsize=(12, 8))

pos = np.arange(len(top_crimes))

plt.barh(pos, top_crimes.values, color='blue');

plt.yticks(pos, top_crimes.index);
top_addresses = train_sf.Address.value_counts()[:15]

plt.figure(figsize=(12, 8))



pos = np.arange(len(top_addresses))

plt.bar(pos, top_addresses.values)

plt.xticks(pos, top_addresses.index, rotation = 70)

plt.title('Top 15 localizações com mais crimes')

plt.xlabel('Local')

plt.ylabel('Número de crimes')

plt.show()
top_days = train_sf.DayOfWeek.value_counts()

plt.figure(figsize=(12, 8))



pos = np.arange(len(top_days))

plt.bar(pos, top_days.values)

plt.xticks(pos, top_days.index, rotation = 70)

plt.title('Crimes por dia')

plt.xlabel('Dia da semana')

plt.ylabel('Número de crimes')

plt.show()
train_data = pd.DataFrame(train_sf) 



#renomear colunas para melhor entendimento

train_data = train_data.rename(columns={'Dates': 'Datas', 'Category': 'Categoria', 'Descript': 'Descrição', 'DayOfWeek': 'Dias da Semana', 'PdDistrict': 'Distrito', 'Resolution': 'Resolução', 'Address': 'Endereço', 'X': 'Longitude', 'Y': 'Latitude'})

# Verificando dados iniciais do dataset importado



train_data.head()
test_data = pd.DataFrame(test_sf) 



#renomear colunas para melhor entendimento



test_data = test_data.rename(columns={'Id': 'ID', 'Dates': 'Datas', 'DayOfWeek': 'Dias da Semana', 'PdDistrict': 'Distrito', 'Resolution': 'Resolução', 'Address': 'Endereço', 'X': 'Longitude', 'Y': 'Latitude'})

# Verificando dados iniciais do dataset importado



test_data.head()
# Verificando detalhes estatísticos do dataset



train_data.describe()
# Verificando detalhes estatísticos do dataset



test_data.describe()
# O atributo shape retorna uma tupla, mostrando quantas linhas e colunas temos 

train_data.shape
test_data.shape
# Entendendo o dataset, colunas, data types, quantidade de registros por coluna



train_data.info()
# Entendendo o dataset, colunas, data types, quantidade de registros por coluna



test_data.info()
# Verificar se há algum valor nulo

train_data.isnull().sum()
test_data.isnull().sum()
# identificar o tipo de dados

train_data.dtypes
test_data.dtypes
# Plotar um histograma



plt.hist(train_data['Categoria'])

plt.show()
print(train_data['Categoria'])
# .unique retira os valores repetidos



target = train_data["Categoria"].unique()

print(target)
data_dict = {}

count = 1

for data in target:

    data_dict[data] = count

    count+=1

train_data["Categoria"] = train_data["Categoria"].replace(data_dict)
#Substituindo os dias da semana por números:

data_week_dict = {

    "Monday": 1,

    "Tuesday":2,

    "Wednesday":3,

    "Thursday":4,

    "Friday":5,

    "Saturday":6,

    "Sunday":7

}

train_data["Dias da Semana"] = train_data["Dias da Semana"].replace(data_week_dict)

test_data["Dias da Semana"] = test_data["Dias da Semana"].replace(data_week_dict)
#Substituindo cada distrito por um número:

departamento = train_data["Distrito"].unique()

data_dict_departamento = {}

count = 1

for data in departamento:

    data_dict_departamento[data] = count

    count+=1 

test_data["Distrito"] = test_data["Distrito"].replace(data_dict_departamento)

train_data["Distrito"] = train_data["Distrito"].replace(data_dict_departamento)
print(train_data.head())
columns_train = train_data.columns

print(columns_train)

columns_test = test_data.columns

print(columns_test)
cols = columns_train.drop("Resolução")

print(cols)
train_data_new = train_data[cols]

print(train_data_new.head())
print(train_data_new.describe())
corr = train_data_new.corr()

print(corr["Categoria"])
columns_train_data_new = train_data_new.columns

print(columns_train_data_new)
from sklearn.model_selection import train_test_split





train_df = train_data_new[["Dias da Semana", "Distrito",  "Longitude", "Latitude"]]

X_treino, X_teste, y_treino, y_teste = train_test_split(train_df, train_data_new["Categoria"], random_state=0)
features = ["Dias da Semana", "Distrito",  "Longitude", "Latitude"]
#Importando o modelo KNN

from sklearn.neighbors import KNeighborsClassifier



# Definindo o valor de vizinhos

knn = KNeighborsClassifier(n_neighbors=5)



#Treinando o modelo, com dados de treinamento

knn.fit(X_treino, y_treino)
knn.score(X_teste, y_teste)
predictions = knn.predict(X_teste)
predictions
from collections import OrderedDict

data_dict_new = OrderedDict(sorted(data_dict.items()))

print(data_dict_new)
#print(type(predictions))PARA USAR NO KAGGLE

result_dataframe = pd.DataFrame({

    "Id": test_data["ID"]

})

for key,value in data_dict_new.items():

    result_dataframe[key] = 0

count = 0

for item in predictions:

    for key,value in data_dict.items():

        if(value == item):

            result_dataframe[key][count] = 1

    count+=1

result_dataframe.to_csv("submission_knn.csv", index=False)
knn.predict(X_teste.iloc[0:2])
# Importando métricas para validação do modelo

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Imprimindo a matriz confusa

print("Matriz Confusa: ")

print(confusion_matrix(y_teste, predictions))
# Imprimindo o relatório de classificação

print("Relatório de classificação: \n", classification_report(y_teste, predictions))
# Imprimindo o quão acurado foi o modelo

print('Acurácia do modelo: ' , accuracy_score(y_teste, predictions))
error = []



# Calculating error for K values between 1 and 12

for i in range(1, 12):  

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_treino, y_treino)

    pred_i = knn.predict(X_teste)

    error.append(np.mean(pred_i != y_teste))
plt.figure(figsize=(12, 6))  

plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',  

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')  

plt.xlabel('K Value')  

plt.ylabel('Mean Error')  
# Treinando o modelo KNN com o melhor parâmetro para K



from sklearn.neighbors import KNeighborsClassifier  

classifier = KNeighborsClassifier(n_neighbors=9)  

classifier.fit(X_treino, y_treino)  
# Aplicando os valores de teste novamente

y_pred = classifier.predict(X_teste)
# Importando métricas para validação do modelo

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Imprimindo a matriz confusa

print("Matriz Confusa: ")

print(confusion_matrix(y_teste, y_pred), "\n")
# Imprimindo o relatório de classificação

print("Relatório de classificação: \n", classification_report(y_teste, y_pred))
# Imprimindo o quão acurado foi o modelo

print('Acurácia do modelo: ' , accuracy_score(y_teste, y_pred))
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X_treino, y_treino)
decision_tree.score(X_teste, y_teste)
import matplotlib.pyplot as plt
plot_dataset(decision_tree)