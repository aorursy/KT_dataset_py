# Importando as bibliotecas necessárias para análise dos dados

import pandas as pd # Criação e manipulação de dataset

import numpy as np # Manipulação de matrizes

import matplotlib.pyplot as plt # Plotagem de dados

import seaborn as sns # Plotagem e visualização dos dados

from sklearn.preprocessing import MinMaxScaler # Transformação de dados 

from sklearn.model_selection import train_test_split # Partir dados em amostras

from sklearn.neighbors import KNeighborsClassifier # Modelo KNN 

from sklearn.tree import DecisionTreeClassifier # Modelo Árvore de Decisão

from sklearn.ensemble import RandomForestClassifier # Modelo Floresta Aleatória

from sklearn.svm import SVC # Modelo SVM

from sklearn.neural_network import MLPClassifier # Modelo MLP

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Métodos para avaliar acurácia dos modelos
# Criando dataset

df = pd.read_csv("../input/titanic/train_and_test2.csv")



# Visualização inicial do dataset

df.head()
# Número de linhas do Data Frame

df.shape
# Colunas presentes no Data Frame

df.columns
# Avaliando a existência de dados nulos

df.isnull().sum()
# Avaliando valores nulos

df[df.isnull().any(axis=1)]
# Removendo colunas que não serão utilizadas na análise

df.drop(columns=['Passengerid', 'zero', 'zero.1',

       'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7',

       'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13',

       'zero.14', 'zero.15', 'zero.16', 'zero.17',

       'zero.18'], inplace=True)
# Removendo valores nulos para local de embarque

df.dropna(inplace=True)
# Visualizando o novo dataset, somente com as colunas de interesse

df.head()
# Alterar o nome das colunas e das variáveis para que sejam de mais fácil entendimento 

df.columns = ["Idade", "Preço da Passagem", "Sexo", "Familiares", "Dependentes", "Classe", "Local de Embarque", "Sobreviveu"]
# Criação de três faixas etárias para simplificar a correlação e exploração do dataset

conditions = [

    (df['Idade'] <= 12.0),

    (df['Idade'] <= 30.0),

    (df['Idade'] > 30.0)]

choices = [0,1,2]

df['Faixa Etária'] = np.select(conditions, choices, default='black')

df['Faixa Etária'] = df['Faixa Etária'].astype('float64')
# Criação de duas faixas de renda, de acordo com o valor pago pela passagem

conditions = [

    (df['Preço da Passagem'] <= 100.0),

    (df['Preço da Passagem'] > 100.0)]

choices = [0, 1]

df['Classe de Renda'] = np.select(conditions, choices, default='black')

df['Classe de Renda'] = df['Classe de Renda'].astype('float64')
# Avaliando se a nova formatação do dataset está de acordo com o desejado

df.head()
# Colunas finais do Dataframe

df.columns
# Anláise geral dos dados

df.describe()
# Número de Sobreveiventes por Classe

sns.countplot(data=df, x="Sobreviveu", hue="Classe")
# Proporção de sobreviventes por Classe

sobreviventes_classe = df.copy()

sobreviventes_classe['Sobreviveu'] = df['Sobreviveu'] == 1

sns.barplot(data=sobreviventes_classe, y='Sobreviveu', x='Classe', hue="Classe")
# Número de Sobreveiventes por Sexo

sns.countplot(data=df, x="Sobreviveu", hue="Sexo")
# Proporção de sobreviventes por Sexo

sobreviventes_sexo = df.copy()

sobreviventes_sexo['Sobreviveu'] = df['Sobreviveu'] == 1

sns.barplot(data=sobreviventes_sexo, y='Sobreviveu', x='Sexo', hue="Sexo")
# Proporção de sobreviventes por Sexo

sobreviventes_sexo = df.copy()

sobreviventes_sexo['Sobreviveu'] = df['Sobreviveu'] == 1

sns.barplot(data=sobreviventes_sexo, y='Sobreviveu', x='Sexo', hue="Classe")
# Número de Sobreveiventes por Faixa Etária

sns.countplot(data=df, x="Sobreviveu", hue="Faixa Etária")
# Proporção de sobreviventes por Faixa Etária

sobreviventes_faixa = df.copy()

sobreviventes_faixa['Sobreviveu'] = df['Sobreviveu'] == 1

sns.barplot(data=sobreviventes_faixa, y='Sobreviveu', x='Faixa Etária', hue='Sexo')
# Proporção de sobreviventes por Valor Pago na Passagem

sobreviventes_sexo = df.copy()

sobreviventes_sexo['Sobreviveu'] = df['Sobreviveu'] == 1

sns.barplot(data=sobreviventes_sexo, y='Sobreviveu', x='Classe de Renda', hue="Sexo")
# Análise de correlação entre as variáveis

df.corr().style.format('{:.2}').background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
# Normalizando os dados para que a escala não interfira no peso da variável

raw_variables = ['Idade', 'Preço da Passagem', 'Sexo', 'Familiares', 'Dependentes','Classe', 'Local de Embarque', 'Faixa Etária', 'Classe de Renda']

adjusted_variables = MinMaxScaler()
# Definindo as variáveis de Entrada e Saída dos modelos

input_variables = adjusted_variables.fit_transform(df[raw_variables])

output_variables = df['Sobreviveu']
# Criando os conuntos de variáveis que serão partidas em Treino e Teste

x = input_variables   

y = output_variables
# Defininando os dados de Treino (70%) e Teste (30%) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=42) 
print(f'Entradas para treino: {len(x_train)}')

print(f'Entradas para teste: {len(x_test)}')
# Contruíndo o modelo KNN com 20 vizinhos (K=20)

clf_KNN = KNeighborsClassifier(n_neighbors=20)



# Treinando o modelo 

clf_KNN.fit(x_train, y_train)



# Testando o modelo 

y_forecast_KNN = clf_KNN.predict(x_test)
# Construíndo o modelo de Árvore de Decisão

clf_tree = DecisionTreeClassifier(random_state=1)



# Treinando o modelo

clf_tree.fit(x_train, y_train)



# Testando o modelo

y_forecast_tree = clf_tree.predict(x_test)
# Construíndo o modelo Random Forest  

clf_forest = RandomForestClassifier(max_depth=5, random_state=1)



# Treinando o modelo

clf_forest.fit(x_train, y_train)



# Testando o modelol

y_forecast_forest = clf_forest.predict(x_test)
# Construíndo o modelo SVM 

clf_svm=SVC(gamma='auto',random_state=1)



# Treinando o modelo

clf_svm.fit(x_train, y_train)



# Testando o modelo

y_forecast_svm = clf_svm.predict(x_test)
# Construíndo o modelo MLP 

clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(20, 20), random_state=1)



# Treinando o modelo

clf_mlp.fit(x_train, y_train)



# Testando o modelo

y_forecast_mlp = clf_mlp.predict(x_test)
# Comparando a acurácia dos modelos

print(f'Acurácia do Modelo KNN: {accuracy_score(y_test, y_forecast_KNN)*100}')

print(f'Acurácia do Modelo Decision Tree: {accuracy_score(y_test, y_forecast_tree)*100}')

print(f'Acurácia do Modelo Random Forest: {accuracy_score(y_test, y_forecast_forest)*100}')

print(f'Acurácia do Modelo SVM: {accuracy_score(y_test, y_forecast_svm)*100}')

print(f'Acurácia do Modelo MLP: {accuracy_score(y_test, y_forecast_mlp)*100}')