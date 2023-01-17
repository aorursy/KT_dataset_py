import pandas as pd #para lidar com a base de dados

import sklearn as skl #para usar os classificadores k-nn

import matplotlib.pyplot as plt #para plotar gráficos

import seaborn as sns #também para plotar gráficos, mais ligada a datasets

import numpy as np
treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",na_values="?")

#células vazias são preenchidas com ponto de interrogação

treino.shape
treino.tail(n=10) #mostra as últimas 10 linhas do Dataset
treino.describe(include = [np.number]) #incluindo apenas dados numéricos
treino.describe(exclude = [np.number])
treino = treino.drop(['native.country','education'], axis=1) #função drop remove a coluna native.country e education

treino.head()
sns.catplot(x="age", y="income", kind="violin", split=True, data=treino, height=5, palette="Set1", aspect=1)
sns.catplot(x="workclass", hue="income", kind="count", data=treino, height=5, aspect=1, palette="Set1")

plt.xticks(rotation=60)
from sklearn.preprocessing import LabelEncoder



treino2 = treino.copy() #cópia para modificação

#Para estabelecer a correlação com variáveis numéricas, os valores categóricos devem ser transformados também em números

treino2['income'] = LabelEncoder().fit_transform(treino2['income']) #uso da cópia dos dados de treino

treino2['marital.status'] = LabelEncoder().fit_transform(treino2['marital.status']) 

treino2['relationship'] = LabelEncoder().fit_transform(treino2['relationship'])

treino2['race'] = LabelEncoder().fit_transform(treino2['race'])

treino2['sex'] = LabelEncoder().fit_transform(treino2['sex'])
plt.figure(figsize=(10,10))



sns.heatmap(treino2.corr().round(3), square = True, annot=True, vmin=-1, vmax=1, cmap='magma')

plt.show()
sns.catplot(x="fnlwgt", y="income", kind="violin", split=True, data=treino, height=5, palette="Set1", aspect=1)
treino = treino.drop(['fnlwgt'], axis=1) #função drop remove a coluna native.country e education

treino.head()
treino.drop_duplicates(keep='first', inplace=True)
treino.isnull().sum() #mostra o número de valores faltantes por coluna
treino_copia = treino.copy() #cópia dos dados para posterior comparação

treino_copia = treino_copia.dropna()

treino_copia.shape
y_treino_copia = treino_copia.pop("income") #divisão da base de dados 

x_treino_copia = treino_copia
numerical_cols_copia = list(x_treino_copia.select_dtypes(include=[np.number]).columns.values)



numerical_cols_copia.remove('capital.gain')

numerical_cols_copia.remove('capital.loss')



sparse_cols_copia = ['capital.gain','capital.loss']



categorical_cols_copia = list(x_treino_copia.select_dtypes(exclude=[np.number]).columns.values)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='most_frequent') #definição da estratégia: input da moda
from sklearn.preprocessing import OneHotEncoder



one_hot = OneHotEncoder(sparse=False) #configuração 
from sklearn.pipeline import Pipeline



categorical_pipeline = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(drop='if_binary'))])
from sklearn.preprocessing import StandardScaler, RobustScaler



numerical_pipeline = Pipeline(steps = [

    ('scaler', StandardScaler())])

sparse_pipeline = Pipeline(steps = [

    ('scaler', RobustScaler())])
from sklearn.compose import ColumnTransformer



preprocess = ColumnTransformer(transformers = [ #pré-processador

    ('num', numerical_pipeline, numerical_cols_copia),

    ('spr', sparse_pipeline, sparse_cols_copia),

    ('cat', categorical_pipeline, categorical_cols_copia)])



x_treino_copia = preprocess.fit_transform(x_treino_copia) #aplica as transformações
y_treino = treino.pop("income") #divisão da base de dados 

x_treino = treino
numerical_cols = list(x_treino.select_dtypes(include=[np.number]).columns.values)



numerical_cols.remove('capital.gain')

numerical_cols.remove('capital.loss')



sparse_cols = ['capital.gain','capital.loss']



categorical_cols = list(x_treino.select_dtypes(exclude=[np.number]).columns.values)
preprocess1 = ColumnTransformer(transformers = [ #pré-processador

    ('num', numerical_pipeline, numerical_cols),

    ('spr', sparse_pipeline, sparse_cols),

    ('cat', categorical_pipeline, categorical_cols)])



x_treino = preprocess1.fit_transform(x_treino) #aplica as transformações
from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import cross_val_score



knn = KNeighborsClassifier(n_neighbors=10)



score = cross_val_score(knn, x_treino, y_treino, cv = 5, scoring="accuracy") #cv é o número de divisões da validação cruzada

score1 = cross_val_score(knn, x_treino_copia, y_treino_copia, cv = 5, scoring="accuracy")

print(score.mean(), score1.mean())
vizinhos = [25, 30, 32] #lista contendo os valores de k a serem testados



neighbors_scores = {} #dicionário que guarda o desempenho de cada k



for k in vizinhos:

    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), x_treino, y_treino, cv = 5, scoring="accuracy").mean()

    neighbors_scores[k] = score

    #média e acurácia são guardadas



melhor_k = max(neighbors_scores, key=neighbors_scores.get) #variável contendo o melhor valor de k



print("Hiperparâmetro com melhor desempenho: ", melhor_k)

print("Acurácia obtida: ", neighbors_scores[melhor_k])
knn = KNeighborsClassifier(n_neighbors=melhor_k)

knn.fit(x_treino, y_treino) #aplica-se o hiperparâmetro de melhor desempenho ao dataset
teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",na_values="?")

teste = teste.drop(['fnlwgt','education','native.country'], axis=1) #remoção dos atributos definidos como inadequados

teste.head()
x_teste = teste

x_teste = preprocess1.transform(x_teste)



previsao = knn.predict(x_teste)

previsao
submission = pd.DataFrame () #cria-se um data frame



submission[0] = teste.index

submission[1] = previsao

submission.columns = ['Id','income'] #inclusão das colunas Id e previsão no arquivo

submission.to_csv('submission.csv',index = False) #converte o data frame para um arquivo csv

submission.head()