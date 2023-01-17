# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression

from sklearn.metrics import (

                mean_absolute_error, 

                mean_squared_error,

                median_absolute_error,

                max_error,

                r2_score,

                completeness_score,

                fowlkes_mallows_score,

                homogeneity_score,

                v_measure_score )

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df=pd.read_csv('../input/property-sales/raw_sales.csv')



print(df.head())

print(df.isna().sum())

print(df.info())



df_recente = df[(df['datesold'] > '2017-01-01 00:00:00')]

# drop da mesma

df_recente=df_recente.drop('datesold',axis=1)



print(df_recente.head())

print(df_recente.info())



print(df_recente.price.count())
ax = sns.countplot(x="bedrooms", data=df_recente)

for p in ax.patches:

    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
_ = sns.swarmplot(x='propertyType', y='price', data=df_recente)





# Label the axes

_ = plt.xlabel('Tipo de propriedade')

_ = plt.ylabel('Preço')





# Show the plot

plt.show()
_ = sns.swarmplot(x='bedrooms', y='price', data=df_recente)





# Label the axes

_ = plt.xlabel('Quartos')

_ = plt.ylabel('Preço')





# Show the plot

plt.show()
df_recente['propertyType'].replace('unit', 0,inplace=True)

df_recente['propertyType'].replace('house', 1,inplace=True)
# Selecao da coluna a ser treinada

preco = df_recente.price

# drop da mesma

df1=df_recente.drop('price',axis=1)

print(df1.info())

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(df1, preco, test_size = 0.2)



def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
modeloRegrLinear =  LinearRegression().fit(train_X, train_y)

y_pred =  modeloRegrLinear.predict(val_X)

modeloRidge = Ridge().fit(train_X, train_y)

y_predRidge =  modeloRidge.predict(val_X)

modeloLasso = Lasso().fit(train_X, train_y)

y_predLasso =  modeloLasso.predict(val_X)
print("Regressão linear \n\t\t\t\t Acurácia treino: ", modeloRegrLinear.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloRegrLinear.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_pred),

      "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_pred),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_pred),

      "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_pred),

      "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_pred),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_pred)

         )

print("Regressão Ridge \n\t\t\t\t Acurácia treino: ", modeloRidge.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloRidge.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_predRidge),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_predRidge),

     "\n\t\t\t\t R2 score: ", r2_score(val_y,y_predRidge))



print("Regressão Lasso \n\t\t\t\t Acurácia treino: ", modeloLasso.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloLasso.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_predLasso),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_predLasso),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_predLasso),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_predLasso),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_predLasso),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_predLasso))

# Selecao da coluna a ser treinada

tipo = df_recente.propertyType

# drop da mesma

df2=df_recente.drop('propertyType',axis=1)

print(df1.info())

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(df2, tipo, random_state = 1,test_size = 0.2)
modeloRegrLinear =  LinearRegression().fit(train_X, train_y)

y_pred =  modeloRegrLinear.predict(val_X)



modeloRidge = Ridge().fit(train_X, train_y)

y_predRidge =  modeloRidge.predict(val_X)



modeloLogRegression = LogisticRegression().fit(train_X,train_y)

y_predLog =  modeloLogRegression.predict(val_X)
print("Regressão linear \n\t\t\t\t Acurácia treino: ", modeloRegrLinear.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloRegrLinear.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_pred),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_pred),      

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_pred),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_pred),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_pred),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_pred)

         )

print("Regressão Ridge \n\t\t\t\t Acurácia treino: ", modeloRidge.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloRidge.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_predRidge),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_predRidge),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_predRidge))



print("Regressão Regressão Logística \n\t\t\t\t Acurácia treino: ", modeloLogRegression.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloLogRegression.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_predLog),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_predLog),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_predLog),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_predLog),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_predLog),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_predLog))
df_recente_semCEP=df_recente.drop('postcode',axis=1)

# Selecao da coluna a ser treinada

preco_semCEP = df_recente_semCEP.price

# drop da mesma

df1_semCEP=df_recente_semCEP.drop('price',axis=1)

print(df1_semCEP.info())

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(df1_semCEP, preco_semCEP, test_size = 0.2)

modeloRegrLinear =  LinearRegression().fit(train_X, train_y)

y_pred =  modeloRegrLinear.predict(val_X)



modeloRidge = Ridge().fit(train_X, train_y)

y_predRidge =  modeloRidge.predict(val_X)



modeloLasso = Lasso().fit(train_X, train_y)

y_predLasso =  modeloLasso.predict(val_X)

print("Regressão linear \n\t\t\t\t Acurácia treino: ", modeloRegrLinear.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloRegrLinear.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_pred),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_pred),

      "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_pred),

      "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_pred),

      "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_pred),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_pred)

         )

print("Regressão Ridge \n\t\t\t\t Acurácia treino: ", modeloRidge.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloRidge.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_predRidge),      

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_predRidge),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_predRidge),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_predRidge))



print("Regressão Lasso \n\t\t\t\t Acurácia treino: ", modeloLasso.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloLasso.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_predLasso),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_predLasso),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_predLasso),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_predLasso),

    "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_predLasso),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_predLasso))

# Selecao da coluna a ser treinada

tipo_semCEP = df_recente_semCEP.propertyType

# drop da mesma

df2_semCEP=df_recente_semCEP.drop('propertyType',axis=1)

print(df2_semCEP.info())

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(df2_semCEP, tipo_semCEP, random_state = 1,test_size = 0.2)
modeloRegrLinear =  LinearRegression().fit(train_X, train_y)

y_pred =  modeloRegrLinear.predict(val_X)



modeloRidge = Ridge().fit(train_X, train_y)

y_predRidge =  modeloRidge.predict(val_X)



modeloLogRegression = LogisticRegression().fit(train_X,train_y)

y_predLog =  modeloLogRegression.predict(val_X)
print("Regressão linear \n\t\t\t\t Acurácia treino: ", modeloRegrLinear.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloRegrLinear.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_pred),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_pred),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_pred),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_pred),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_pred),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_pred)

         )

print("Regressão Ridge \n\t\t\t\t Acurácia treino: ", modeloRidge.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloRidge.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_predRidge),

      "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_predRidge),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_predRidge),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_predRidge),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_predRidge))



print("Regressão Regressão Logística \n\t\t\t\t Acurácia treino: ", modeloLogRegression.score(train_X,train_y),

     "\n\t\t\t\t Acurácia validação: ", modeloLogRegression.score(val_X,val_y),

     "\n\t\t\t\t Erro médio absoluto: ", mean_absolute_error(val_y,y_predLog),

     "\n\t\t\t\t Erro Percentual médio absoluto: ", mean_absolute_percentage_error(val_y,y_predLog),

     "\n\t\t\t\t Erro médio quadrático: ", mean_squared_error(val_y,y_predLog),

     "\n\t\t\t\t Erro mediano absoluto: ", median_absolute_error(val_y,y_predLog),

     "\n\t\t\t\t Máximo erro: ", max_error(val_y,y_predLog),

      "\n\t\t\t\t R2 score: ", r2_score(val_y,y_predLog))
zoo2 = pd.read_csv("../input/zoo-animals-extended-dataset/zoo2.csv")

zoo3 = pd.read_csv("../input/zoo-animals-extended-dataset/zoo3.csv")
# peeking at the dataset

print(zoo3.head())

#Descriptive stats of the variables in data

print(zoo3.describe())

# verificando se dados nulos

print(zoo3.isna().sum())
animal_name = zoo3.animal_name

class_t = zoo3.class_type

zoo3_Km = zoo3.drop(columns=['animal_name','class_type'],axis=1)
plt.figure(figsize=(10, 8))



wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(zoo3_Km)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Número de Clusters')

plt.ylabel('Coeficiente de Silhueta')

plt.show()
kmeans = KMeans(n_clusters=6, random_state = 0)

kmeans.fit(zoo3_Km)
labels = kmeans.predict(zoo3_Km)

centroides = kmeans.cluster_centers_

own_labels = np.array(['Mamíferos', 'Reptéis e Anfíbios', 'Aves', 'Peixe'])

print("Labels: \n", labels);

print("Centroides: \n", centroides);



print(animal_name)



dummy_data3 = {

        'nome': animal_name,

        'labels': labels}

df_label = pd.DataFrame(dummy_data3, columns = ['nome', 'labels'])

group = df_label.groupby('labels')



#df_label.to_csv('labels.csv')

labels_true = [1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5,3,3,

3,3,3,3,3,3,0,0,0,0,0,0,4,4,4,4,4,4,3,4,4]



print(group.count())



for key, item in group:

    print(group.get_group(key), "\n\n")

tsne_out = TSNE(n_components=2,  n_iter=1000, init='pca').fit_transform(zoo3_Km)

plt.figure(figsize=(13,7))

plt.scatter(tsne_out[:,0], tsne_out[:,1], c=kmeans.labels_,cmap='cividis');
print("Métricas da Cluster \n\t\t\t\t Completude: ", completeness_score(labels_true,labels),

     "\n\t\t\t\t Acurácia fowlkes: ", fowlkes_mallows_score(labels_true,labels),

     "\n\t\t\t\t Homogenidade: ", homogeneity_score(labels_true,labels),

     "\n\t\t\t\t V-measure: ", v_measure_score(labels_true,labels) )
kmeans2 = KMeans(n_clusters=7, random_state = 0)

kmeans2.fit(zoo3_Km)
labels2 = kmeans2.predict(zoo3_Km)

centroides2 = kmeans2.cluster_centers_

print("Labels: \n", labels2);

print("Centroides: \n", centroides2);



dummy_data3 = {

        'nome': animal_name,

        'labels': labels2}

df_label = pd.DataFrame(dummy_data3, columns = ['nome', 'labels'])

group2 = df_label.groupby('labels')



print(group2.count())



for key, item in group2:

    print(group2.get_group(key), "\n\n")

tsne_out = TSNE(n_components=2,  n_iter=1000, init='pca').fit_transform(zoo3_Km)

plt.figure(figsize=(13,7))

plt.scatter(tsne_out[:,0], tsne_out[:,1], c=kmeans2.labels_,cmap='coolwarm');
print("Métricas da Cluster \n\t\t\t\t Completude: ", completeness_score(class_t,labels2),

     "\n\t\t\t\t Acurácia fowlkes: ", fowlkes_mallows_score(class_t,labels2),

     "\n\t\t\t\t Homogenidade: ", homogeneity_score(class_t,labels2),

     "\n\t\t\t\t V-measure: ", v_measure_score(class_t,labels2) )
racing_king_train = pd.read_csv("../input/racingkings/racing_king_train.csv")

racing_king_validate = pd.read_csv("../input/racingkings/racing_king_validate.csv")
print(racing_king_train.head())

print((racing_king_train.info()))