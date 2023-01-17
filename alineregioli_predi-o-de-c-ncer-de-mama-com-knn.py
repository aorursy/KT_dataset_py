# Carregar as bibliotecas necessárias: 



import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score 

import sklearn.metrics





# Carregar a base de dados:

rout_path = "../input/data.csv"

dados = pd.read_csv(rout_path)
# Mostrar detalhes dos 5 primeiros registros da base:



dados.head() 
# Colocar no vetor Y os valores da classe objetivo



Y = dados.diagnosis                         





# Fazer a remoção das colunas desnecessárias



list = ['Unnamed: 32','id','diagnosis']        # lista com as colunas a serem removidas

X = dados.drop(list,axis = 1 )          

X.head()

ax = sns.countplot(Y,label="Quantidade")       # M = 212, B = 357

B, M = Y.value_counts()

print('Quantidade de Benignos: ',B)

print('Quantidade de Malignos: ',M)
# mostra soma, média , desvio padrão, min, max, valor dos 25%, 50%(mediana) e 75%



X.describe() 
# Mostrar mapa de calor 



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(X.corr(), annot=True, fmt= '.1f', cmap ='RdYlGn')
droplist_se_worst = ['radius_se', 	'texture_se',	'perimeter_se',	'area_se',	'smoothness_se',	'compactness_se',	'concavity_se',	'concave points_se',	'symmetry_se',	'fractal_dimension_se', 'radius_worst',	'texture_worst',	'perimeter_worst',	'area_worst',	'smoothness_worst',	'compactness_worst',	'concavity_worst',	'concave points_worst',	'symmetry_worst',	'fractal_dimension_worst']



somente_mean = X.drop(droplist_se_worst, axis = 1)

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(somente_mean.corr(), annot=True, linewidths=.5, fmt= '.3f', cmap ='RdYlGn')
droplist_mean_worst = ['radius_mean', 'texture_mean',	'perimeter_mean',	'area_mean', 	'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',	'symmetry_mean',	'fractal_dimension_mean', 'radius_worst',	'texture_worst',	'perimeter_worst',	'area_worst',	'smoothness_worst',	'compactness_worst',	'concavity_worst',	'concave points_worst',	'symmetry_worst',	'fractal_dimension_worst']



somente_se = X.drop(droplist_mean_worst, axis = 1)

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(somente_se.corr(), annot=True, linewidths=.5, fmt= '.3f', cmap ='RdYlGn')
droplist_mean_se = ['radius_mean', 'texture_mean',	'perimeter_mean',	'area_mean', 	'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',	'symmetry_mean',	'fractal_dimension_mean', 'radius_se', 	'texture_se',	'perimeter_se',	'area_se',	'smoothness_se',	'compactness_se',	'concavity_se',	'concave points_se',	'symmetry_se',	'fractal_dimension_se']



somente_worst = X.drop(droplist_mean_se, axis = 1)

f,ax = plt.subplots(figsize=(10, 10))





sns.heatmap(somente_worst.corr(), annot=True, linewidths=.5, fmt= '.3f', cmap ='RdYlGn')
droplist_final = ['radius_mean', 	'perimeter_mean',	'concavity_mean',	'radius_se', 	'perimeter_se',	'radius_worst',	'perimeter_worst']





data_dia = Y

data = X.drop(droplist_final, axis = 1)                     #retirada de atributos 

data_n_2 = (data - data.mean()) / (data.std())              # normalização

data = pd.concat([Y,data_n_2],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="Atributos",

                    value_name='Valores')

plt.figure(figsize=(15,15))

sns.violinplot(x="Atributos", y="Valores", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=90)
# Mostrar correlação entre classes e atributos 'mean'

sns.pairplot(dados, kind="scatter", diag_kind="hist", hue="diagnosis" ,  markers=["o", "D"], vars=["area_mean", "texture_mean", "smoothness_mean", "compactness_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"] ) 

plt.show()
# Mostrar correlação entre classes e atributos 'se'

sns.pairplot(dados, kind="scatter", diag_kind="hist", hue="diagnosis" ,  markers=["o", "D"], vars=["area_se", "texture_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se"] ) 

plt.show()

# Mostrar correlação entre classes e atributos 'worst'

sns.pairplot(dados, kind="scatter", diag_kind="hist", hue="diagnosis" ,  markers=["o", "D"], vars=["area_worst", "texture_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst" , "fractal_dimension_worst"] ) 

plt.show()

# Remoção dos atributos que tinham alta correlação



droplist_final = ['radius_mean', 	'perimeter_mean',	'concavity_mean',	'radius_se', 	'perimeter_se',	'radius_worst',	'perimeter_worst']



data = X.drop(droplist_final, axis = 1)                     #retirada de atributos 





# Separar dados em Treino e Teste 



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=10) # random_state=10 foi mantido para questão de REPRODUCIBILIDADE. (Separar os dados da mesma forma independente da execução). 



print('Quantidade de registros para treino: ', x_train.shape[0]) 

print('Quantidade de registros para teste: ',x_test.shape[0]) #qtd de registros para teste

# Calcular a acurácia para K de 1 a 15 utilizando Cross Validation

tr_acc = []

k_set = range(1,15)



for n_neighbors in k_set:

  knn = KNeighborsClassifier(n_neighbors=n_neighbors)

  scores = cross_val_score(knn, x_train, y_train, cv=10) #testa eficacia com cross validation na base de treinamento

  tr_acc.append(scores.mean())

  

best_k = np.argmax(tr_acc) #retorna o indice do maior

print('Melhor k no treinamento com Cross Validation: ', k_set[best_k]) #mostra melhor k do treinamento com cross validation
te_acc = []

k_set = range(1,15)



for n_neighbors in k_set:

  knn = KNeighborsClassifier(n_neighbors=n_neighbors)

  knn.fit(x_train, y_train)

  y_pred = knn.predict(x_test) #aplica x_test no modelo

  te_acc.append(sklearn.metrics.accuracy_score(y_test, y_pred)) #compara y_test com y_pred

    

melhor_k =np.argmax(te_acc)

print('Melhor k nos testes: ', k_set[melhor_k]) #melhor k do treinamento normal + teste
import matplotlib.pyplot as plt



plt.plot(k_set,tr_acc, label='Treino')

plt.plot(k_set,te_acc, label='Teste')

plt.ylabel('Acurácia')

plt.xlabel('k')

plt.legend()



plt.show()
# Reaplicar o modelo com k=7 , que é o melhor k

clf = KNeighborsClassifier(n_neighbors = 11)

clf.fit(x_train, y_train)





# Mostrar Score

pred_scores = clf.predict_proba(x_test)

print(pred_scores)
# Calcular a acurácia do modelo aplicado nos dados de teste

y_pred = clf.predict(x_test)

te_acc= (sklearn.metrics.accuracy_score(y_test, y_pred)) 

print ('Acurácia obtida: ', te_acc)
conf_mat = sklearn.metrics.confusion_matrix(y_test, y_pred)



df_cm = pd.DataFrame(conf_mat, index = [i for i in ['maligno', 'benigno']],

                  columns = [i for i in ['maligno', 'benigno']])



cmap = sns.light_palette("navy", as_cmap=True)

plt.figure()

sns.heatmap(df_cm, annot=True, cmap=cmap)