# Manipulação e visualização de dados

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 



import warnings

warnings.filterwarnings("ignore")
# Dados normalizados

from scipy.stats import normaltest

# Pré-processamento dos dados

from sklearn.preprocessing import StandardScaler

# Metricas (avaliação do modelo)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
# Modelos de ML

from sklearn.model_selection import GridSearchCV

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier
# Carregar Dados

df_train = pd.read_csv('../input/dataset_treino.csv')

df_test = pd.read_csv('../input/dataset_teste.csv')
# Dimensão

print(df_train.shape)

print(df_test.shape)
# Descrição dos dados

#df_train.info()

#df_train.head()

df_train.describe()
# Identificar existência de observações NAN

print(df_train.isnull().values.any())

print(df_test.isnull().values.any())
# Avaliar distribuição da variável target

df_target = dict(df_train.groupby('classe').size())



names = list(df_target.keys())

values = list(df_target.values())



fig = plt.figure()

ax = fig.add_subplot(111)

ax.bar(names, values)



fig.suptitle("Distribuição da variável 'Classe'")
# Tratar outliers

Q1 = df_train.quantile(0.25)

Q3 = df_train.quantile(0.75)

IQR = Q3 - Q1



df2_train = df_train[~((df_train < (Q1 - 1.5 * IQR)) |(df_train > (Q3 + 1.5 * IQR))).any(axis=1)]

df2_train.shape
# Definir conjunto de dados - X e Y

x_train = df2_train.iloc[:, 1:9].values.astype(float)

y_train = df2_train.iloc[:, 9].values.astype(float)

x_test = df_test.iloc[:, 1:9].values.astype(float)
# Normalizar Dados

scalar_train = StandardScaler().fit(x_train)

x_train_norm = scalar_train.transform(x_train)



scalar_test = StandardScaler().fit(x_test)

x_test_norm = scalar_train.transform(x_test)
# Feature Selection

# Algoritmo Random Forest Classifier



labels = df_train.iloc[:, 1:9].columns



X = x_train_norm

y = y_train



modelo = RandomForestClassifier(n_estimators  = 10)

modelo = modelo.fit(X, y)



# Extraindo a importância

importances = modelo.feature_importances_

indices = np.argsort(importances)



# Obtém os índices

ind=[]

for i in indices:

    ind.append(labels[i])



# Plot da Importância dos Atributos

plt.figure(1)

plt.title('Importância dos Atributos')

plt.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')

plt.yticks(range(len(indices)),ind)

plt.xlabel('Importância Relativa')

plt.show()
df_cor_train = pd.DataFrame(x_train_norm)

df_cor_train.columns = df_train.iloc[:, 1:9].columns

df_cor_train['classe'] = pd.DataFrame(y_train)



correlations = df_cor_train.corr(method = 'pearson')

colunas = df_cor_train.columns



# Plot

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0,len(colunas))

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(colunas)

ax.set_yticklabels(colunas)

plt.show()
select_feature = ['glicose','bmi','idade','indice_historico','classe']

df3_train = df2_train[select_feature]



x2_train = df3_train.iloc[:, 0:4].values.astype(float)

y2_train = df3_train.iloc[:, 4].values.astype(float)



scalar_train = StandardScaler().fit(x2_train)

x2_train_norm = scalar_train.transform(x2_train)
seed = 123

# Separando o array em componentes de input e output

X = x_train_norm

Y = y_train



# Definindo os valores para o número de folds

num_folds = 20

num_instances = len(X)



# Preparando os modelo

modelos = []

modelos.append(('LR', LogisticRegression()))

modelos.append(('KNN', KNeighborsClassifier()))

modelos.append(('SVM', SVC()))

modelos.append(('CART', DecisionTreeClassifier()))

modelos.append(('LDA', LinearDiscriminantAnalysis()))

modelos.append(('AB', AdaBoostClassifier()))

modelos.append(('GBC', GradientBoostingClassifier()))

modelos.append(('RF', RandomForestClassifier()))

modelos.append(('ET', ExtraTreesClassifier()))

modelos.append(('XGB', XGBClassifier()))

modelos.append(('NB', GaussianNB()))

modelos.append(('SGDC', SGDClassifier()))

modelos.append(('MLPC', MLPClassifier()))



# Avaliando cada modelo

resultados = []

nomes = []



for nome, modelo in modelos:

    kfold = cross_validation.KFold(n = num_instances, n_folds = num_folds, random_state = seed)

    cv_results = cross_validation.cross_val_score(modelo, X, Y, cv = kfold, scoring = 'accuracy')

    resultados.append(cv_results)

    nomes.append(nome)

    resultado = "%s: %f (%f)" % (nome, cv_results.mean(), cv_results.std())

    print(resultado)
logreg = LogisticRegression().fit(x_train_norm, y_train)

print("Acurácia: {:.3f}".format(logreg.score(x_train_norm, y_train)))



logreg001 = LogisticRegression(C=0.01).fit(x_train_norm, y_train)

print("Acurácia: {:.3f}".format(logreg001.score(x_train_norm, y_train)))



logreg100 = LogisticRegression(C=100).fit(x_train_norm, y_train)

print("Acurácia: {:.3f}".format(logreg100.score(x_train_norm, y_train)))
training_accuracy = []



neighbors_settings = range(1, 11)



for n_neighbors in neighbors_settings:

    # build the model

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn.fit(x_train_norm, y_train)

    # record training set accuracy

    training_accuracy.append(knn.score(x_train_norm, y_train))





plt.plot(neighbors_settings, training_accuracy, label="training accuracy")

plt.ylabel("Acurácia")

plt.xlabel("n_neighbors")

plt.legend()

plt.savefig('knn_compare_model')
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train_norm, y_train)



print('Acurácia: {:.3f}'.format(knn.score(x_train_norm, y_train)))
svc = SVC(C = 3)

svc.fit(x_train_norm, y_train)



print("Acurácia: {:.3f}".format(svc.score(x_train_norm, y_train)))
tree = DecisionTreeClassifier(max_depth=3, random_state=0)

tree.fit(x_train_norm, y_train)

print("Acurácia: {:.3f}".format(tree.score(x_train_norm, y_train)))
rf1 = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0)

rf1.fit(x_train_norm, y_train)

print("Acurácia: {:.3f}".format(rf1.score(x_train_norm, y_train)))
gb1 = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.2)

gb1.fit(x_train_norm, y_train)



print("Acurácia: {:.3f}".format(gb1.score(x_train_norm, y_train)))
mlp = MLPClassifier(max_iter=100, random_state=0)

mlp.fit(x_train_norm, y_train)



print("Acurácia: {:.3f}".format(mlp.score(x_train_norm, y_train)))
plt.figure(figsize=(20, 5))

plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')

plt.yticks(range(8), df2_train.iloc[:,1:9].columns)

plt.xlabel("Colunas na matriz peso")

plt.ylabel("Atributos de entrada")

plt.colorbar()
# Selecionar melhor modelo - previsão

best_model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0)
best_model.fit(x_train_norm, y_train)
predictions = best_model.predict(x_test_norm)
predictions
plt.hist(predictions)
result = pd.DataFrame()

result['id'] = df_test['id'].astype(int)

result['classe'] = pd.DataFrame(predictions).astype(int)
result.to_csv("submission2.csv", index = False)