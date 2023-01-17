import numpy as np # álgebra Linear

import pandas as pd # processamento de dados, E / S de arquivo CSV (por exemplo, pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



# Todos os resultados que você escreve no diretório atual são salvos como saída.
# ler o  csv (valor separado por vírgula) nos dados



data = pd.read_csv("../input/column_2C_weka.csv")

print(plt.style.available) # veja os estilos de plotagem disponíveis

plt.style.use('ggplot')
# para ver recursos e variável de destino

data.head()
# Bem, a pergunta é: existe algum valor de NaN e comprimento desses dados, então vamos ver as informações

data.info()
data.describe()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x,y)

prediction = knn.predict(x)

print('Predição: {}'.format(prediction))
# teste de treino de divisão

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)



# print ('Predição: {}'.format(prediction))

print('Com KNN (K=3) a acurácia é: ',knn.score(x_test,y_test)) # acurácia
# Complexidade do Modelo

neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []



# Repetir (loop) valores diferentes de k

for i, k in enumerate(neig):

    # k de 1 to 25(excluo)

    knn = KNeighborsClassifier(n_neighbors=k)

    # ajuste com knn

    knn.fit(x_train,y_train)

    # acurácia de treino

    train_accuracy.append(knn.score(x_train, y_train))

    # acurária de teste

    test_accuracy.append(knn.score(x_test, y_test))

    

# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Acurácia de Teste')

plt.plot(neig, train_accuracy, label = 'Acurácia de Treino')

plt.legend()

plt.title('-valor VS Acurácia')

plt.xlabel('Número de "vizinhos"')

plt.ylabel('Acurácia')

plt.xticks(neig)

plt.savefig('grafico.png')

plt.show()

print("A melhor acurária é {} com K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
# create data1: que inclui pelvic_incidence que é feature e sacral_slope que é variável de destino

data1 = data[data['class'] =='Abnormal']

x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)

# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# Regressão Linear

from sklearn.linear_model import LinearRegression

reg = LinearRegression()



# Espaço Preditivo

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)



# Ajuste

reg.fit(x,y)



# Predição

predicted = reg.predict(predict_space)



# R^2

print('Resultado de R^2: ',reg.score(x, y))



# Plotar linha de regressão e dispersão

plt.plot(predict_space, predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# CV

from sklearn.model_selection import cross_val_score

reg = LinearRegression()

k = 5

cv_result = cross_val_score(reg,x,y,cv=k) # usa R ^ 2 como pontuação

print('Pontuações CV: ',cv_result)

print('Média de Pontuações CV: ',np.sum(cv_result)/k)
# O Cume (topo = maior)

from sklearn.linear_model import Ridge

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)

ridge = Ridge(alpha = 0.1, normalize = True)

ridge.fit(x_train,y_train)

ridge_predict = ridge.predict(x_test)

print('Pontuação do Cume: ',ridge.score(x_test,y_test))
# Lasso

from sklearn.linear_model import Lasso

x = np.array(data1.loc[:,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','pelvic_radius']])

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3, test_size = 0.3)

lasso = Lasso(alpha = 0.1, normalize = True)

lasso.fit(x_train,y_train)

ridge_predict = lasso.predict(x_test)

print('Pontuação Lasso: ',lasso.score(x_test,y_test))

print('Coeficientes Lasso: ',lasso.coef_)
# Confusion matrix with random forest

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

rf = RandomForestClassifier(random_state = 4)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print('Matriz de Confusão: \n',cm)

print('Relatório de classificação: \n',classification_report(y_test,y_pred))
# visuzalização com a biblioteca Seaborn

sns.heatmap(cm,annot=True,fmt="d") 

plt.show()
# Curva ROC com regressão logística

from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report



# anormal = 1 e normal = 0

data['class_binary'] = [1 if i == 'Abnormal' else 0 for i in data.loc[:,'class']]

x,y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:,'class_binary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred_prob = logreg.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Traçar a curva ROC

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('Taxa de falsos positivos')

plt.ylabel('Taxa positiva verdadeira')

plt.title('ROC')

plt.show()
# validação cruzada de pesquisa de grid com 1 hiperparâmetro

from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV

knn_cv.fit(x,y)# Ajuste



# Plotar Hiperparâmetro

print("Hiperparâmetro K Tunado: {}".format(knn_cv.best_params_)) 

print("Melhor pontuação: {}".format(knn_cv.best_score_))
# validação cruzada de pesquisa de grade com 2 hiperparâmetros

# 1. o hiperparâmetro é C: parâmetro de regularização da regressão logística

# 2. penalidade l1 ou l2

# Grade de hiperparâmetro

param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 12)

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,param_grid,cv=3)

logreg_cv.fit(x_train,y_train)



# Imprima os parâmetros ideais e a melhor pontuação

print("Hiperparâmetros Tunados : {}".format(logreg_cv.best_params_))

print("Melhor Acurácia: {}".format(logreg_cv.best_score_))
# Carregar dados

data = pd.read_csv('../input/column_2C_weka.csv')



# get_dummies

df = pd.get_dummies(data)

df.head(10)
# apagar (drop) um dos recursos

df.drop("class_Normal",axis = 1, inplace = True) 

df.head(10)

# em vez de dois passos, podemos fazê-lo com um passo pd.get_dummies (data, drop_first = True)
# SVM, pré-processo e pipeline

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

steps = [('scalar', StandardScaler()),

         ('SVM', SVC())]

pipeline = Pipeline(steps)

parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)

cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)

cv.fit(x_train,y_train)



y_pred = cv.predict(x_test)



print("Acurácia: {}".format(cv.score(x_test, y_test)))

print("Parâmetros do Modelo Ajustado: {}".format(cv.best_params_))
# Como você pode ver, não há rótulos nos dados

data = pd.read_csv('../input/column_2C_weka.csv')

plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])

plt.xlabel('pelvic_radius')

plt.ylabel('degree_spondylolisthesis')

plt.show()
# KMeans Clustering

data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data2)

labels = kmeans.predict(data2)

plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)

plt.xlabel('pelvic_radius')

plt.xlabel('degree_spondylolisthesis')

plt.show()
# tabela de tabulação cruzada

df = pd.DataFrame({'labels':labels,"class":data['class']})

ct = pd.crosstab(df['labels'],df['class'])

print(ct)
# Inércia (inertia)

inertia_list = np.empty(8)

for i in range(1,8):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(data2)

    inertia_list[i] = kmeans.inertia_

plt.plot(range(0,8),inertia_list,'-o')

plt.xlabel('Nº de Clusters')

plt.ylabel('Inércia')

plt.show()
data = pd.read_csv('../input/column_2C_weka.csv')

data3 = data.drop('class',axis = 1)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kmeans)

pipe.fit(data3)

labels = pipe.predict(data3)

df = pd.DataFrame({'labels':labels,"class":data['class']})

ct = pd.crosstab(df['labels'],df['class'])

print(ct)
from scipy.cluster.hierarchy import linkage,dendrogram



merg = linkage(data3.iloc[200:220,:],method = 'single')

dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)

plt.show()
from sklearn.manifold import TSNE

model = TSNE(learning_rate=100)

transformed = model.fit_transform(data2)

x = transformed[:,0]

y = transformed[:,1]

plt.scatter(x,y,c = color_list )

plt.xlabel('pelvic_radius')

plt.xlabel('degree_spondylolisthesis')

plt.show()
# PCA

from sklearn.decomposition import PCA

model = PCA()

model.fit(data3)

transformed = model.transform(data3)

print('Componentes principais: ',model.components_)
# Variação de PCA

scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler,pca)

pipeline.fit(data3)



plt.bar(range(pca.n_components_), pca.explained_variance_)

plt.xlabel('Recurso PCA')

plt.ylabel('variação')

plt.show()
# aplicar PCA

pca = PCA(n_components = 2)

pca.fit(data3)

transformed = pca.transform(data3)

x = transformed[:,0]

y = transformed[:,1]

plt.scatter(x,y,c = color_list)

plt.show()