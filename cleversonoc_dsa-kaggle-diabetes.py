import pandas as pd
df = pd.read_csv("../input/dataset_treino.csv")
df.head()
df.dtypes
df.describe()
# Distribuição das classes

df.groupby('classe').size()
# Correlação de Pearson

df.corr(method = 'pearson')
# Verificando o skew de cada atributo

df.skew()
import matplotlib.pyplot as plt

# Por se tratar de um conjunto de gráficos menores, pode ser mais interessante gerar os gráficos em janela separada

%matplotlib inline
# Density Plot Univariado

df.plot(kind = 'density', subplots = True, layout = (3,4), sharex = False)

plt.show()
# Box and Whisker Plots

df.plot(kind = 'box', subplots = True, layout = (3,4), sharex = False, sharey = False)

plt.show()
#Matriz de Correlação com nomes das variáveis

correlations = df.corr()

colunas = df.columns



# Plot

import numpy as np

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0, 9, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(colunas)

ax.set_yticklabels(colunas)

plt.show()
import seaborn as sns
# Pairplot

sns.pairplot(df)
# Boxplot com orientação vertical

sns.boxplot(data = df, orient = "v")
# Transformando os dados para a mesma escala (entre 0 e 1)



# Import dos módulos

from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df.shape
df.head()
array = df.values[:,1:10]
array.shape
array[0]
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
# Separando o array em componentes de input e output

array = df.values[:,1:10]



X = array[:,0:8]

Y = array[:,8]
# Criação do modelo

modelo = LogisticRegression()



# RFE

rfe = RFE(modelo, 4)

fit = rfe.fit(X, Y)



# Gerando a nova escala

scaler = MinMaxScaler(feature_range = (0, 1))

X = scaler.fit_transform(X)



# Print dos resultados

print("Número de Atributos: %d" % fit.n_features_)

print(df.columns[1:9])

print("Atributos Selecionados: %s" % fit.support_)

print("Ranking dos Atributos: %s" % fit.ranking_)
fields = ['glicose','pressao_sanguinea','grossura_pele','insulina','bmi','indice_historico']



for field in fields :

    print('campo %s : num entradas-0: %d' % (field, len(df.loc[ df[field] == 0, field ])))
def replace_zero_field(data, field):

    nonzero_vals = data.loc[data[field] != 0, field]

    avg = nonzero_vals.median()

    length = len(data.loc[ data[field] == 0, field])   # num of 0-entries

    data.loc[ data[field] == 0, field ] = avg

    print('Campo: %s; alterado %d entrada com o valor médio: %.3f' % (field,length, avg))



for field in fields :

    replace_zero_field(df,field)

print()

for field in fields :

    print('Campo %s : num entradas-0: %d' % (field, len(df.loc[ df[field] == 0, field ])))
best_features = ['num_gestacoes', 'glicose', 'bmi', 'indice_historico']



df_feat = df[best_features]



X = df_feat.values



# Gerando a nova escala

scaler = MinMaxScaler(feature_range = (0, 1))

X = scaler.fit_transform(X)



# Definindo o tamanho dos dados de treino e de teste

teste_size = 0.20

seed = 7



# Criando o dataset de treino e de teste

X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size = teste_size, random_state = seed)



# Sumarizando os dados transformados

print(X_treino[0:8,:])
X_treino.shape
from sklearn.ensemble import ExtraTreesClassifier
# Criação do Modelo - Feature Selection

modelo = ExtraTreesClassifier()

modelo.fit(X_treino, y_treino)



# Print dos Resultados

print(df.columns[1:9])

print(modelo.feature_importances_)
# Criação do modelo

modelo = LogisticRegression()

modelo.fit(X_treino, y_treino)



# Score

result = modelo.score(X_teste, y_teste)

print("Acurácia: %.3f%%" % (result * 100.0))
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
# Definindo os valores para os folds

num_folds = 10

num_instances = len(X_treino)

seed = 7



# Separando os dados em folds

kfold = model_selection.KFold(num_folds, True, random_state = seed)



# Criando o modelo

modelo = LogisticRegression()

resultado = model_selection.cross_val_score(modelo, X_treino, y_treino, cv = kfold)



modelo.fit(X_treino, y_treino)



# Usamos a média e o desvio padrão

print("Acurácia: %.3f%% (%.3f%%)" % (resultado.mean()*100.0, resultado.std() * 100.0))



# Fazendo previsões

y_pred = modelo.predict(X_teste)

previsoes_lg = [round(value) for value in y_pred]



# Avaliando as previsões

accuracy = accuracy_score(y_teste, previsoes_lg)

print("Acurácia: %.2f%%" % (accuracy * 100.0))

from sklearn import model_selection

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Definindo os valores para o número de folds

num_folds = 10

num_instances = len(X_treino)

seed = 7



# Separando os dados em folds

kfold = model_selection.KFold(num_folds, True, random_state = seed)

cart = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')



# Definindo o número de trees

num_trees = 100



# Criando o modelo

modelo = BaggingClassifier(base_estimator = cart, n_estimators = num_trees, random_state = seed)

resultado = model_selection.cross_val_score(modelo, X_treino, y_treino, cv = kfold)



# Print do resulatdo

print(resultado.mean())



modelo.fit(X_treino, y_treino)



# Fazendo previsões

y_pred = modelo.predict(X_teste)

previsoes_rf = [round(value) for value in y_pred]



# Avaliando as previsões

accuracy = accuracy_score(y_teste, previsoes_rf)

print("Acurácia: %.2f%%" % (accuracy * 100.0))



modelo3 = modelo

import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score

from sklearn import model_selection

from xgboost import XGBClassifier



# Definindo os valores para os folds

num_folds = 10

num_instances = len(X)

seed = 7



# Definindo os valores que serão testados

valores_depth = np.array([1,2,3,4,5])

valores_learning = np.array([1,0.1,0.01,0.001,0.0001,0])

valores_estimator = np.array([100,200,300,400,500])

valores_grid = dict(learning_rate = valores_learning, max_depth = valores_depth, n_estimators=valores_estimator)



print(valores_grid)



# Criando o modelo

modelo = XGBClassifier()



# Criando o grid

grid = model_selection.GridSearchCV(estimator = modelo, param_grid = valores_grid)

grid.fit(X_treino, y_treino)



# Print do resultado

print('Melhor Score:',grid.best_score_)

print('Melhor Learning Rate:',grid.best_estimator_.learning_rate)

print('Melhor Max Depth:',grid.best_estimator_.max_depth)

print('Melhor Estimator:',grid.best_estimator_.n_estimators)



# Fazendo previsões

y_pred = grid.predict(X_teste)

previsoes_gs = [round(value) for value in y_pred]



# Avaliando as previsões

accuracy = accuracy_score(y_teste, previsoes_gs)

print("Acurácia: %.2f%%" % (accuracy * 100.0))



modelo2 = grid

# Definindo os valores para o número de folds

num_folds = 10

num_instances = len(X)

seed = 7



# Separando os dados em folds

kfold = model_selection.KFold(num_folds, True, random_state = seed)



# Criando o modelo

modelo = LinearDiscriminantAnalysis()

resultado = model_selection.cross_val_score(modelo, X_treino, y_treino, cv = kfold)



modelo.fit(X_treino, y_treino)

# Print do resultado

print(resultado.mean())



# Fazendo previsões

y_pred = modelo.predict(X_teste)

previsoes = [round(value) for value in y_pred]



# Avaliando as previsões

accuracy = accuracy_score(y_teste, previsoes)

print("Acurácia: %.2f%%" % (accuracy * 100.0))



modelo1 = modelo
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier



# Separando os dados em folds

kfold = model_selection.KFold(num_folds, True, random_state = seed)



# Criando os sub-modelos

estimators = []



estimators.append(('lda', modelo1))



estimators.append(('xgb', modelo2))



estimators.append(('bagging', modelo3))



# Criando o modelo ensemble

ensemble = VotingClassifier(estimators)

resultado = model_selection.cross_val_score(ensemble, X_treino, y_treino, cv = kfold)



print(resultado.mean())



ensemble.fit(X_treino, y_treino)



# Fazendo previsões

y_pred = ensemble.predict(X_teste)

previsoes = [round(value) for value in y_pred]



# Avaliando as previsões

accuracy = accuracy_score(y_teste, previsoes)

print("Acurácia: %.2f%%" % (accuracy * 100.0))

# Import dos módulos

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
# Dados de treino

n_train = 10

np.random.seed(0)

df_train = X_treino

# Dados de treino

n_valid = 3

np.random.seed(1)

df_valid = X_teste

# Reduzindo a dimensionalidade para 3 componentes

pca = PCA(n_components = 3) 



# Aplicando o PCA aos datasets

newdf_train = pca.fit_transform(df_train)

newdf_valid = pca.transform(df_valid) 



# Gerando novos datasets

features_train = pd.DataFrame(newdf_train)

features_valid = pd.DataFrame(newdf_valid)  



# Cria o modelo de regressão logística

regr = LogisticRegression() 



# Usando o recurso de pipeline do scikit-learn para encadear 2 algoritmos em um mesmo modelo, no caso PCA e Regressão Logística

pipe = Pipeline([('pca', pca), ('logistic', regr)])

pipe.fit(features_train, y_treino)

previsoes_pl = pipe.predict(features_valid)



accuracy = accuracy_score(y_teste, previsoes_pl)

print("Acurácia: %.2f%%" % (accuracy * 100.0))



import keras

import numpy

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

import pandas as pd

import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from keras.callbacks import ModelCheckpoint



#Create first network with Keras 



# Com todas as variáveis

array = df.values[:,1:10]



X = array[:,0:8]

Y = array[:,8]



#best_features = ['num_gestacoes', 'glicose', 'bmi', 'indice_historico']



#df_feat = df[best_features]



#X = df_feat.values



# Gerando a nova escala

scaler = MinMaxScaler(feature_range = (0, 1))

X = scaler.fit_transform(X)



#Rescale X

standardized_X = preprocessing.scale(X)



# Definindo o tamanho dos dados de treino e de teste

teste_size = 0.20

seed = 7



# Criando o dataset de treino e de teste

#X_treino, X_teste, y_treino, y_teste = train_test_split(standardized_X, Y, test_size = teste_size, random_state = seed)



X_treino = standardized_X

y_treino = Y



# Com One Hot Encoder



y_treino_hot = []

for clas in y_treino:

    if clas == 1:

        y_treino_hot.append([1, 0])  # classe 1

    elif clas == 0:

        y_treino_hot.append([0, 1])  # classe 0



y_treino_hot = np.array(y_treino_hot)

        

y_teste_hot = []

for clas in y_teste:

    if clas == 1:

        y_teste_hot.append([1, 0])  # classe 1

    elif clas == 0:

        y_teste_hot.append([0, 1])  # classe 0



y_teste_hot = np.array(y_teste_hot)

        

# create model

model = Sequential()

#model.add(Dense(512, input_dim=8, activation='sigmoid'))

#model.add(Dense(8, activation='sigmoid'))

#model.add(Dropout(0.3))

#model.add(Dense(2, activation='sigmoid'))

#model.add(Dense(12, input_dim=8, init='uniform', activation='sigmoid'))

#model.add(Dense(8, init='uniform', activation='sigmoid'))

#model.add(Dropout(0.3))

#model.add(Dense(2,init='uniform', activation='sigmoid'))

model.add(Dense(512, input_dim=8, init='uniform', activation='sigmoid'))

model.add(Dense(16, init='uniform', activation='relu'))

model.add(Dense(2, init='uniform', activation='sigmoid'))



# Compile model

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])



# checkpoint: store the best model

#ckpt_model = 'neural-weights.best.hdf5'

#checkpoint = ModelCheckpoint(ckpt_model, 

#                            monitor='val_acc',

#                            verbose=2,

#                            save_best_only=True,

#                            mode='max')

#callbacks_list = [checkpoint]



NB_EPOCHS = 1000  # num of epochs to test for

BATCH_SIZE = 16



print('Starting training...')

# train the model, store the results for plotting

history = model.fit(X_treino,

                    y_treino_hot,

#                    validation_data=(X_teste, y_teste_hot),

                    nb_epoch=NB_EPOCHS,

                    batch_size=BATCH_SIZE,

#                    callbacks=callbacks_list,

                    verbose=2)

# Model accuracy

#plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

#plt.title('Model Accuracy')

#plt.ylabel('accuracy')

#plt.xlabel('epoch')

#plt.legend(['train', 'test'])

#plt.show()
# Model Losss

#plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

#plt.title('Model Loss')

#plt.ylabel('loss')

#plt.xlabel('epoch')

#plt.legend(['train', 'test'])

#plt.show()
#model = Sequential()

#model.add(Dense(512, input_dim=8, init='uniform', activation='sigmoid'))

#model.add(Dense(16, init='uniform', activation='relu'))

#model.add(Dense(2, init='uniform', activation='sigmoid'))



# load weights

#model.load_weights("../output/neural-weights.best.hdf5")



# Compile model

#model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])



# estimate accuracy on whole dataset using loaded weights

#scores = model.evaluate(X_teste, y_teste_hot, verbose=0)



#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions

# calculate predictions

#predictions = model.predict_classes(X_teste)    # predicting Y only using X

#print(predictions)



# Round predictions

#rounded = [int(numpy.round(x, 0)) for x in predictions]

#print(rounded)



#result = [1 if x == 0 else 0 for x in rounded]



#print("Rounded type: ", type(rounded)) # rounded is a 'list' class

#print("Shape of rounded: ", len(rounded))

#print("Dataset type: ", type(dataset)) # numpy array?

#print("Shape of dataset: ", dataset.shape)



#accuracy = accuracy_score(y_teste, result)

#print("Acurácia: %.2f%%" % (accuracy * 100.0))

df_valid = pd.read_csv("../input/dataset_teste.csv")
df_valid.head()
for field in fields :

    replace_zero_field(df_valid,field)

print()

for field in fields :

    print('Campo %s : num entradas-0: %d' % (field, len(df_valid.loc[ df_valid[field] == 0, field ])))
# Executar para todas as variáveis



array = df_valid.values[:,1:9]



X_valid = array[:,0:8]



# Gerando a nova escala

scaler = MinMaxScaler(feature_range = (0, 1))

X_valid = scaler.fit_transform(X_valid)



#Rescale X

X_valid = preprocessing.scale(X_valid)
# Executar somente com melhores variáveis



#best_features = ['num_gestacoes', 'glicose', 'bmi', 'indice_historico']



#df_feat = df_valid[best_features]



#X_valid = df_feat.values



# Gerando a nova escala

#scaler = MinMaxScaler(feature_range = (0, 1))

#X_valid = scaler.fit_transform(X_valid)
X_valid.shape
# calculate predictions

predictions = model.predict_classes(X_valid)   # predicting Y only using X

#print(predictions)



# Round predictions

rounded = [int(numpy.round(x, 0)) for x in predictions]

print(rounded)



result = [1 if x == 0 else 0 for x in rounded]

np.array(result)
rounded = result
indice = df_valid['id']
classe = np.array(rounded)
classe.shape
dataset = pd.DataFrame({'id':indice,'classe':classe})
dataset.to_csv('Submission.csv',columns=['id','classe'], index=False)
dataset