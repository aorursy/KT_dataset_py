# Imports:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
# Importanto a base - dados para treino (train) e teste (test) e, dados mensais de treino (train_mensal) e teste (test_mensal):
train = pd.read_csv("../input/regression_features_train.csv", engine="python", na_values="NaN")
train_mensal = pd.read_csv("../input/regression_targets_train.csv", engine="python")
test = pd.read_csv("../input/regression_features_test.csv", engine="python", na_values="NaN")
test_mensal = pd.read_csv("../input/regression_targets_test.csv", engine="python")
# Tamanho da base
train.shape
# Visualizando amostra da base: 
train.sample(n=10)
# Verificando a existência de dados faltantes (missing data) na base
train.isnull().sum()
# Eliminando os dados faltantes
Xtrain_no_na = train.dropna()

# Verificando o tamanho (número de linhas) da base sem dados faltantes para saber quantas observações foram retiradas
Xtrain_no_na.shape
# Separando features para treino da variável target de treino - DADOS DE HORA EM HORA:
Xtrain = Xtrain_no_na.drop(columns=['precipitation']) #retira coluna com a precipitação medida naquele momento
Ytrain = Xtrain_no_na[['precipitation']]
# A feature 'cbwd', que apresenta a direção do vento média no momento da medição, é categórica e, dessa forma, vale a pena analisar os possíveis valores recebidos na mesma:
Xtrain['cbwd'].value_counts().plot(kind="bar")
Xtrain['cbwd'].value_counts()
# Plotando a precipitação 
plt.scatter(train['cbwd'], train['precipitation'])
plt.ylabel('Precipitation')
plt.xlabel('Combined Wind Direction')
plt.show()

# Calculando a média registrada de precipitação para cada direção
nw = train[(train['cbwd'] == "NW")]['precipitation'].mean()
cv = train[(train['cbwd'] == "cv")]['precipitation'].mean()
ne = train[(train['cbwd'] == "NE")]['precipitation'].mean()
se = train[(train['cbwd'] == "SE")]['precipitation'].mean()
sw = train[(train['cbwd'] == "SW")]['precipitation'].mean()

cbwd = np.asarray(['NW', 'cv', 'NE', 'SE', 'SW'])
avg = np.asarray([nw, cv, ne, se, sw])

comparacao = pd.DataFrame({'CBWD':cbwd[:],'Precipitação - Média':avg[:]})
comparacao
# Observando o comportamento da precipitação registrada em função de cada uma das features
for i in list(Xtrain.columns.values):
    plt.scatter(Xtrain[i], Ytrain)
    plt.ylabel('Precipitation')
    plt.xlabel(i)
    plt.show()
# Histograma de cada uma das variáveis contínuas - distribuição de valores das mesmas
for i in list(Xtrain.columns.values):
    if i != 'cbwd':
        plt.hist(Xtrain[i], color = 'blue', edgecolor = 'black', bins = int(180/5))
        plt.ylabel('Number of observations')
        plt.xlabel(i)
        plt.show()
# Visualização por meio de boxplot
for i in list(Xtrain.columns.values):
    if i != 'cbwd':
        sns.boxplot(x=Xtrain[i])
        plt.show()
# Identificando outliers por Zscore - para isso, a coluna 'cbwd' será removida
Xtrain_new = Xtrain.drop(columns=['cbwd'])
threshold = 3 # zscore acima de 3 serão considerados outliers e removidos
z = np.abs(stats.zscore(Xtrain_new))
rows = (z < threshold).all(axis=1)
Xtrain_o = Xtrain_new[rows] # retirando observações com outliers do dataframe de features
Ytrain_o = Ytrain[rows] #retirando observações com outliers do dataframe de target

Xtrain_o.shape #verificando o número de linhas restantes
# Observando o comportamento da precipitação registrada em função de cada uma das features sem outliers
for i in list(Xtrain_o.columns.values):
    plt.scatter(Xtrain_o[i], Ytrain_o)
    plt.ylabel('Precipitation')
    plt.xlabel(i)
    plt.show()
# Escolhe-se regressor ridge com regularização I2 para diminuir a variancia das estimações
reg = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, 
       max_iter=None, tol=0.001, solver='auto', random_state=None)

# Ajuste do regressor
reg.fit(Xtrain_o, Ytrain_o)
# Preparo da base de teste - retirada da coluna 'cbwd' e, dos dados faltantes
Xtest = test.drop(columns=['cbwd']).dropna()

# Previsões
Ypred = reg.predict(Xtest)

# Verificando número de dados faltantes na base de teste
print(test.isnull().sum())


md = len(test) - len(Xtest) #numero de observações descartadas por conterem missing data
print("O número de observações retiradas da base de teste devido à dados faltantes foi: %d" %md)
# Tamanho do gabarito mensal
test_mensal.shape
Ypred_mensal = np.zeros(120) # Inicializando array de zeros do tamanho das predições mensais (120 posições)
Xtest['preds'] = Ypred #juntanto as predições ao dataframe de features
i = 0
for year in [2014, 2015]:
    for month in range(1,13):
        for city in range(5):
            rows = Xtest[(Xtest['year'] == year) & (Xtest['month'] == month) & (Xtest['city'] == city)].index.tolist()
            Ypred_mensal[i] = Xtest.loc[rows, 'preds'].sum()
            i = i + 1
# RMSE
rmse = sqrt(mean_squared_error(test_mensal['monthly_precipitation'], Ypred_mensal))
rmse
# Importanto a base - dados para treino (Xtrain e Ytrain) e teste (Xtest e Ytest):
Ytrain = pd.read_csv("../input/classification_targets_train.csv", engine="python", na_values="NaN")
Xtrain = pd.read_csv("../input/classification_features_train.csv", engine="python", na_values = "NaN")
Ytest = pd.read_csv("../input/classification_targets_test.csv", engine="python", na_values="NaN")
Xtest = pd.read_csv("../input/classification_features_test.csv", engine="python", na_values="NaN")

Xtrain = Xtrain.drop(columns=['precipitation']) # Retira coluna de precipitação pois essa variável não consta na base de teste
# Separando a coluna target de treino e teste
Y_train = Ytrain['rain']
Y_test = Ytest['rain']
# Imputador
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# Retirando coluna 'cbwd'
Xtrain_prep = Xtrain.drop(columns=['cbwd'])
Xtest_prep = Xtest.drop(columns=['cbwd'])

# Realizando imputação nas bases de treino e teste
Xtrain_imp = pd.DataFrame(imp.fit_transform(Xtrain_prep))
Xtrain_imp.columns = Xtrain_prep.columns.values
Xtest_imp = pd.DataFrame(imp.transform(Xtest_prep))
Xtest_imp.columns = Xtest_prep.columns.values
# Passando os dados de TREINO para o padrão diário apresentado pela variável target
tamanho = len(Ytrain)
season =  np.zeros(tamanho)
dewp =  np.zeros(tamanho)
humi =  np.zeros(tamanho)
pres =  np.zeros(tamanho)
temp =  np.zeros(tamanho)
iws =  np.zeros(tamanho)
city =  np.zeros(tamanho)

i = 0
for y in [2010, 2011, 2012, 2013]:
    for m in range(1,13):
        for d in range(1, 32):
            for cty in range(5):
                rows = Xtrain_imp[(Xtrain_imp['year'] == y) & (Xtrain_imp['month'] == m) & (Xtrain_imp['day'] == d) & (Xtrain_imp['city'] == cty)].index.tolist()
                if rows != []:
                    season[i] = Xtrain_imp.loc[rows, 'season'].mean()
                    dewp[i] = Xtrain_imp.loc[rows, 'DEWP'].mean()
                    humi[i] = Xtrain_imp.loc[rows, 'HUMI'].mean()
                    pres[i] = Xtrain_imp.loc[rows, 'PRES'].mean()
                    temp[i] = Xtrain_imp.loc[rows, 'TEMP'].mean()
                    iws[i] = Xtrain_imp.loc[rows, 'Iws'].mean()
                    i = i + 1

Ytrain['season'] = season
Ytrain['DEWP'] = dewp
Ytrain['HUMI'] = humi
Ytrain['PRES'] = pres
Ytrain['TEMP'] = temp
Ytrain['Iws'] = iws
# Passando os dados de TESTE para o padrão diário apresentado pela variável target
tamanho2 = len(Ytest)
season =  np.zeros(tamanho2)
dewp =  np.zeros(tamanho2)
humi =  np.zeros(tamanho2)
pres =  np.zeros(tamanho2)
temp =  np.zeros(tamanho2)
iws =  np.zeros(tamanho2)
city =  np.zeros(tamanho2)

i = 0
for y in [2014, 2015]:
    for m in range(1,13):
        for d in range(1, 32):
            for cty in range(5):
                rows = Xtest_imp[(Xtest_imp['year'] == y) & (Xtest_imp['month'] == m) & (Xtest_imp['day'] == d) & (Xtest_imp['city'] == cty)].index.tolist()
                if rows != []:
                    season[i] = Xtest_imp.loc[rows, 'season'].mean()
                    dewp[i] = Xtest_imp.loc[rows, 'DEWP'].mean()
                    humi[i] = Xtest_imp.loc[rows, 'HUMI'].mean()
                    pres[i] = Xtest_imp.loc[rows, 'PRES'].mean()
                    temp[i] = Xtest_imp.loc[rows, 'TEMP'].mean()
                    iws[i] = Xtest_imp.loc[rows, 'Iws'].mean()
                    i = i + 1

Ytest['season'] = season
Ytest['DEWP'] = dewp
Ytest['HUMI'] = humi
Ytest['PRES'] = pres
Ytest['TEMP'] = temp
Ytest['Iws'] = iws
# Novas bases de features para treino (X_new_train) e teste (X_new_test)
X_new_train = Ytrain.drop(columns=['rain'])
X_new_test = Ytest.drop(columns=['rain'])
# Verificando a frequência de observações de cada uma das classes na base de treino
Ytrain['rain'].value_counts().plot(kind="bar")
Ytrain['rain'].value_counts()
# Classificador - Regressão Logística
clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                         intercept_scaling=1, class_weight=None, 
                         random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, 
                         warm_start=False, n_jobs=None)
# Realizando seleção de features para verificar o ganho em acurácia
score_list = []
maxscore = 0
featmax = 0
clf = LogisticRegression(C=0.2)
for feat in range (1, 11):
    select = SelectKBest(k=feat)
    Xselect = select.fit_transform(X_new_train, Y_train)
    scores = cross_val_score(clf, Xselect, Y_train, cv=5, scoring = 'accuracy')
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        featmax = feat
plt.plot(np.arange(1, 11),score_list)
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.show()
print(maxscore, featmax) 
# Ajuste do hiperparâmetro C com 5-fold cross validation na base de treino avaliando-se a acurácia
score_list = []
maxscore = 0
cmax = 0
for c in np.arange(0.1, 1.2, 0.1).tolist():
    clf = LogisticRegression(C=c)
    scores = cross_val_score(clf, X_new_train, Y_train, cv=5, scoring = 'accuracy')
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        cmax = c
plt.plot(np.arange(0.1, 1.2, 0.1),score_list)
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.show()
print(maxscore, cmax) 
# Ajustando o classificador com C=0.2
clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.2, fit_intercept=True, 
                         intercept_scaling=1, class_weight=None, 
                         random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, 
                         warm_start=False, n_jobs=None)

clf.fit(X_new_train, Y_train)

#Realizando previsões na base de teste
Ypred_classif = clf.predict(X_new_test)
# Avaliando a área abaixo da curva ROC
roc_auc_score(Y_test, Ypred_classif)
# Importando dados
data = pd.read_csv("../input/clustering_dataset.csv")

# Visualizando amostra dos dados
data.sample(n=10)
# Pode-se realizar uma redução dimensional, utilizando PCA, para melhor visualização dos dados em 2 dimensões e verificação de possíveis clusters
pca = PCA(n_components=2)

data_reduced = pd.DataFrame(pca.fit_transform(data)) #reduzindo para as 2 componentes principais
data_reduced.columns = ['z1', 'z2']
data_reduced
# Visualização dos dados em duas dimensões
plt.scatter(data_reduced['z1'], data_reduced['z2'])
plt.ylabel('z2')
plt.xlabel('z1')
plt.show()
# Removendo outliers
data_reduced_o = data_reduced.drop(data_reduced.index[[10, 17]])

# Visualização dos dados em duas dimensões
plt.scatter(data_reduced_o['z1'], data_reduced_o['z2'])
plt.ylabel('z2')
plt.xlabel('z1')
plt.show()
# Clustering usando K-means (K-médias) - K = 2
clust2 = KMeans(n_clusters = 2)
clust2.fit(data_reduced_o) # Ajuste do método de clustering
label2 = clust2.predict(data_reduced_o) # Obteçao do label de cada ponto

# Visualização
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data_reduced_o['z1'], data_reduced_o['z2'], c=label2, s=50)
ax.set_title('K-Means Clustering - K = 2')
ax.set_xlabel('z1')
ax.set_ylabel('z2')
plt.colorbar(scatter)
# Avaliando por meio do coef de Silhouette
silhouette_score(data_reduced_o, label2, metric='sqeuclidean')
# Clustering usando K-means (K-médias) - K = 3
clust3 = KMeans(n_clusters = 3)
clust3.fit(data_reduced_o) # Ajuste do método de clustering
label3 = clust3.predict(data_reduced_o) # Obteçao do label de cada ponto

# Visualização
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data_reduced_o['z1'], data_reduced_o['z2'], c=label3, s=50)
ax.set_title('K-Means Clustering - K = 3')
ax.set_xlabel('z1')
ax.set_ylabel('z2')
plt.colorbar(scatter)
# Avaliando por meio do coef de Silhouette
silhouette_score(data_reduced_o, label3, metric='sqeuclidean')
# Clustering usando K-means (K-médias) - K = 4
clust4 = KMeans(n_clusters = 4)
clust4.fit(data_reduced_o) # Ajuste do método de clustering
label4 = clust4.predict(data_reduced_o) # Obteçao do label de cada ponto

# Visualização
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data_reduced_o['z1'], data_reduced_o['z2'], c=label4, s=50)
ax.set_title('K-Means Clustering - K = 4')
ax.set_xlabel('z1')
ax.set_ylabel('z2')
plt.colorbar(scatter)
# Avaliando por meio do coef de Silhouette
silhouette_score(data_reduced_o, label4, metric='sqeuclidean')
# Remoção dos pontos 10 e 17
data_new = data.drop(data.index[[10, 17]])

# Ajuste do método de clustering e obtenção dos labels de cada ponto
clust2_new = KMeans(n_clusters = 2)
clust2_new.fit(data_new) # Ajuste do método de clustering
label2_new = clust2_new.predict(data_new) # Obteçao do label de cada ponto
# Cálculo do Silhouette
silhouette_score(data_new, label2_new, metric='sqeuclidean')