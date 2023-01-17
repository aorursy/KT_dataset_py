import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

import plotly.graph_objs as go

from scipy import stats

from scipy.stats import kstest

from scipy.stats import ks_2samp

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')
# Visualizando as 5 primeiras linhas da base de treinamento

train.head()
# Estrutura da base quanto ao número de linhas e colunas: (linhas, colunas)

train.shape
Tab = pd.crosstab(index=train["target"],columns="QTD")

Tabela = pd.concat([Tab,100*(Tab/Tab.sum())],axis=1)

Tabela.columns = ["QTD","%TOTAL"]

Tabela.index = ["NÃO SINISTRO","SINISTRO"]

Tabela    
train.drop_duplicates()

train = train.drop(['id'], axis = 1)

train.shape
colunas = train.columns.tolist()

colunas_reg = [col for col in colunas if 'reg' in col]

colunas_cat = [col for col in colunas if 'cat' in col]

colunas_bin = [col for col in colunas if 'bin' in col]

colunas_car = [col for col in colunas if 'car' in col and 'cat' not in col]

colunas_calc = [col for col in colunas if 'calc' in col]

print(colunas_cat)
train.loc[:,colunas_reg].describe()
continuas = [colunas_reg]

def correl(t):

    correlacao = train[t].corr()

    cmap = sns.diverging_palette(220, 10, as_cmap = True)



    fig, ax = plt.subplots(figsize = (10,10))

    sns.heatmap(correlacao, cmap = cmap, vmax = 1.0, center = 0, fmt = '.2f',

           square = True, linewidths = .5, annot = True, cbar_kws ={"shrink": .75})

    plt.show();

    

# Variáveis reg

for j in continuas:

    correl(j)
r = train.loc[:, colunas_reg]

reg = pd.concat([train.target,train.loc[:, colunas_reg]],axis=1)

#Tabela = reg.pivot_table(index=["target"], aggfunc=np.mean)

#Tabela



df_0 = reg[reg['target'] == 0]

df_1 = reg[reg['target'] == 1]



#stats.ttest_ind(df_0.ps_calc_02,df_1.ps_calc_02).pvalue



var_MP = []

for f1 in r.columns:

    MP = stats.ttest_ind(df_0[f1],df_1[f1]).pvalue

    if MP < 0.05:

        var_MP.append(f1)

        print('Variável {} tem o pvalor {:.2}'.format(f1,MP))

print('Qtd de variáveis com média diferentes {} de um total de {}'.format(len(var_MP),r.shape[1]))



var_KS = []

for f2 in r.columns:

    KSS = ks_2samp(df_0[f2],df_1[f2]).statistic

    KSP = ks_2samp(df_0[f2],df_1[f2]).pvalue

    if KSP < 0.05:

        var_KS.append(f2)

        print('A Variável {} tem um KS de {:.2} com um pvalue de {:.2}'.format(f2,KSS,KSP))

print('Qtd de variáveis com as distribuições diferentes {} de um total de {}'.format(len(var_KS),r.shape[1]))
Tabela = reg.pivot_table(index=["target"], aggfunc=np.mean)

Tabela
train.loc[:, colunas_car].describe()
continuas = [colunas_car]

def correl(t):

    correlacao = train[t].corr()

    cmap = sns.diverging_palette(220, 10, as_cmap = True)



    fig, ax = plt.subplots(figsize = (10,10))

    sns.heatmap(correlacao, cmap = cmap, vmax = 1.0, center = 0, fmt = '.2f',

           square = True, linewidths = .5, annot = True, cbar_kws ={"shrink": .75})

    plt.show();

    

# Variáveis reg

for j in continuas:

    correl(j)
c1 = train.loc[:, colunas_car]

car = pd.concat([train.target,train.loc[:, colunas_car]],axis=1)

df_0 = car[car['target'] == 0]

df_1 = car[car['target'] == 1]



#stats.ttest_ind(df_0.ps_calc_02,df_1.ps_calc_02).pvalue



var_MP = []

for f3 in c1.columns:

    MP = stats.ttest_ind(df_0[f3],df_1[f3]).pvalue

    if MP < 0.05:

        var_MP.append(f3)

        print('Variável {} tem o pvalor {:.2}'.format(f3,MP))

print('Qtd de variáveis com média diferentes {} de um total de {}'.format(len(var_MP),c1.shape[1]))



var_KS = []

for f4 in c1.columns:

    KSS = ks_2samp(df_0[f4],df_1[f4]).statistic

    KSP = ks_2samp(df_0[f4],df_1[f4]).pvalue

    if KSP < 0.05:

        var_KS.append(f4)

        print('A Variável {} tem um KS de {:.2} com um pvalue de {:.2}'.format(f4,KSS,KSP))

print('Qtd de variáveis com as distribuições diferentes {} de um total de {}'.format(len(var_KS),c1.shape[1]))
Tabela = car.pivot_table(index=["target"], aggfunc=np.mean)

Tabela
train.loc[:, colunas_calc].describe()
continuas = [colunas_calc]

def correl(t):

    correlacao = train[t].corr()

    cmap = sns.diverging_palette(220, 10, as_cmap = True)



    fig, ax = plt.subplots(figsize = (10,10))

    sns.heatmap(correlacao, cmap = cmap, vmax = 1.0, center = 0, fmt = '.2f',

           square = True, linewidths = .5, annot = True, cbar_kws ={"shrink": .75})

    plt.show();

    

# Variáveis reg

for j in continuas:

    correl(j)
c2 = train.loc[:, colunas_calc]

calc = pd.concat([train.target,train.loc[:, colunas_calc]],axis=1)

#Tabela = calc.pivot_table(index=["target"], aggfunc=np.mean)

#Tabela



df_0 = calc[calc['target'] == 0]

df_1 = calc[calc['target'] == 1]



#stats.ttest_ind(df_0.ps_calc_02,df_1.ps_calc_02).pvalue



var_MP = []

for f5 in c2.columns:

    MP = stats.ttest_ind(df_0[f5],df_1[f5]).pvalue

    if MP < 0.05:

        var_MP.append(f5)

        print('Variável {} tem o pvalor {:.2}'.format(f5,MP))

print('Qtd de variáveis com média diferentes {} de um total de {}'.format(len(var_MP),c2.shape[1]))



var_KS = []

for f6 in c2.columns:

    KSS = ks_2samp(df_0[f6],df_1[f6]).statistic

    KSP = ks_2samp(df_0[f6],df_1[f6]).pvalue

    if KSP < 0.05:

        var_KS.append(f6)

        print('A Variável {} tem um KS de {:.2} com um pvalue de {:.2}'.format(f6,KSS,KSP))

print('Qtd de variáveis com as distribuições diferentes {} de um total de {}'.format(len(var_KS),c2.shape[1]))
Tabela = calc.pivot_table(index=["target"], aggfunc=np.mean)

Tabela
var_missing = []



for f in train.columns:

    missings = train[train[f] == -1][f].count()

    if missings > 0:

        var_missing.append(f)

        missings_perc = missings/train.shape[0]

        

        print('Variável {} tem {} exemplos ({:.2%}) com valores omissos'.format(f, missings, missings_perc))

        

print('No total, existem {} variáveis com valores omissos'.format(len(var_missing)))
variaveis_excluir = ['ps_car_03_cat', 'ps_car_05_cat']

train.drop(variaveis_excluir, inplace=True, axis=1)

train.drop(colunas_calc, inplace=True, axis=1)
train.shape
cat = pd.concat([train.target,train.loc[:, colunas_cat]],axis=1)

tab = 100*(cat.pivot_table(index=["ps_car_02_cat"], values = ['target'],aggfunc=[np.mean]))

tab2= pd.crosstab(cat["ps_car_02_cat"],cat["target"])

tab3= tab2[0]+tab2[1]

tab4= 100*(tab3/tab3.sum())



Tabela = pd.concat([tab2,tab3,tab4,tab],axis=1)

Tabela.columns = ["N Sinistro","Sinistro","Total","%Total","%Taxa Sinistro"]

Tabela
c3 = train.loc[:, colunas_cat]

cat = pd.concat([train.target,train.loc[:, colunas_cat]],axis=1)

#Tabela = calc.pivot_table(index=["target"], aggfunc=np.mean)

#Tabela



df_0 = cat[cat['target'] == 0]

df_1 = cat[cat['target'] == 1]



var_KS = []

for f7 in c3.columns:

    KSS = ks_2samp(df_0[f7],df_1[f7]).statistic

    KSP = ks_2samp(df_0[f7],df_1[f7]).pvalue

    if KSP < 0.05:

        var_KS.append(f7)

        print('A Variável {} tem um KS de {:.2} com um pvalue de {:.2}'.format(f7,KSS,KSP))

print('Qtd de variáveis com as distribuições diferentes {} de um total de {}'.format(len(var_KS),c3.shape[1]))
c4 = train.loc[:, colunas_bin]

bbin = pd.concat([train.target,train.loc[:, colunas_bin]],axis=1)

#Tabela = calc.pivot_table(index=["target"], aggfunc=np.mean)

#Tabela



df_0 = bbin[bbin['target'] == 0]

df_1 = bbin[bbin['target'] == 1]



var_KS = []

for f8 in c4.columns:

    KSS = ks_2samp(df_0[f8],df_1[f8]).statistic

    KSP = ks_2samp(df_0[f8],df_1[f8]).pvalue

    if KSP < 0.05:

        var_KS.append(f8)

        print('A Variável {} tem um KS de {:.2} com um pvalue de {:.2}'.format(f8,KSS,KSP))

print('Qtd de variáveis com as distribuições diferentes {} de um total de {}'.format(len(var_KS),c4.shape[1]))
colunas_cat.remove('ps_car_03_cat')

colunas_cat.remove('ps_car_05_cat')

print(colunas_cat)
for i in colunas_cat:

    plt.figure()

    fig, ax = plt.subplots(figsize = (20,10))

    sns.barplot(ax = ax, x = i, y = 'target', data = train)

    plt.ylabel('% target', fontsize = 18)

    plt.xlabel(i, fontsize = 18)

    plt.tick_params(axis = 'both', which= 'major', labelsize = 18)

    plt.show()
# Imputando com a média e a moda

media_imp = Imputer(missing_values=-1, strategy='mean', axis=0)

moda_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

train['ps_reg_03'] = media_imp.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_14'] = media_imp.fit_transform(train[['ps_car_14']]).ravel()

train['ps_car_11'] = moda_imp.fit_transform(train[['ps_car_11']]).ravel()

train.shape
for i in train.columns:

    if train[i].dtype == 'int64' and i != 'target':

        train[i] = train[i].astype('category')

train.info()
X = train.drop(["target"], axis = 1)

y = train["target"]
# checando a dimensão

X.shape
# função get_dummies transforma as categorias em variáveis binárias

X = pd.get_dummies(X)

X.head()
# Checando a dimensão

X.shape
X_train, X_test, y_train, y_test = train_test_split(

    X, y, stratify=y, random_state=0)
# X_train

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)





# X

scaler = MinMaxScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)
# Coeficiente de Gini Normalizado:

def gini(actual, pred):

    assert (len(actual) == len(pred))

    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)

    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    totalLosses = all[:, 0].sum()

    giniSum = all[:, 0].cumsum().sum() / totalLosses



    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)





def gini_normalized(actual, pred):

    return gini(actual, pred) / gini(actual, actual)
# Testando com a regressão logística com penalização L2

lr = LogisticRegression(penalty='l2', random_state=1)

lr.fit(X_train_scaled, y_train)

prob = lr.predict_proba(X_test_scaled)[:,1]

print("Índice de Gini normalizado para a Regressão Logística: ",gini_normalized(y_test, prob))
# Random Forest com 20 árvores

rf = RandomForestClassifier(n_estimators = 20, max_depth = 4, random_state = 1, max_features = 20)

rf.fit(X_train_scaled, y_train)

predictions_prob = rf.predict_proba(X_test_scaled)[:,1]

print("Índice de Gini normalizado para o Random Forest: ", gini_normalized(y_test, predictions_prob))
# taxa de aprendizagem = 0.05

xgb = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.05, random_state = 1)

xgb.fit(X_train_scaled, y_train)

prob_xgb = xgb.predict_proba(X_test_scaled)[:,1]

print("--------------------------------------------------------------------------------------------")

print("Coef. de Gini normalizado para o XGBoost com learning_rate = 0.05: ", gini_normalized(y_test, prob_xgb))

print("--------------------------------------------------------------------------------------------")
# Testaremos nos dados completo de treinamento.

prob_xgb_y = xgb.predict_proba(X_scaled)[:,1]

print("--------------------------------------------------------------------------------------------")

print("Coef. de Gini para o XGBoost com learning_rate = 0.05 nos dados de treinamento: ", gini_normalized(y, prob_xgb_y))

print("--------------------------------------------------------------------------------------------")
# Importação dos dados de teste

test = pd.read_csv('../input/test.csv')
# Excluindo dados duplicados

test.drop_duplicates()

# salvando id

test_id = test['id']

test.info()
# Excluindo 'id'

test = test.drop(['id'], axis = 1)

test.shape
# Excluindo as variáveis 'calc' e 'ps_car_03_cat' e 'ps_car_05_cat'

test.drop(variaveis_excluir, inplace=True, axis=1)

test.drop(colunas_calc, inplace=True, axis=1)
# Imputando a média e a moda

media_imp = Imputer(missing_values=-1, strategy='mean', axis=0)

moda_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

test['ps_reg_03'] = media_imp.fit_transform(test[['ps_reg_03']]).ravel()

test['ps_car_14'] = media_imp.fit_transform(test[['ps_car_14']]).ravel()

test['ps_car_11'] = moda_imp.fit_transform(test[['ps_car_11']]).ravel()

test.shape
# transformando variáveis em 'category'

for i in test.columns:

    if test[i].dtype == 'int64':

        test[i] = test[i].astype('category')

test.info()
# One-hot encoding

# função get_dummies transforma as categorias em variáveis binárias

test = pd.get_dummies(test)

test.shape
# garantindo que a base de teste sejam as mesmas colunas da base de treinamento

missing_cols = set( X.columns ) - set( test.columns )

for c in missing_cols:

    test[c] = 0

test = test[X.columns]
test.shape
# Normalização min-max

scaler = MinMaxScaler()

scaler.fit(test)

test_scaled = scaler.transform(test)
# Criando base para submissão com o modelo XGBoost

prob_xgb_teste = xgb.predict_proba(test_scaled)[:,1]

# Em results_df está a base de teste escorada, a coluna target possui as probabilidades

results_df = pd.DataFrame(data={'id':test_id, 'target':prob_xgb_teste})

print(results_df)

results_df.to_csv('submission1.csv', index=False)
#import pandas as pd

#from pandas import read_csv, DataFrame

#import numpy as np

#from numpy.random import seed

#from sklearn.preprocessing import minmax_scale

#from sklearn.preprocessing import MinMaxScaler

#from sklearn.model_selection import train_test_split

#from sklearn import datasets

#from keras.layers import Input, Dense

#from keras.models import Model

#from matplotlib import pyplot as plt



# Carregamento das bases de treinamento e teste em dataframes

#train = pd.read_csv('../input/train.csv')



#print(train.shape)



# X armazena dos dados em um dataframe

#X = train.iloc[:,2:]

# y armazena os labels em um dataframe

#y = train.iloc[:,1:2]



# target_names armazena os valores distintos dos labels

#target_names = train['target'].unique()



# Normaliza os dados de treinamento

#scaler = MinMaxScaler()

#scaler.fit(X)

#X_scaled = scaler.transform(X)



# Criação do AutoEncoder com 3 neurônios na camada escondida usando Keras.

#input_dim = X_scaled.shape[1]



# Definição do número de variáveis resultantes do Encoder

#encoding_dim = 10



#input_data = Input(shape=(input_dim,))



# Configurações do Encoder

#encoded = Dense(encoding_dim, activation='linear')(input_data)

#encoded = Dense(encoding_dim, activation='sgmoid')(input_data)

#encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_data)



#encoded1 = Dense(20, activation = 'relu')(input_data)

#encoded2 = Dense(10, activation = 'relu')(encoded1)

#encoded3 = Dense(5, activation = 'relu')(encoded2)

#encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)



# Configurações do Decoder

#decoded = Dense(input_dim, activation='linear')(encoded)

#decoded = Dense(input_dim, activation='sgmoid')(encoded)



#decoded1 = Dense(5, activation = 'relu')(encoded4)

#decoded2 = Dense(10, activation = 'relu')(decoded1)

#decoded3 = Dense(20, activation = 'relu')(decoded2)

#decoded4 = Dense(input_dim, activation = 'sigmoid')(decoded3)



# Combinando o Encoder e o Decoder em um modelo AutoEncoder

#autoencoder = Model(input_data, decoded4)

#autoencoder.compile(optimizer='adam', loss='mse')

#print(autoencoder.summary())

# Treinamento de fato - Definição de alguns parâmetros como número de épocas, batch size, por exemplo.

#history = autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=256, shuffle=True, validation_split=0.1, verbose = 1)



#plot a loss 

#plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

#plt.title('Model Train vs Validation Loss')

#plt.ylabel('Loss')

#plt.xlabel('Epoch')

#plt.legend(['Train', 'Validation'], loc='upper right')

#plt.show()



# Utilização do Encoder gerado para realizar a compressão e reduzir a dimensão da base de treinamento



#test = pd.read_csv('../input/test.csv')



#print(test.shape)



# X armazena dos dados em um dataframe

#X = test.iloc[:,1:]



# Normaliza os dados de treinamento

#scaler = MinMaxScaler()

#scaler.fit(X)

#X_scaled = scaler.transform(X)



# Utilizar o Encoder para codificar os dados de entrada

#encoder = Model(input_data, encoded4)

#encoded_data = encoder.predict(X_scaled)



#print(encoded_data)