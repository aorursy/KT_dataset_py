# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', thousands=',')

test = pd.read_csv('../input/test.csv', thousands=',')
df = pd.concat([train, test], sort=False)

df.head(10).T
# Verificando os tipos de dados

df.info()
# Verificando as capitais

df[df['capital']==1]
# Tem que arrumar: populacao (tem virgula e (1)) comissionados_por_servidor (% e DIV/0)

import re

df['populacao'] = df['populacao'].str.replace(',', '').str.replace('.', '')

df['populacao'] = df['populacao'].apply(lambda x: re.sub(r'(\(\d\))', '', x))

df['populacao'] = df['populacao'].astype(float)



# Recalculando comissionados por servidor

df['comissionados_por_servidor'] = df['comissionados'] / df['servidores']



# Recalculando pib_pc

df['pib_pc'] = df['pib'] / df['populacao']



# Recalculando densidade_dem

df['densidade_dem'] = df['populacao'] / df['area']
# Verificando novamente

df.head(10).T
df.describe()
df.hist(alpha=0.5, figsize=(16, 10))
# Ordenando o DataFrame

df = df.sort_values(['estado', 'municipio'])
# Aqueles que tem zero mas deveria ser NA

cols = ['jornada_trabalho', 'anos_estudo_empreendedor', 'taxa_empreendedorismo']

df[cols] = df[cols].replace(0, np.NaN)
# Verificando quantos NA

print(df.isna().sum())



# Preenchendo os NA

# Descobrindo quais colunas tem NA

cols = df.columns[df.isna().any()].tolist()

cols.remove('nota_mat')  # tirando a nota de matematica

print(cols)



## Tentativa dropando os NA pra treinar melhor 

#df_orig = df

#df = df.copy().dropna()



for c in cols:  # Preencher NA com a média do estado daquele municipio pra cada uma das colunas

    df[c] = df.groupby('estado')[c].transform(lambda x: x.fillna(x.median()))

    

# Criando variáveis



# Gasto em Educação em Relação ao PIB_PC

df['educacao_pelo_pib'] = df['gasto_pc_educacao'] / df['pib_pc']



# Servidores x População

df['servidores_pc'] = df['servidores'] / df['populacao']



# Tempo de estudo empreendedor é acima da média

# media = df['anos_estudo_empreendedor'].mean()

# df['tempo_estudo'] = [1 if tempo > media else 0 for tempo in df['anos_estudo_empreendedor']]
# Criando dummies

categoricas = ['regiao', 'porte']

for cat in categoricas:

    dummies = pd.get_dummies(df[cat], prefix=cat)

    df = pd.concat([df, dummies], axis=1)



# Remove as originais

cols = [c for c in df.columns if c not in categoricas]

df = df[cols]
# Normalizando os dados 

# Transformando para log valores com muita diferença (de acordo com os histogramas)

cols = ['populacao', 'area', 'pib', 'pib_pc', 'servidores', 'comissionados', 'educacao_pelo_pib', 'densidade_dem', 'gasto_pc_educacao', 'gasto_pc_saude', 'hab_p_medico']

for col in cols:

    df[col] = np.log(df[col]+1)





# Percentual da população economicamente ativa não está na escala 0 a 1

df['perc_pop_econ_ativa'] = df['perc_pop_econ_ativa'] / 100



# Pegando os valores não dummies

cols = [c for c in df.columns if ((df[c].dtype==np.float64) or (df[c].dtype==np.int64))]

cols.remove('nota_mat')

cols.remove('codigo_mun')

cols.remove('comissionados_por_servidor')

for c in ['populacao', 'area', 'pib', 'pib_pc', 'servidores', 'comissionados', 'perc_pop_econ_ativa',]:

    cols.remove(c)



# Colocando na forma normal

for c in cols:

    df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())  # min/max

    #df[c] = (df[c]-df[c].mean())/df[c].std()  # media/std

    

df.head()
df.head(10).T
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10, 10))



import seaborn as sns

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# Separando novamente as variáveis de acordo com nota_mat

test = df[df['nota_mat'].isnull()]

train = df[~df['nota_mat'].isnull()]

print(test.shape, train.shape)



# Colunas que não serão utilizadas

remove = ['municipio', 'codigo_mun', 'nota_mat', 'estado', 'porte_Grande porte', 'regiao_CENTRO-OESTE', 'municipio_ACRELANDIA', 'estado_AC',

          'capital',  'area', 'comissionados', 'servidores', 'comissionados_por_servidor']

feats = [c for c in train.columns if c not in remove]

print(feats)

# Separando em treino e validação

from sklearn.model_selection import train_test_split

train, valid = train_test_split(train, random_state=42, test_size=0.2)

train.shape, valid.shape
# Testando Classificador de Floresta

from sklearn.ensemble import RandomForestClassifier

params = {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 250, 'oob_score': False}

rf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)

rf.fit(train[feats], train['nota_mat'])

preds = rf.predict(valid[feats])
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10, 10))



pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score



print(mean_squared_error(valid['nota_mat'], preds)**(1/2))

print(accuracy_score(valid['nota_mat'], preds))
preds = rf.predict(test[feats])

test['nota_mat'] = preds

# Utilizando forestclassifier

test[['codigo_mun', 'nota_mat']].to_csv('saida_forest.csv', index=False)
def cv(df, test, feats, y_name, k=5, oob_score=False):

    preds, score, fis, acc = [], [], [], []

    chunk = df.shape[0] // k

    for i in range(k):

        if i + 1 < k:

            valid = df.iloc[i*chunk: (i+1)*chunk]

            train = df.iloc[:i*chunk].append(df.iloc[(i+1)*chunk:])

        else:

            valid = df.iloc[i*chunk:]

            train = df.iloc[:i*chunk]            

        rf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_depth=5, oob_score=oob_score)

        rf.fit(train[feats], train[y_name])

        acc.append(accuracy_score(valid[y_name], rf.predict(valid[feats])))

        score.append(mean_squared_error(valid[y_name], rf.predict(valid[feats]))**(1/2))

        preds.append(rf.predict(test[feats]))

        fis.append(rf.feature_importances_)

        print(i, 'OK')

    return score, preds, fis, acc
score, preds, fis, acc = cv(train, test, feats, 'nota_mat')
score, acc
fig=plt.figure(figsize=(20, 20))

tamanho = len(fis)

for i, f in enumerate(fis):

    plt.subplot(tamanho, 1, i+1)

    pd.Series(f, index=feats).sort_values().plot.barh()
x = pd.DataFrame(preds)
test['nota_mat'] = x.mode(axis=0).T
# Utilizando crossvalidation

test[['codigo_mun', 'nota_mat']].to_csv('saida_cross.csv', index=False)
from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(train[feats], train['nota_mat'])

preds = clf.predict(valid[feats])
print(mean_squared_error(valid['nota_mat'], preds)**(1/2))

print(accuracy_score(valid['nota_mat'], preds))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=500)

lr.fit(train[feats], train['nota_mat'])

preds = lr.predict(valid[feats])
print(mean_squared_error(valid['nota_mat'], preds)**(1/2))

print(accuracy_score(valid['nota_mat'], preds))
preds = lr.predict(test[feats])

test['nota_mat'] = preds

# Utilizando logistic regression

test[['codigo_mun', 'nota_mat']].to_csv('saida_lr.csv', index=False)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

import time



# Treina tudo e acha o melhor

def testa_tudo(train, feats, alvo):

    '''

    Função que tenta um monte de algoritmo de classificação e ve qual que presta. Demora um pouco.

    '''

    start_time = time.time()

    melhores = dict()

    # modelos

    models = {

        'RandomForest': RandomForestClassifier(random_state=42),

        'ExtraTrees': ExtraTreesClassifier(random_state=42),

        'AdaBoost': AdaBoostClassifier(random_state=42),

        'GradientBoosting': GradientBoostingClassifier(random_state=42),

        'DecisionTree': DecisionTreeClassifier(random_state=42),

        'LogisticRegression': LogisticRegression(penalty='l2', random_state=42),

        'KNN 11': KNeighborsClassifier(),

        'MLPClassifier': MLPClassifier(random_state=42),

        'SVC': SVC(),

        'GaussianNB': GaussianNB(),

        'BernoulliNB': BernoulliNB(),

     }

    # parametros

    params = {

        'RandomForest': {'n_estimators': [100,150,200,250], 'max_depth': [2,3,4,5,6,7], 'oob_score': [False, True], 'min_samples_split': [2,3,4,5]},

        'ExtraTrees': {'n_estimators': [100,150,200,250], 'max_depth': [2,3,4,5,6,7], 'min_samples_split': [2,3,4,5]},

        'AdaBoost': {'n_estimators': [50,100,150,200], 'learning_rate': [0.5,1.0,1.5]},

        'GradientBoosting': {'n_estimators': [100,200,300,500], 'max_leaf_nodes': [2,3,4,5], 'max_depth': [4,5,6], 'min_samples_split': [2,3,4,5]},

        'DecisionTree': {'max_leaf_nodes': [2,3,4,5], 'max_depth': [4,5,6], 'min_samples_split': [2,3,4,5]},

        'LogisticRegression': {'dual': [True, False], 'max_iter': [110,130,150,200,300], 'C': [0.1,0.5,1.0,1.5,2.0]},

        'KNN 11': {'n_neighbors': [5,8,11,15], 'leaf_size': [20,30,50]},

        'MLPClassifier': {'alpha': [1e-2,1e-3,1e-4,1e-5], 'hidden_layer_sizes': [(20, 10, 2),(40, 20, 10),(10, 5, 1)]},

        'SVC': {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1],'kernel': ['rbf','linear']},

        'GaussianNB': {'var_smoothing': [1e-9,1e-10,1e-11]},

        'BernoulliNB': {'alpha': [0,0.01,0.1,1,1.1]},        

    }

    for nome, modelo in models.items():

        grid = GridSearchCV(estimator=modelo, param_grid=params[nome], cv = 4, n_jobs=-1)

        grid_result = grid.fit(train[feats], train[alvo])

        print("Modelo: %s, melhor classificação: %f usando %s" % (nome, grid_result.best_score_, grid_result.best_params_))

        melhores[nome] = (grid_result.best_score_, grid_result.best_params_)

    print("Tempo rodando: " + str((time.time() - start_time)) + ' ms')

    return melhores



def moda(melhores, treino, teste, feats, alvo):

    preds = []

    models = {

        'RandomForest': RandomForestClassifier(random_state=42, **melhores['RandomForest'][1]),

        'ExtraTrees': ExtraTreesClassifier(random_state=42, **melhores['ExtraTrees'][1]),

        'AdaBoost': AdaBoostClassifier(random_state=42, **melhores['AdaBoost'][1]),

        'GradientBoosting': GradientBoostingClassifier(random_state=42, **melhores['GradientBoosting'][1]),

        'DecisionTree': DecisionTreeClassifier(random_state=42, **melhores['DecisionTree'][1]),

        'LogisticRegression': LogisticRegression(penalty='l2', random_state=42, **melhores['LogisticRegression'][1]),

        'KNN 11': KNeighborsClassifier(**melhores['KNN 11'][1]),

        'MLPClassifier': MLPClassifier(random_state=42, **melhores['MLPClassifier'][1]),

        'SVC': SVC(**melhores['SVC'][1]),

        'GaussianNB': GaussianNB(**melhores['GaussianNB'][1]),

        'BernoulliNB': BernoulliNB(**melhores['BernoulliNB'][1]),

     }

    for nome, modelo in models.items():

        m = modelo

        m.fit(treino[feats], treino[alvo])

        preds.append(m.predict(teste[feats]))

    df = pd.DataFrame(preds)

    print(df)

    return df.T.mode(axis=1)

    
treino = df[-df['nota_mat'].isnull()]



#melhores = testa_tudo(treino, feats, 'nota_mat')
melhores#
m = moda(melhores, train, test, feats, 'nota_mat')

test['nota_mat'] = m

test[['codigo_mun', 'nota_mat']].to_csv('saida_moda.csv', index=False)

print(test[['codigo_mun', 'nota_mat']])
param = {'max_depth': 4, 'max_leaf_nodes': 2, 'min_samples_split': 2, 'n_estimators': 200}

gb = GradientBoostingClassifier(**param)

gb.fit(treino[feats], treino['nota_mat'])

preds = gb.predict(test[feats])

test['nota_mat'] = preds

test[['codigo_mun', 'nota_mat']].to_csv('saida_gb.csv', index=False)
from sklearn.decomposition import PCA



pca = PCA(n_components=4)

pca.fit(treino[feats])

print(pca.explained_variance_ratio_) 

print(pca.singular_values_)

novo = pca.fit_transform(treino[feats])

print(novo, novo.shape)

novo_valid = pca.transform(valid[feats])

novo_teste = pca.fit_transform(test[feats])



params =  {'C': 0.1, 'dual': True, 'max_iter': 130}

lg = LogisticRegression(penalty='l2', **params)

lg.fit(novo, treino['nota_mat'])



pred = lg.predict(novo_valid)



print(valid.shape)

print(accuracy_score(valid['nota_mat'], pred))

print(lg.score(novo_valid, valid['nota_mat']))



test['nota_mat'] = preds

test[['codigo_mun', 'nota_mat']].to_csv('saida_lg2.csv', index=False)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB(var_smoothing=1e-9)

gnb.fit(train[feats], train['nota_mat'])

print(gnb.score(valid[feats], valid['nota_mat']))



pred = gnb.predict(test[feats])

test[['codigo_mun', 'nota_mat']].to_csv('saida_gnb.csv', index=False)
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(train[feats], train['nota_mat'])

print(mnb.score(valid[feats], valid['nota_mat']))



pred = mnb.predict(test[feats])

test['nota_mat'] = pred

test[['codigo_mun', 'nota_mat']].to_csv('saida_mnb.csv', index=False)
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB(alpha=0)

bnb.fit(train[feats], train['nota_mat'])

print(bnb.score(valid[feats], valid['nota_mat']))



pred = bnb.predict(test[feats])

test['nota_mat'] = pred

test[['codigo_mun', 'nota_mat']].to_csv('saida_bnb.csv', index=False)
from sklearn.ensemble import VotingClassifier

melhores = {'max_depth': 2,   'min_samples_split': 2,   'n_estimators': 100,   'oob_score': False}



lr =  LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42)

rf = RandomForestClassifier(random_state=42, **melhores)

bnb = BernoulliNB(alpha=0)

gnb = GaussianNB(var_smoothing=1e-9)

vc = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('gnb', gnb), ('bnb', bnb)], voting='hard')

vc.fit(train[feats], train['nota_mat'])

vc.score(valid[feats], valid['nota_mat'])



pred = vc.predict(test[feats])

test['nota_mat'] = pred

test[['codigo_mun', 'nota_mat']].to_csv('voting.csv', index=False)
