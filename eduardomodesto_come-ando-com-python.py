%matplotlib inline

import pandas as pd

import numpy as np

import pylab as plt

from tqdm import tqdm_notebook

np.random.seed(5)



plt.rc('figure', figsize=(10, 5))

fizsize_with_subplots = (10, 10)

bin_size = 10



# df_train é o nosso dataframe com os dados de treinamento para construção de nosso modelo.

df_train = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

df_test = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')

df_train.head(10)
df_train.describe().T
df_train.isnull().any()
df_null = df_train.isnull().sum()

print(df_null)
df_train.info()
y = df_train.target

df_train.drop('target', axis=1, inplace=True) #removemos a coluna com a resposta que foi armazenada em y

n_train = df_train.ID.count()

n_test = df_test.ID.count()



df_todos = pd.concat([df_train, df_test]) # concatena os dataframes

n_todos = df_todos.ID.count()



print(n_train, n_test, n_train+n_test, n_todos)

def rmissingvaluecol(dff,threshold):

    l = []

    l = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index))>=threshold))].columns, 1).columns.values)

    print("# Columns having more than %s percent missing values:"%threshold,(dff.shape[1] - len(l)))

    print("Columns:\n",list(set(list((dff.columns.values))) - set(l)))

    return l

rmissingvaluecol(df_todos,3)
#retiramos as colunas com mais de 3% de valores faltantes

l = rmissingvaluecol(df_todos,3)

df1 = df_todos[l]

df1.head(10)
df1.describe().T
### Aqui fazemos pela média a substituição dos nulos

df1.loc[df1['v10'].isnull(), 'v10'] = int(df1.v10.mean())

df1.loc[df1['v12'].isnull(), 'v12'] = int(df1.v10.mean())

df1.loc[df1['v14'].isnull(), 'v14'] = int(df1.v10.mean())

df1.loc[df1['v21'].isnull(), 'v21'] = int(df1.v10.mean())

df1.loc[df1['v34'].isnull(), 'v34'] = int(df1.v10.mean())

df1.loc[df1['v40'].isnull(), 'v40'] = int(df1.v10.mean())

df1.loc[df1['v50'].isnull(), 'v50'] = int(df1.v10.mean())

df1.loc[df1['v114'].isnull(), 'v114'] = int(df1.v10.mean())

df1.describe().T
#os dados categoricos precisam de tratamento

df1_null = df1.isnull().sum()

print(df1_null)
df_v22 = df1.groupby('v22')

df_v22 = df_v22.size()

df_v22.sort_values(inplace=True, ascending=False)

df_v22

## Tornando os nulos em v22 igual à moda (valor mais comum)

df1.loc[df1['v22'].isnull(), 'v22'] = 'AGDF'

df_v52 = df1.groupby('v52')

df_v52 = df_v52.size()

df_v52.sort_values(inplace=True, ascending=False)

df_v52
## Tornando os nulos em v52 igual à moda (valor mais comum)

df1.loc[df1['v52'].isnull(), 'v52'] = 'J'
df_v91 = df1.groupby('v91')

df_v91 = df_v91.size()

df_v91.sort_values(inplace=True, ascending=False)

df_v91
## Tornando os nulos em v91 igual à moda (valor mais comum)

df1.loc[df1['v91'].isnull(), 'v91'] = 'A'
df_v107 = df1.groupby('v107')

df_v107 = df_v107.size()

df_v107.sort_values(inplace=True, ascending=False)

df_v107
## Tornando os nulos em v107 igual à moda (valor mais comum)

df1.loc[df1['v107'].isnull(), 'v107'] = 'E'
df_v112 = df1.groupby('v112')

df_v112 = df_v112.size()

df_v112.sort_values(inplace=True, ascending=False)

df_v112
## Tornando os nulos em v112 igual à moda (valor mais comum)

df1.loc[df1['v112'].isnull(), 'v112'] = 'F'
df_v125 = df1.groupby('v125')

df_v125 = df_v125.size()

df_v125.sort_values(inplace=True, ascending=False)

df_v125
## Tornando os nulos em v125 igual à moda (valor mais comum)

df1.loc[df1['v125'].isnull(), 'v125'] = 'BM'
df1.isnull().any()
df1.dtypes

# convertendo as variaveis categoricas

for col in df1.columns:

    if df1[col].dtype =='object':

        df1[col] = pd.factorize(df_todos[col])[0]

df1.head(3)

df1.dtypes
df_todos_features = df1
## checando se estamos com a quantidade certa de linhas. Vai lançar uma exceção se for diferente

assert df_todos_features.ID.count()==n_todos

ID = df_todos['ID'] # precisamos guardar para fazer a submissão para o kaggle
X_train_feature = df_todos_features[:n_train].values

y_train_feature = y.values
# Eliminação Recursiva de Variáveis



# Import dos módulos

import pandas as pd

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



# Separando o array em componentes de input e output

dados = pd.DataFrame(X_train_feature, columns=df_todos_features.columns)

X = X_train_feature

Y = y_train_feature



# Criação do modelo

modelo = LogisticRegression()



# RFE

rfe = RFE(modelo, 19)

fit = rfe.fit(X, Y)



# Print dos resultados

print("Variáveis Preditoras:", list(dados.columns.values))

print("Variáveis Selecionadas: %s" % fit.support_)

print("Ranking dos Atributos: %s" % fit.ranking_)

print("Número de Melhores Atributos: %d" % fit.n_features_)
df_todos_final = df1.drop([ 'ID', 'v22', 'v52', 'v107', 'v91','v112','v125'], 

                               axis=1, inplace=False)

df_todos_final.head(3)

from sklearn import preprocessing

import numpy as np

scaled_features = preprocessing.StandardScaler().fit_transform(df_todos_final)

df_todos_final_scaled = pd.DataFrame(scaled_features, index=df_todos_final.index, columns=df_todos_final.columns)

df_todos_final = df_todos_final_scaled

df_todos_final.head(3)
X_train = df_todos_final[:n_train].values

X_test = df_todos_final[n_train:].values

y_train = y.values

Id_test = ID[n_train:].values ## só nos interessa os ids do conjunto de teste para submissão

print(X_train.shape, y_train.shape, X_test.shape, Id_test.shape)
## Agora já podemos embaralhar os dados de treino

from sklearn.model_selection import KFold

nfolds=4

kf = KFold(n_splits=nfolds, shuffle=True, random_state=777)



from sklearn.metrics import accuracy_score 

from sklearn.metrics import log_loss ## Essa é a métrica usada na competição do kaggle

from xgboost.sklearn import XGBClassifier



y_full_test =[] ##Aqui guardamos as previsões de cada modelo (classificador) em todo o dado de teste

y_full_valid = np.zeros(len(y_train)) ##Aqui fazemos a previsão de valiação out-of-fold



for train, valid in tqdm_notebook(kf.split(X_train, y_train)):

    ## Separamos os dados dos folds

    x_train_fold = X_train[train]

    y_train_fold = y_train[train]

    x_valid = X_train[valid]

    y_valid = y_train[valid]

    

    ##Treinamos o classificador, avaliamos nos dados de validação e medimos o desempenho

    clf =  XGBClassifier(random_state=777, max_depth=6, subsample=0.5, learning_rate = 0.07,n_estimators = 150, min_child_weight=10)

    clf.fit(x_train_fold, y_train_fold, eval_metric='logloss')

    y_full_valid[valid] = clf.predict(x_valid)

    

    

    ##Aqui realizamos a previsão nos dados de teste. Para cada modelo (fold) vamos gerar as previsões completas

    ##nesses dados

    y_full_test.append(clf.predict_proba(X_test)[:,1])

    

    print('acurácia na validação', accuracy_score(y_train, y_full_valid))



    # Predição no dataset de treino

    train_pred = clf.predict(x_train_fold)

    train_pred_prob = clf.predict_proba(x_train_fold)[:,1]

    print("Log Loss: %f" % log_loss(y_train_fold, train_pred_prob))





# Realizando as previsoes

test_pred_prob = clf.predict_proba(X_test)[:,1]  
## soma as previsões de cada classificador, que no final pode dar até nfold no total 

total = np.sum(y_full_test, axis=0)

## Agora dividimos pelo numero de folds 

preds_test = np.divide(total,nfolds)

print(test_pred_prob)

print(preds_test)
df_result = pd.DataFrame(Id_test, columns=['ID'])

df_result['PredictedProb'] = (preds_test)

print(df_result.head(10))
df_result2 = pd.DataFrame(Id_test, columns=['ID'])

df_result2['PredictedProb'] = (test_pred_prob)

print(df_result2.head(10))
df_result.to_csv('submittion.csv', index=False) #Index=false remove uma coluna inútil numerada de 0 a n

df_result.head()
