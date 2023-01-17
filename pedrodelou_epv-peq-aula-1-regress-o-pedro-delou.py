# comandos mágicos que não se comunicam com a linguagem Python e sim diretamente com o kernel do Jupyter
# começam com %

%load_ext autoreload
%autoreload 2

%matplotlib inline
# importando os principais módulos que usaremos ao longo da aula

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

# você também pode importar apenas uma parte de cada módulo, por exemplo:
# from sklearn.ensemble import RandomForestRegressor()
boston = sklearn.datasets.load_boston()
boston
print(boston.DESCR)
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
                                                                            test_size = 0.1, 
                                                                            random_state = 0)
m = sklearn.ensemble.RandomForestRegressor()
m.fit(X_train, y_train)
y_test_pred = m.predict(X_test)
# plotando valores verdadeiros contra predições
plt.plot(y_test, y_test_pred,'.')

# plotando a reta x=y
plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim())

# legenda dos eixos
plt.xlabel('y_test')
plt.ylabel('y_test_pred');
mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
r2 = sklearn.metrics.r2_score(y_test, y_test_pred)

print(f'MAE: {mae}')
print(f'R2: {r2}')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import sklearn.neural_network

import pandas as pd

boston = sklearn.datasets.load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
                                                                            test_size = 0.1, 
                                                                            random_state = 0)

#X_train = X_train[:,(5,12)]
#X_test = X_test[:,(5,12)]

# Modelo Random Forest
m = sklearn.ensemble.RandomForestRegressor(random_state=0)
m.fit(X_train, y_train)
importance = m.feature_importances_

y_test_pred = m.predict(X_test)

# Modelo NN MLP
m2 = sklearn.neural_network.MLPRegressor(random_state=0, max_iter=10000, hidden_layer_sizes=(20,20,20,20,20))
m2.fit(X_train, y_train)
y_test_pred2 = m2.predict(X_test)

# plotando valores verdadeiros contra predições
plt.figure()
plt.plot(y_test, y_test_pred,'.', label='RF')
plt.plot(y_test, y_test_pred2,'.', label='NLP')
plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), label='y=x')
plt.xlabel('y_test')
plt.ylabel('y_test_pred');
plt.legend();

# métricas de avaliação
mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
mse = sklearn.metrics.mean_squared_error(y_test, y_test_pred)
maxerror = sklearn.metrics.max_error(y_test, y_test_pred)
r2 = sklearn.metrics.r2_score(y_test, y_test_pred)

mae2 = sklearn.metrics.mean_absolute_error(y_test, y_test_pred2)
mse2 = sklearn.metrics.mean_squared_error(y_test, y_test_pred2)
maxerror2 = sklearn.metrics.max_error(y_test, y_test_pred2)
r22 = sklearn.metrics.r2_score(y_test, y_test_pred2)

nomes =['MAE','MSE','Erro máximo','R^2']

dados = {'Random Forest': [mae,mse,maxerror,r2],
         'NLP': [mae2,mse2,maxerror2,r22]}

df1 = pd.DataFrame(dados, index = nomes)
display(df1)

plt.figure()
plt.bar(boston.feature_names, importance)
plt.xticks(rotation=45)
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import sklearn.neural_network

import pandas as pd

diabetes = sklearn.datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
                                                                            test_size = 0.1, 
                                                                            random_state = 0)

# Modelo Random Forest
m = sklearn.ensemble.RandomForestRegressor(random_state=0)
m.fit(X_train, y_train)
importance = m.feature_importances_

y_test_pred = m.predict(X_test)

# Modelo NN MLP
m2 = sklearn.neural_network.MLPRegressor(random_state=0, max_iter=10000, hidden_layer_sizes=(20,20,20,20))
m2.fit(X_train, y_train)
y_test_pred2 = m2.predict(X_test)

# plotando valores verdadeiros contra predições
plt.figure()
plt.plot(y_test, y_test_pred,'.', label='RF')
plt.plot(y_test, y_test_pred2,'.', label='NLP')
plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), label='y=x')
plt.xlabel('y_test')
plt.ylabel('y_test_pred');
plt.legend();

# métricas de avaliação
mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
mse = sklearn.metrics.mean_squared_error(y_test, y_test_pred)
maxerror = sklearn.metrics.max_error(y_test, y_test_pred)
r2 = sklearn.metrics.r2_score(y_test, y_test_pred)

mae2 = sklearn.metrics.mean_absolute_error(y_test, y_test_pred2)
mse2 = sklearn.metrics.mean_squared_error(y_test, y_test_pred2)
maxerror2 = sklearn.metrics.max_error(y_test, y_test_pred2)
r22 = sklearn.metrics.r2_score(y_test, y_test_pred2)

nomes =['MAE','MSE','Erro máximo','R^2']

dados = {'Random Forest': [mae,mse,maxerror,r2],
         'NLP': [mae2,mse2,maxerror2,r22]}

df1 = pd.DataFrame(dados, index = nomes)
display(df1)

plt.figure()
plt.bar(diabetes.feature_names, importance)
plt.xticks(rotation=45)
plt.show()

PATH = "../input/bluebook-for-bulldozers/"

df_raw = pd.read_csv(f'{PATH}Train.zip',
                     compression='zip', 
                     low_memory=False, 
                     parse_dates=["saledate"])
df_raw.shape
with pd.option_context("display.max_columns", 100): 
    display(df_raw)
    display(df_raw.describe(include='all'))
df_raw.dtypes
for n, c in df_raw.items():
    if not pd.api.types.is_numeric_dtype(c) and not pd.api.types.is_datetime64_any_dtype(c):
        print(f'{n} ({len(c.unique())}): {c.unique()}')
# para poder importar módulos que não estejam nos kernels do kaggle, 
# devemos instalá-los com o pip

!pip install missingno
import missingno

missingno.bar(df_raw)
missingno.matrix(df_raw);
df_raw.SalePrice = np.log(df_raw.SalePrice)
def pre_process (df):
    
    new_df = pd.DataFrame()
    
    for n,c in df.items():
                
        if pd.api.types.is_numeric_dtype(c):
            # substituindo NaN numericos pelas medianas de cada coluna
            new_df[n] = c.fillna(value=c.median())
        else:
            # interpretando o que nao for numerico como variaveis categoricas 
            # e transformando cada categoria em um numero
            new_df[n] = pd.Categorical(c.astype('category').cat.as_ordered()).codes+1
    
    return new_df     
df_proc = pre_process(df_raw)
X, y = df_proc.drop('SalePrice', axis=1), df_proc['SalePrice']
n_valid = 12000
n_trn = len(df_proc)-n_valid

X_treino, X_validacao = X[:n_trn].copy(), X[n_trn:].copy()
y_treino, y_validacao = y[:n_trn].copy(), y[n_trn:].copy()

y_treino.shape, y_validacao.shape
def rmse(x,y): 
    
    return np.sqrt(sklearn.metrics.mean_squared_error(x,y))

def display_score(m):
    
    res = [[rmse(m.predict(X_treino), y_treino), rmse(m.predict(X_validacao), y_validacao)],
          [m.score(X_treino, y_treino), m.score(X_validacao, y_validacao)]]
    
    score = pd.DataFrame(res, index=['RMSE','R2'], columns = ['Treino','Validação'])
    
    if hasattr(m, 'oob_score_'): 
        score.loc['OOB R2'] = [m.oob_score_,'-']
        
    display(score)
m_base = sklearn.ensemble.RandomForestRegressor(n_jobs=-1, oob_score = True, random_state = 0)
%time m_base.fit(X_treino, y_treino)
display_score(m_base)
m = sklearn.ensemble.RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)
def draw_tree(t, df, size=10, ratio=1, precision=0):
   
    import re
    import graphviz
    import sklearn.tree
    import IPython.display
    
    s=sklearn.tree.export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                                   special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))
draw_tree(m.estimators_[0], X_treino, precision=3)
m = sklearn.ensemble.RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)
display_score(m_base)
preds = np.stack([t.predict(X_validacao) for t in m_base.estimators_]).T
preds_df = pd.DataFrame(preds)

preds_df['medias'] = preds_df.mean(axis=1)
preds_df['stds'] = preds_df.std(axis=1)
preds_df['valor real'] = y_validacao.values
preds_df
plt.plot(y_validacao.values, preds_df.mean(axis=1), '.')

plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim());

plt.xlabel('y_valid')
plt.ylabel('y_valid_pred');
plt.plot([sklearn.metrics.r2_score(y_validacao, np.mean(preds[:,:i+1], axis=1)) for i in range(100)]);
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 10, n_jobs=-1, oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)
m = sklearn.ensemble.RandomForestRegressor(max_samples = 40000, n_jobs=-1, oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)
m = sklearn.ensemble.RandomForestRegressor(min_samples_leaf = 3, max_features = 0.5, 
                                           n_jobs=-1, oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 50, min_samples_leaf = 3, 
                                           max_features = 0.5, n_jobs=-1, 
                                           oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)
#n_estimators: diminuir não prejudica significativamente nas métricas de treino e validação e melhora muito o custo computacional (mantive 50 para comparação)
#min_samples_leaf: identifiquei que 4 fica melhor que 3 ou 5, talvez seja um valor ótimo para esse conjunto de dados e essa configuração
#max_features: mexer no max_features não melhorou o treinamento
#max_leaf_nodes: não surtiu efeito positivo nas métricas de treino e validação, na realidade esse hiperparâmetro compete com o min_samples_leaf, que apresentou melhora significativa
#min_impurity_decrease: o aumento desse hiperparâmetro piorou muito as métricas de treino e validação
#bootstrap: não consegui rodar com a opção false (aparece um erro)
#verbose: somente mostra um log mais detalhado do processo de treinamento
#warm_start: partir da solução anterior não apresentou benefício
#ccp_alpha: o aumento desse hiperparâmetro piorou muito as métricas de treino e validação
#max_samples: limitar o número de amostras não melhorou o processo de treinamento

m = sklearn.ensemble.RandomForestRegressor(n_estimators = 50, min_samples_leaf = 4, 
                                           max_features = 0.5, n_jobs=-1,
                                           oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)
def plotar_importancias(modelo, tags, n=10):
    
    fig, ax = plt.subplots(1,2, figsize = (20,4))

    coefs = []
    abs_coefs = []

    if hasattr(modelo,'coef_'):
        imp = modelo.coef_
    elif hasattr(modelo,'feature_importances_'):
        imp = modelo.feature_importances_
    else:
        print('sorry, nao vai rolar!')
        return

    coefs = (pd.Series(imp, index = tags))
    coefs.plot(use_index=False, ax=ax[0]);
    abs_coefs = (abs(coefs)/(abs(coefs).sum()))
    abs_coefs.sort_values(ascending=False).plot(use_index=False, ax=ax[1],marker='.')

    ax[0].set_title('Importâncias relativas das variáveis')
    ax[1].set_title('Importâncias relativas das variáveis - ordem decrescente')

    abs_coefs_df = pd.DataFrame(np.array(abs_coefs).T,
                                columns = ['Importancias'],
                                index = tags)

    df = abs_coefs_df['Importancias'].sort_values(ascending=False)
    
    print(df.iloc[0:n])
    plt.figure()
    df.iloc[0:n].plot(kind='barh', figsize=(15,0.25*n), legend=False)
    
    return df
imp = plotar_importancias(m, X_validacao.columns,30)
to_keep = imp[imp>0.005].index
to_keep.shape
X_treino = X_treino[to_keep]
X_validacao = X_validacao[to_keep]
def dendogram_spearmanr(df, tags):

    import scipy.cluster.hierarchy
    import scipy.stats
    
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = scipy.cluster.hierarchy.distance.squareform(1-corr)
    z = scipy.cluster.hierarchy.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(18,8))
    dendrogram = scipy.cluster.hierarchy.dendrogram(z, labels=tags, orientation='left', leaf_font_size=16)
    plt.show()
dendogram_spearmanr(X_treino, X_treino.columns)
def get_oob(X):
    m = sklearn.ensemble.RandomForestRegressor(n_estimators=30, min_samples_leaf=5, 
                                               max_features=0.6, n_jobs=-1, max_samples = 100000,
                                               oob_score=True, random_state = 0)
    m.fit(X, y_treino)
    return m.oob_score_
get_oob(X_treino)
for c in ('Grouser_Tracks', 'Hydraulics_Flow', 'Coupler_System',
          'fiModelDesc', 'fiBaseModel','ProductGroupDesc', 'ProductGroup'):
    print(c, get_oob(X_treino.drop(c, axis=1)))
to_drop = ['ProductGroupDesc', 'fiModelDesc', 'Grouser_Tracks', 'Hydraulics_Flow']
get_oob(X_treino.drop(to_drop, axis=1))
X_treino = X_treino.drop(to_drop, axis=1)
X_treino.shape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import sklearn.neural_network

import pandas as pd

boston = sklearn.datasets.load_boston()
X, y = boston.data, boston.target

X_treino, X_validacao, y_treino, y_validacao = sklearn.model_selection.train_test_split(X, y, 
                                                                            test_size = 0.2, 
                                                                            random_state = 0)

# Modelo Random Forest - Caso Base
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 130, min_samples_leaf = 1, 
                                           max_features = 0.99, n_jobs=-1,
                                           oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)

# Análise de importância das variáveis
imp = plotar_importancias(m, boston.feature_names,30)
display(boston.feature_names)
#removendo variáveis com imp<0.01
X_treino = X_treino[:,[0,4,5,6,7,9,10,12]]
X_validacao = X_validacao[:,[0,4,5,6,7,9,10,12]]

# Modelo Random Forest - Removendo Variáveis Irrelevantes
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 130, min_samples_leaf = 1, 
                                           max_features = 0.99, n_jobs=-1,
                                           oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)

# Análise da correlação entre as variáveis

dendogram_spearmanr(X_treino, boston.feature_names)
# removendo a variável RM, pois ela tem correlação com AGE
X_treino, X_validacao, y_treino, y_validacao = sklearn.model_selection.train_test_split(X, y, 
                                                                            test_size = 0.2, 
                                                                            random_state = 0)

X_treino = X_treino[:,[0,4,6,7,9,10,12]]
X_validacao = X_validacao[:,[0,4,6,7,9,10,12]]

m = sklearn.ensemble.RandomForestRegressor(n_estimators = 130, min_samples_leaf = 1, 
                                           max_features = 0.99, n_jobs=-1,
                                           oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)

# Neste modelo não vale a pena remover variáveis, nem pelo grau de importância, nem pela correlação. 
# A remoção das variáveis com importancia menor do que 0.01 não impactou de forma sensível no R^2 da
# validação, porém também não se refletiu em um ganho de tempo computacional.
# A remoção da variável RM, por ter correlação com AGE acarretou em uma queda forte do R^2, não se
# refletindo em qualquer benefício para o modelo.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import sklearn.neural_network

import pandas as pd

diabetes = sklearn.datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_treino, X_validacao, y_treino, y_validacao = sklearn.model_selection.train_test_split(X, y, 
                                                                            test_size = 0.1, 
                                                                            random_state = 0)

display(X_treino.shape)

# Modelo Random Forest - Caso Base
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 100, min_samples_leaf = 1,
                                           n_jobs=-1, oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)

#preds = np.stack([t.predict(X_validacao) for t in m.estimators_]).T
#preds_df = pd.DataFrame(preds)

#preds_df['medias'] = preds_df.mean(axis=1)
#preds_df['stds'] = preds_df.std(axis=1)
#preds_df['valor real'] = y_validacao
#plt.plot([sklearn.metrics.r2_score(y_validacao, np.mean(preds[:,:i+1], axis=1)) for i in range(100)]);


# Análise de importância das variáveis
imp = plotar_importancias(m, diabetes.feature_names,30)
display(diabetes.feature_names)
#removendo variáveis com imp<0.05
X_treino = X_treino[:,[0,2,3,5,6,8,9]]
X_validacao = X_validacao[:,[0,2,3,5,6,8,9]]

# Modelo Random Forest - Removendo Variáveis Irrelevantes
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 100, min_samples_leaf = 1,
                                           n_jobs=-1, oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)

# Análise da correlação entre as variáveis
dendogram_spearmanr(X_treino, diabetes.feature_names)

# Neste modelo não vale a pena remover variáveis, nem pelo grau de importância, nem pela correlação. 
# A remoção das variáveis com importancia menor do que 0.05 impactou de forma sensível no R^2 da
# validação, que já se mostrou muito baixo.
# Não há variáveis fortemente correlacionadas.
def pre_process_OHE (df, max_cats = 10):
    
    new_df = pd.DataFrame()
    
    for n,c in df.items():
                
        if pd.api.types.is_numeric_dtype(c):
            # substituindo NaN numericos pelas medianas de cada coluna
            new_df[n] = c.fillna(value=c.median())
        else:
            # interpretando o que nao for numerico como variaveis categoricas 
            new_df[n] = pd.Categorical(c.astype('category').cat.as_ordered())
            # transformando cada categoria em um numero, caso nao va fazer one hot encoding com ela
            if len(c.astype('category').cat.categories) > max_cats:
                new_df[n] = pd.Categorical(new_df[n]).codes+1
    
    # a função pd.get_dummies faz o one-hot encoding
    return pd.get_dummies(new_df)
df_raw
df_proc_ohe = pre_process_OHE(df_raw)
df_proc_ohe
X, y = df_proc_ohe.drop('SalePrice', axis=1), df_proc_ohe['SalePrice']

n_valid = 12000
n_trn = len(df_proc)-n_valid

X_treino, X_validacao = X[:n_trn].copy(), X[n_trn:].copy()
y_treino, y_validacao = y[:n_trn].copy(), y[n_trn:].copy()
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 50, min_samples_leaf = 3, 
                                           max_features = 0.5, n_jobs=-1, 
                                           oob_score = True, random_state = 0)
%time m.fit(X_treino, y_treino)
display_score(m)
plotar_importancias(m, X_validacao.columns,30)
!pip install treeinterpreter
from treeinterpreter import treeinterpreter as ti
row = X_validacao.values[np.newaxis,0]

prediction, bias, contributions = ti.predict(m, row)

idxs = np.argsort(contributions[0])
[o for o in zip(X_validacao.columns[idxs], X_validacao.iloc[0][idxs], contributions[0][idxs])]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import sklearn.neural_network

import pandas as pd

boston = sklearn.datasets.load_boston()
X, y = boston.data, boston.target

X_treino, X_validacao, y_treino, y_validacao = sklearn.model_selection.train_test_split(X, y, 
                                                                            test_size = 0.2, 
                                                                            random_state = 0)

# Modelo Random Forest - Caso Base
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 130, min_samples_leaf = 1, 
                                           max_features = 0.99, n_jobs=-1,
                                           oob_score = True, random_state = 0)
m.fit(X_treino, y_treino)

row = X_validacao[np.newaxis,0]
prediction, bias, contributions = ti.predict(m, row)

idxs = np.argsort(contributions[0])

rotulos=boston.feature_names[idxs]

for o in range(12):
    print(rotulos[o], row[0,o], contributions[0,o])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import sklearn.neural_network

import pandas as pd

diabetes = sklearn.datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_treino, X_validacao, y_treino, y_validacao = sklearn.model_selection.train_test_split(X, y, 
                                                                            test_size = 0.1, 
                                                                            random_state = 0)

# Modelo Random Forest - Caso Base
m = sklearn.ensemble.RandomForestRegressor(n_estimators = 100, min_samples_leaf = 1,
                                           n_jobs=-1, oob_score = True, random_state = 0)
m.fit(X_treino, y_treino)

row = X_validacao[np.newaxis,0]
prediction, bias, contributions = ti.predict(m, row)

idxs = np.argsort(contributions[0])

rotulos = np.array(diabetes.feature_names)
rotulos = rotulos[idxs]

for o in range(8):
    print(rotulos[o], row[0,o], contributions[0,o])
# criando exemplo simples com tendência linear
N = 30
x = np.arange(N)
y = 2*x

# adicionando ruído
y = y+ 3*np.random.randn(N)

# separando em treino e teste

n = int(N/2)

# se eu nao criar esse novo eixo em x a seguir, o sklearn reclama, 
# pq pra ele a array de dados preditores tem q ter 2 dimensoes:
x = x[:,np.newaxis]   

x_treino, y_treino = x[:n], y[:n]
x_treino, y_treino = x[:n], y[:n]

x_teste, y_teste = x[n:], y[n:]
x_teste, y_teste = x[n:], y[n:]

# especificando modelos

import sklearn.linear_model
import sklearn.neural_network
import sklearn.svm
import sklearn.neighbors

modelos = [sklearn.linear_model.LinearRegression(),
           sklearn.neural_network.MLPRegressor(),
           sklearn.ensemble.RandomForestRegressor(),
           sklearn.neighbors.KNeighborsRegressor(),
           sklearn.svm.SVR()]

# preparando janela do gráfico
fig, ax = plt.subplots(1,5,figsize=(20,3))

# calculando e plotando
for i in range(len(modelos)):
    modelos[i].fit(x_treino, y_treino)
    ax[i].plot(x, y)
    ax[i].plot(x, modelos[i].predict(x),'.')
    ax[i].set_title(modelos[i].__class__.__name__)
    ax[i].axvline(n,ls='--',c='k')
df_final = pre_process(df_raw)

X, y = df_final.drop('SalePrice', axis=1)[to_keep].drop(to_drop, axis=1), df_final['SalePrice']

m = sklearn.ensemble.RandomForestRegressor(min_samples_leaf = 3, 
                                           max_features = 0.5, n_jobs=-1, 
                                           oob_score = True, random_state = 0)
%time m.fit(X, y)
m.oob_score_