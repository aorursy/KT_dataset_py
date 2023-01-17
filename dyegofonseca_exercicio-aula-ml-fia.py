# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dados_cancelamento = pd.read_csv('/kaggle/input/CartaoCredito_cancelamento.csv')
dados_sem_cancelamento = pd.read_csv('/kaggle/input/CartaoCredito_semtarget.csv')
dados_cancelamento.head(1)
dados_sem_cancelamento.head(1)
dados_cancelamento['Cancelamento'] = 1
dados_cancelamento.head(2)
dados_sem_cancelamento['Cancelamento'] = 0
dados_sem_cancelamento.head()
dados_cancelamento.shape
dados_cancelamento.columns
dados_cancelamento.dtypes
dados_cancelamento.head(2)
dados_cancelamento['Anuidade'] = dados_cancelamento.Anuidade.str.replace(',','.').values.astype(float)
dados_cancelamento.dtypes['Anuidade']
import matplotlib.pyplot as plt
import seaborn as sns
dados_cancelamento.dtypes
# Método que irá plotar o box plot para as análise a seguir

def plotar_box(dataframe, log=False):
    
    dataframe = dataframe.drop(['ID', 
                    'PerfilEconomico', 'Sexo', 'Idade', 'PerfilCompra',
                   'UF', 'CidadeResidencia',
                   'RegiaodoPais', 'MesesDesempregado',
                   'Cancelamento'],axis=1)
    if log == True:
        dataframe = np.log(dataframe)
        print(dados_cluster2.isnull().sum())
    else:
        None
    
    for i in dataframe.columns:
        if (dataframe.dtypes[i] == 'int64') or (dataframe.dtypes[i] == 'float64'):
            print('Box-plot referente ao: ', i)
            sns.boxplot(i, data=dataframe)
            plt.show()
plotar_box(dados_cancelamento, log=False)
dados_cancelamento.describe().round()
dados_cancelamento.loc[dados_cancelamento.NumeroComprasOnline == 34107625.0]
dados_cancelamento.sort_values('NumeroComprasOnline').round().tail()
dados_cancelamento.shape
dados_cancelamento.index
dados_cancelamento.loc[dados_cancelamento.NumeroComprasOnline == 34107625.0].index
dados_cancelamento.drop(dados_cancelamento.index[259], inplace=True)
dados_cancelamento.shape
dados_sem_cancelamento.shape
dados_sem_cancelamento.columns
dados_sem_cancelamento.dtypes
dados_sem_cancelamento.head(2)
dados_sem_cancelamento['Anuidade'] = dados_sem_cancelamento.Anuidade.str.replace(',','.').values.astype(float)
dados_sem_cancelamento.dtypes
plotar_box(dados_sem_cancelamento)
dados_sem_cancelamento.describe().round()
dados_sem_cancelamento.loc[dados_sem_cancelamento.NumeroComprasOnline == 34107625.0]
dados_sem_cancelamento.loc[dados_sem_cancelamento.NumeroComprasOnline == 34107625.0].index
dados_sem_cancelamento.drop(dados_sem_cancelamento.index[445], inplace=True)
dados_sem_cancelamento.describe().round().tail()
lista_dtype = dados_cancelamento.dtypes == dados_sem_cancelamento.dtypes
lista_dtype = lista_dtype.reset_index()

lista_dtype.rename(columns={"index": "Variavel", 0: "Resposta"}, inplace=True)

lista_dtype['dados_cancelamento.dtypes'] = dados_cancelamento.dtypes.values
lista_dtype['dados_sem_cancelamento.dtypes'] = dados_sem_cancelamento.dtypes.values

lista_dtype[lista_dtype.Resposta == False]
dados_cancelamento['ValorCompraAnual'] = dados_cancelamento.ValorCompraAnual.values.astype(float)
frames = [dados_sem_cancelamento, dados_cancelamento]
dados_cartao = pd.concat(frames)
dados_cartao.shape
dados_cartao.tail(2)
dados_cartao.shape
# Observamos que há 61 registros de usuário que tem menos de 18 anos de idade
dados_cartao[dados_cartao.Idade < 18].shape
dados_cartao = dados_cartao[dados_cartao.Idade >= 18.0]
dados_cartao.isnull().sum()
dados_cartao.fillna(dados_cartao.ValorCompraAnual.mean(), inplace=True)
dados_cartao.isnull().sum()
plotar_box(dados_cartao)
dados_cartao.describe().round()
limite_superior = dados_cartao.quantile(.75) + 1.5 * (dados_cartao.quantile(.75) - dados_cartao.quantile(.25))
limite_superior
limite_inferior = dados_cartao.quantile(.25) - 1.5 * (dados_cartao.quantile(.75) - dados_cartao.quantile(.25))
limite_inferior
dados_cartao[dados_cartao[list(limite_superior.index)] >  limite_superior].describe().round().iloc[0]
dados_cartao[dados_cartao[list(limite_superior.index)] <  limite_inferior].describe().round().iloc[0]
dados_cartao['Idade'].values[0]
def plotar_dist(dataframe):
    
    dataframe = dataframe.drop(['ID', 
                    'PerfilEconomico', 'Sexo', 'Idade', 'PerfilCompra',
                   'UF', 'CidadeResidencia',
                   'RegiaodoPais', 'MesesDesempregado',
                   'Cancelamento'],axis=1)
    
    for i in dataframe.columns:
        if (dataframe.dtypes[i] == 'int64') or (dataframe.dtypes[i] == 'float64'):
            print('Distribuição referente ao: ', i)
            sns.distplot(dataframe[i])
            plt.show()
plotar_dist(dados_cartao)
from sklearn.preprocessing import MinMaxScaler
dados_cartao.dtypes.index
dados_cartao.dtypes
df_treino_kmeans = dados_cartao[['PerfilEconomico', 'Idade', 
                                 'PerfilCompra','ValorCompraAnual', 
                                 'GastoMax', 'GastoMedio', 'NumeroComprasOnline', 
                                 'MesesDesempregado', 'Anuidade']]
minmax = MinMaxScaler()
minmax.fit(df_treino_kmeans)
dados_normalizados = minmax.transform(df_treino_kmeans)
df_treino_kmeans_normalizado = pd.DataFrame(dados_normalizados, columns=df_treino_kmeans.columns)
df_treino_kmeans_normalizado
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, n_init=10)
dados_cartao['Cluster'] = kmeans.fit_predict(dados_normalizados)
dados_cartao.groupby("Cluster").aggregate("mean").plot.bar(figsize=(10,7.5), )
from sklearn.metrics import silhouette_score
def grafico_cotovelo(data, cor, nome_curva):
    plt.plot(data, marker='o', linestyle='dashed', color=cor)
    plt.xlabel('Números de Cluster')
    plt.ylabel('Score')
    plt.title('Curva {}'.format(nome_curva))
    plt.show()

    
    
def calcular_cotovelo(data, cor, nome_curva):
    lista_score = []
    for n in range(2, 50, 2):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        lista_score.append(kmeans.inertia_)
        
    return grafico_cotovelo(lista_score, cor, nome_curva)



def calcular_silhouette(data, cor, nome_curva):
    lista_score = []
    for n in range(2, 50, 2):
        kmeans = KMeans(n_clusters=n)
        # kmeans.fit(X=data)
        lista_score.append(silhouette_score(data, kmeans.fit_predict(X=data)))
        
    return grafico_cotovelo(lista_score, cor, nome_curva)

calcular_cotovelo(dados_normalizados, 'b', 'Cotovelo')


calcular_silhouette(dados_normalizados, 'r', 'Silhouette')
dados_cartao.head(2)
dados_cartao.groupby("Cluster").max().round()
dados_cartao[dados_cartao > limite_superior].describe().round()
dados_cartao[dados_cartao.ValorCompraAnual > 426941].round()
dados_cartao[dados_cartao.ValorCompraAnual > 3000000].index
dados_cartao.drop([258, 679, 737], axis=0, inplace=True)
dados_cartao[dados_cartao.ValorCompraAnual > 3000000]
dados_cluster2 = dados_cartao[dados_cartao.Cluster == 1]
dados_cluster2.head(2)
dados_cluster2.describe().round()
dados_cluster2[(dados_cluster2.ValorCompraAnual > 500000) & 
               (dados_cluster2.ValorCompraAnual > 30000) & 
               (dados_cluster2.NumeroComprasOnline > 4000)]
excluir_registro = dados_cluster2[(dados_cluster2.ValorCompraAnual > 500000) & 
                   (dados_cluster2.ValorCompraAnual > 30000) & 
                   (dados_cluster2.NumeroComprasOnline > 4000)].index.values
dados_cluster2.drop(index=excluir_registro, axis=0, inplace = True)
plotar_box(dados_cluster2)
plt.subplots(figsize=(10, 10))
sns.heatmap(dados_cartao.corr(),annot=True);
dados_cartao.nunique()
dados_cartao.dtypes
variaveis_dum = pd.get_dummies(dados_cartao,
                              columns = ['PerfilEconomico', 'Sexo', 'RegiaodoPais', 'Cluster'],
                              drop_first=True,
                              prefix=['PerfilEconomico', 'Sexo', 'RegiaodoPais', 'Cluster'],
                              prefix_sep='_')
variaveis_dum.head()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
var_uf = label_encoder.fit_transform(variaveis_dum['UF'])
var_uf = pd.DataFrame(var_uf, columns=['UF_LE'])
var_uf.head(2)
var_cidade_residencia = label_encoder.fit_transform(variaveis_dum['CidadeResidencia'])
var_cidade_residencia = pd.DataFrame(var_cidade_residencia, columns=['CidadeResidencia_LE'])
var_cidade_residencia.head(2)
dados_tratados = pd.merge(variaveis_dum, var_uf, left_index=True, right_index=True)
dados_tratados = pd.merge(variaveis_dum, var_cidade_residencia, left_index=True, right_index=True)
dados_tratados.head()
dados_tratados.columns
dados_tratados.head(1)
dados_selecionados = dados_tratados[['Idade', 'PerfilCompra', 'ValorCompraAnual', 
                                     'GastoMax', 'GastoMedio', 'NumeroComprasOnline', 
                                     'MesesDesempregado', 'Anuidade', 
                                     'Cancelamento', 'PerfilEconomico_2',
                                    'PerfilEconomico_3', 'Sexo_mulher', 'RegiaodoPais_Região Nordeste',
                                   'RegiaodoPais_Região Norte', 'RegiaodoPais_Região Sudeste',
                                   'RegiaodoPais_Região Sul', 'Cluster_1', 'Cluster_2','Cluster_3',
                                   'CidadeResidencia_LE']]
dados_selecionados.head()
features = dados_selecionados.drop(columns=['Cancelamento'])
target = dados_selecionados['Cancelamento']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.preprocessing import MinMaxScaler
features_normalizadas = MinMaxScaler().fit_transform(features)
chi_selector = SelectKBest(chi2)
chi_selector.fit(features_normalizadas, target)
chi_support = chi_selector.get_support()
chi_feature = features.loc[:, chi_support].columns.tolist()

print(str(len(chi_feature)), 'Quantidade de variáveis selecionadas')
print(chi_feature)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(),
                   step=10,
                   n_features_to_select=5)

rfe_selector.fit(features_normalizadas, target)
rfe_support = rfe_selector.get_support()
rfe_feature = features.loc[:, rfe_support].columns.tolist()

print(str(len(rfe_feature)), 'Sao as quantidades de vars selacionadas')
print(rfe_feature)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100,
                                                     random_state=42))

rf_selector.fit(features_normalizadas, target)
rf_support = rf_selector.get_support()
rf_feature = features.loc[:,rf_support].columns.tolist()

print(str(len(rf_feature)), 'Sao as quantidades de vars selacionadas')
print(rf_feature)
feature_Selection_df = pd.DataFrame({'variaveis': features.columns,
                                     'Chi2': chi_support,
                                     'RFE': rfe_support,
                                     'RF': rf_support})
# count de quanto foi selecionada para cada um dos algs
feature_Selection_df['Total'] = np.sum(feature_Selection_df, axis=1)

#sort
feature_Selection_df = feature_Selection_df.sort_values(['Total', 'variaveis'], ascending=False)
feature_Selection_df.index = range(1, len(feature_Selection_df) + 1)
feature_Selection_df
explicativas = features[['PerfilCompra','MesesDesempregado', 'Anuidade',
                       'PerfilEconomico_2', 'PerfilEconomico_3', 'Sexo_mulher',
                       'RegiaodoPais_Região Nordeste',
                       'RegiaodoPais_Região Sudeste', 'RegiaodoPais_Região Sul', 'Cluster_1',
                       'Cluster_2', 'Cluster_3', 'CidadeResidencia_LE']]
explicativas.head(2)
target.head(2)
explicativas_normalizadas = MinMaxScaler().fit_transform(explicativas)
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(explicativas_normalizadas, target, test_size=0.3, random_state=42)
print('Shape treino: ', X_treino.shape)
print('Shape teste: ', X_teste.shape)
print('Shape treino: ', y_treino.shape)
print('Shape teste: ', y_teste.shape)
from sklearn.model_selection import GridSearchCV
grid_rf ={
    'n_estimators': [10,20,50,100,200,500],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(random_state=42)
grid_rl = {'C': np.logspace(-3,3,7), 'penalty':['l2']}
grid_rl
grid_reg_log = GridSearchCV(reg,
                              grid_rl,
                              scoring='accuracy',
                              cv=10)
grid_reg_log.fit(X_treino, y_treino)
grid_reg_log.best_params_
print('Score na base de treino', grid_reg_log.score(X_treino, y_treino))
print('Score na base de teste', grid_reg_log.score(X_teste, y_teste))
from sklearn.tree import DecisionTreeClassifier
arvore = DecisionTreeClassifier()
grid_arvore = {
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [1,2,3,4,5,10]    
}
grid_arvore = GridSearchCV( arvore,
                            grid_arvore,
                            scoring='accuracy',
                            cv=10)
grid_arvore.fit(X_treino, y_treino)
grid_arvore.best_params_
print('Score na base de treino', grid_arvore.score(X_treino, y_treino))

print('Score na base de teste', grid_arvore.score(X_teste, y_teste))
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
grid_randon_forest ={
    'n_estimators': [10,20,50,100,200,500],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}
grid_random_forest = GridSearchCV(random_forest,
                              grid_randon_forest,
                              scoring='accuracy',
                              cv=10)
grid_random_forest.fit(X_treino, y_treino)
grid_random_forest.best_params_
grid_random_forest.best_score_
print('Score na base de treino', grid_random_forest.score(X_treino, y_treino))

print('Score na base de teste', grid_random_forest.score(X_teste, y_teste))
from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting = GradientBoostingClassifier(random_state=42)
dic_grid_gradient_boosting = {
    'min_samples_leaf':[2, 5],
    'max_depth':[3, 5, 8, 10, 15],
    'n_estimators':[5, 10, 20, 50, 100]
}
grid_gradient_boosting = GridSearchCV(gradient_boosting, 
                                      dic_grid_gradient_boosting, 
                                      scoring='accuracy', cv=5)
grid_gradient_boosting
grid_gradient_boosting.fit(X_treino, y_treino)
grid_gradient_boosting.best_params_
print('Score na base de treino', grid_gradient_boosting.score(X_treino, y_treino))

print('Score na base de teste', grid_gradient_boosting.score(X_teste, y_teste))
grid_gradient_boosting.best_score_
from sklearn.metrics import f1_score
f1_score(y_treino, grid_gradient_boosting.predict(X_treino))
import pickle
pickle.dump(grid_gradient_boosting, open('analise_de_credito.pkl', 'wb'))
