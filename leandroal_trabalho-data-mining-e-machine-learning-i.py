import pandas as pd

pd.options.display.float_format = '{:.2f}'.format

import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix

import seaborn as sns

import os



from matplotlib import pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans

import plotly as py

import plotly.graph_objs as go

import numpy as np

import pandas as pd

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Leitura do conjunto de dados

final = pd.read_csv('/kaggle/input/eleicao_2018_apuracao_final.csv', sep=';', encoding='latin-1')



# Visualização das informações sobre as variáveis

final.info()
def filtro(dataset):# Filtro por Estado e Situação

    dataset = dataset[dataset['SG_UF'].isin(['SP', 'MG'])  & (dataset['DS_SITUACAO_CANDIDATURA'] == 'APTO')]



    # Filtro por Situação Deferida

    dataset = dataset[dataset['DS_DETALHE_SITUACAO_CAND'].isin(['DEFERIDO','DEFERIDO COM RECURSO'])]



    # Filtro por Cargo

    dataset = dataset[dataset['DS_CARGO'].isin(['SENADOR','DEPUTADO FEDERAL','DEPUTADO ESTADUAL'])]



    # Visualização das informações sobre as variáveis

    return dataset
final = filtro(final)
def imputa_dados_despesa(data):

    dataset = data.copy()

    # Separando um conjunto de dados para modelagem

    dataset_desp_pred = dataset[(dataset['Receita_Total'].notna()) & (dataset['Despesa_Total'].notna())]



    X = dataset_desp_pred.iloc[:, -2:-1].values

    

    y = dataset_desp_pred.iloc[:, -1].values



    # Dividindo o conjunto de dados em treino e teste

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12345)



    # Ajustando o modelo de Árvore de Regressão

    regressor = DecisionTreeRegressor(random_state=12345)

    regressor.fit(X_train, y_train)



    # Predizendo resultados de teste

    y_pred = regressor.predict(X_test)



    # Avaliando o índice de acertos

    print('Valor do r²: ', round(r2_score(y_test , y_pred),2))



    # Inputando DESPESAS missing onde RECEITA NOT NULL

    dataset_desp_inpute = dataset[(dataset['Receita_Total'].notna()) & (dataset['Despesa_Total'].isna())]

    X = dataset_desp_inpute.iloc[:, -2:-1].values

    y = dataset_desp_inpute.iloc[:, -1].values



    # Resultado dos valores preditos

    dataset['Despesa_Total'][(dataset['Receita_Total'].notna()) & (dataset['Despesa_Total'].isna())] =  regressor.predict(X)



    return dataset
def imputa_dados_receita(data):

    dataset = data.copy()

    # Separando um conjunto de dados para modelagem

    dataset_rec_pred = dataset[(dataset['Despesa_Total'].notna()) & (dataset['Receita_Total'].notna())]

    X = dataset_rec_pred.iloc[:, -1:].values

    y = dataset_rec_pred.iloc[:, -2].values



    # Dividindo o conjunto de dados em treino e teste

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12345)



    # Ajustando o modelo de Árvore de Regressão

    regressor = DecisionTreeRegressor(random_state=12345)

    regressor.fit(X_train, y_train)



    # Predizendo resultados de teste

    y_pred = regressor.predict(X_test)



    # Evaluating

    print('Valor do r²: ', round(r2_score(y_test , y_pred),2))

    # =============================================================================



    # Inputando DESPESAS missing onde RECEITA NOT NULL

    dataset_desp_inpute = dataset[(dataset['Despesa_Total'].notna()) & (dataset['Receita_Total'].isna())]

    X = dataset_desp_inpute.iloc[:, -1:].values

    y = dataset_desp_inpute.iloc[:, -2].values





    dataset['Receita_Total'][(dataset['Despesa_Total'].notna()) & (dataset['Receita_Total'].isna())] =  regressor.predict(X)

    return dataset
def imputa_media(data):

    dataset = data.copy()

    receitas_inpute = dataset.groupby('SG_PARTIDO', as_index=False)['Receita_Total'].mean()

    despesas_inpute = dataset.groupby('SG_PARTIDO', as_index=False)['Despesa_Total'].mean()





    dataset = dataset.merge(receitas_inpute, how='left', on='SG_PARTIDO')

    dataset = dataset.merge(despesas_inpute, how='left', on='SG_PARTIDO')



    dataset = dataset.rename(columns={'Receita_Total_x':'Receita_Total', 'Despesa_Total_x':'Despesa_Total'})

    dataset['Receita_Total'] = np.where(dataset['Receita_Total'].isna(), dataset['Receita_Total_y'], dataset['Receita_Total'])

    dataset['Despesa_Total'] = np.where(dataset['Despesa_Total'].isna(), dataset['Despesa_Total_y'], dataset['Despesa_Total'])



    dataset = dataset.iloc[:,0:-2]

    return dataset
final = imputa_dados_despesa(final)



final = imputa_dados_receita(final)



final = imputa_media(final)
situacoes = {

    0:"ELEITO",

    1:"NÃO ELEITO"

}
from sklearn import tree

import graphviz 



class ModelMaker():

    def _load_data(self, data):

        dataset = data.copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset.loc[:, dataset.columns.difference(['target', 'situacao',"SG_PARTIDO", "DS_CARGO", "NM_CANDIDATO"])], dataset["target"], test_size=0.3, random_state=42)

        

        

    def __init__(self, sklearn_load_ds):

        self._load_data(sklearn_load_ds)

    

    

    def classify(self, model=LogisticRegression(random_state=42)):

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        self.model = model

        

        class_names = np.unique([situacoes[i] for i in self.y_train])

        feature_names = self.X_train.columns

        

        dot_data = tree.export_graphviz(model, out_file=None,

                                        class_names=class_names,

                                        feature_names=feature_names,

                                        max_depth=5)

        graph = graphviz.Source(dot_data)

        

        png_bytes = graph.pipe(format='png')

        with open('dtree_pipe.png','wb') as f:

            f.write(png_bytes)



        from IPython.display import Image

        display(Image(png_bytes))

   

        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))

        return self



    def predict(self, data):

        dataset = data.copy()

        y_labels = self.clf.predict(dataset.loc[:, dataset.columns.difference(['target', 'situacao',"SG_PARTIDO", "DS_CARGO", "NM_CANDIDATO"])])

        dataset["km_clust"] = y_labels

        dataset["predict"] = self.model.predict(dataset.loc[:, dataset.columns.difference(['target', 'situacao',"SG_PARTIDO", "DS_CARGO", "NM_CANDIDATO"])])

        return dataset

        

    def cluster(self, output='add'):

        n_clusters = len(np.unique(self.y_train))

        clf = KMeans(n_clusters = n_clusters, random_state=42)

        clf.fit(self.X_train)

        y_labels_train = clf.labels_

        y_labels_test = clf.predict(self.X_test)

        self.clf = clf

        

        if output == 'add':

            self.X_train['km_clust'] = y_labels_train

            self.X_test['km_clust'] = y_labels_test

        elif output == 'replace':

            self.X_train = y_labels_train[:, np.newaxis]

            self.X_test = y_labels_test[:, np.newaxis]

        else:

            raise ValueError('output should be either add or replace')

        return self
final["situacao"] = np.where(final['DS_SIT_TOT_TURNO'] == 'NÃO ELEITO', 'NÃO ELEITO', 'ELEITO')

final["target"]   =  pd.factorize(final['situacao'], sort=True)[0]

dataset = final[['Despesa_Total', 'Receita_Total', 'Votos', "target", "situacao", "SG_PARTIDO", "DS_CARGO", "NM_CANDIDATO"]]


dataset.groupby(["target", "situacao"]).nunique()
## realizando cluster e aplicando um modelo DecisionTreeClassifier



model = ModelMaker(dataset).cluster(output="add").classify(model=DecisionTreeClassifier(max_depth=30))

X = model.predict(dataset)
def plot_kmeans(data, label1, label2, titulo):

    dataset = data.copy()

    fig, ax = plt.subplots(2, 1, figsize=(16,12))



    dataset['situacao'] = dataset['target'].map(situacoes)

    dataset['km_clust'] = dataset['km_clust'].map({ 0:"CLUSTER 1", 1:"CLUSTER 2" })



    sns.scatterplot(x=label1, y=label2, hue="situacao", data=dataset, ax=ax[0], s=100, color=".2")

    

    sns.scatterplot(x=label1, y=label2, hue="km_clust", data=dataset, ax=ax[1], s=100, color=".2")





    ax[0].set_xlabel(label1, fontsize=15)

    ax[0].set_ylabel(label2, fontsize=15)

    ax[1].set_xlabel(label1, fontsize=15)

    ax[1].set_ylabel(label2, fontsize=15)

    ax[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)

    ax[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)

    ax[0].set_title('Atual', fontsize=18)

    ax[1].set_title('Cluster', fontsize=18)

    fig.suptitle(titulo, fontsize=20)

plot_kmeans(X, "Receita_Total", "Despesa_Total","Receita x Despesas")
plot_kmeans(X, "Receita_Total", "Votos","Receita x Votos")
plot_kmeans(X, "Despesa_Total", "Votos","Despesa x Votos")
# mapeando qual a situação para o valor predit

X['situacao_predita'] = X['predict'].map(situacoes)



# criando uma tabela cruzada

pd.crosstab(X["situacao"], X["situacao_predita"])
df_sem_definicao = pd.read_csv('/kaggle/input/eleicao_2018_sem_definicao.csv', sep=';', encoding='latin-1')

df_sem_definicao = filtro(df_sem_definicao)

df_sem_definicao = imputa_media(df_sem_definicao)

df_sem_definicao['Receita_Total'][df_sem_definicao['Receita_Total'].isna()] =  df_sem_definicao["Receita_Total"].mean()

df_sem_definicao['Despesa_Total'][df_sem_definicao['Despesa_Total'].isna()] =  df_sem_definicao["Despesa_Total"].mean()



df_sem_definicao["situacao"] = np.where(df_sem_definicao['DS_Sit_Tot_Turno_OLD'] == 'NÃO ELEITO', 'NÃO ELEITO', 'ELEITO')

df_sem_definicao["target"] =  np.where(df_sem_definicao['situacao'] == 'NÃO ELEITO', 1, 0)



df_sem_definicao = df_sem_definicao[['Despesa_Total', 'Receita_Total', 'Votos', "target", "situacao", "SG_PARTIDO", "DS_CARGO", "NM_CANDIDATO"]]
X = model.predict(df_sem_definicao)
plot_kmeans(X, "Receita_Total", "Despesa_Total","Receita x Despesas")
plot_kmeans(X, "Receita_Total", "Votos","Receita x Votos")
plot_kmeans(X, "Despesa_Total", "Votos","Despesa x Votos")
# mapeando qual a situação para o valor predit

X['situacao_predita'] = X['predict'].map(situacoes)



# criando uma tabela cruzada

pd.crosstab(X["situacao"], X["situacao_predita"])
pd.set_option('display.max_rows', len(X))



X[["NM_CANDIDATO", "SG_PARTIDO", "DS_CARGO", "situacao","situacao_predita"]]
# =============================================================================

# Utilizando o dataset "final" como modelo

# =============================================================================

final_modelo = final[['DS_CARGO','Votos','Receita_Total', 'Despesa_Total','DS_SIT_TOT_TURNO']]



final_modelo['DS_SIT_TOT_TURNO'] = np.where(final_modelo['DS_SIT_TOT_TURNO'].isin(['NÃO ELEITO']),0,1)



# =============================================================================

#  Variáveis Dummies

# =============================================================================

df_modelo = pd.get_dummies(final_modelo, columns=['DS_CARGO'], drop_first=True)



colunas = df_modelo.columns.tolist()

cols = colunas[4:] + colunas[:4]

df_modelo = df_modelo[cols]

 

X = df_modelo.iloc[:, :-1].values

y = df_modelo.iloc[:, -1].values



# =============================================================================

# Teste e Treino

# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12345)



# =============================================================================

# Normalização dos dados

# =============================================================================

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# =============================================================================

# Cassificador Random Forest

# =============================================================================

classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state = 12345)

classifier.fit(X_train, y_train)



# =============================================================================

# Predizendo resultados

# =============================================================================

y_pred_final = classifier.predict(X_test)



# =============================================================================

# Matrix de Confusão

# =============================================================================



cm = confusion_matrix(y_test, y_pred_final)



esp = cm[0,0] / (cm[0,0] + cm[1,0])

sens = cm[1,1] / (cm[0,1] + cm[1,1])



print('Sensibilidade: ' + str(round(sens,2)) +'\nEspecificidade: ' + str(round(esp,2)))

sem_def = pd.read_csv('/kaggle/input/eleicao_2018_sem_definicao.csv', sep=';', encoding='latin-1')



# =============================================================================

# Aplicando Filtros

# =============================================================================

sem_def_filtrado = filtro(sem_def)

col = sem_def_filtrado.columns.to_list()

col = col[:-4] + col[-1:] + col[-4:-1]

sem_def_filtrado = sem_def_filtrado[col]



sem_def_filtrado = imputa_dados_despesa(sem_def_filtrado)

sem_def_filtrado = imputa_media(sem_def_filtrado)



# Filtrando onde não existe média de valores por partido, pois todos do partido são missing values

sem_def_filtrado['Receita_Total'] = np.where(sem_def_filtrado['Receita_Total'].isna(), sem_def_filtrado['Receita_Total'].mean(), sem_def_filtrado['Receita_Total'])

sem_def_filtrado['Despesa_Total'] = np.where(sem_def_filtrado['Despesa_Total'].isna(), sem_def_filtrado['Despesa_Total'].mean(), sem_def_filtrado['Despesa_Total'])



# =============================================================================

# Definindo o Modelo

# =============================================================================

modelo_pred = sem_def_filtrado[['DS_CARGO','Votos','Receita_Total', 'Despesa_Total']]

modelo_pred = pd.get_dummies(modelo_pred, columns=['DS_CARGO'], drop_first=True)

colunas = modelo_pred.columns.tolist()

cols = colunas[3:] + colunas[:3]

modelo_pred = modelo_pred[cols]

 

X = modelo_pred.iloc[:, :].values



# =============================================================================

# Normalizando os dados

# =============================================================================

sc = StandardScaler()

X = sc.fit_transform(X)



# =============================================================================

# Predizendo os resultados

# =============================================================================

y_pred_sem_def = classifier.predict(X)



# =============================================================================

# Criando variável de teste para avaliação do resultado

# =============================================================================

sem_def_filtrado['y_test'] = np.where(sem_def_filtrado['DS_Sit_Tot_Turno_OLD'].isin(['NÃO ELEITO']),0,1)

y_test = sem_def_filtrado['y_test'].values



# =============================================================================

# Matrix de Confusão

# =============================================================================

cm = confusion_matrix(y_test, y_pred_sem_def)



esp = cm[0,0] / (cm[0,0] + cm[1,0])

sens = cm[1,1] / (cm[0,1] + cm[1,1])



print('Sensibilidade: ' + str(round(sens,2)) +'\nEspecificidade: ' + str(round(esp,2)))

print('Acurácia: ', round(accuracy_score(y_test, y_pred_sem_def),2))
sem_def_filtrado['Resultado_Final'] = y_pred_sem_def

evalu = sem_def_filtrado[['DS_CARGO','Votos', 'Receita_Total', 'Despesa_Total', 'DS_Sit_Tot_Turno_OLD','y_test','Resultado_Final']]

evalu.head(15)
sem_def_filtrado[['NM_CANDIDATO', 'SG_PARTIDO']][sem_def_filtrado['Resultado_Final'] == 1]