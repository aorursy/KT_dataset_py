import pandas as pd

import numpy as np



from sklearn.pipeline import Pipeline

from sklearn import preprocessing

from sklearn.preprocessing import FunctionTransformer

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator
#verifica se o arquivo está sendo executado diretamente pelo usuário. caso contrário, está sendo chamado por outro notebook. 

execucao_direta = (__name__ == '__main__') and ('__file__' not in globals())



#indica se os passos do pipeline devem ser executados nesse caderno

#True durante o desenvolvimento, False quando estiver sendo utilizado para aprendizado

executar_pipeline = False
# retorna uma lista de True ou False indicando as posições onde um elemento de list1 está contido em list2

# é útil para retornar todas as linhas de um dataframe cuja coluna list1 assuma qualquer dos valores em list2

# exemplo: dataframe[eachIn(dataframe.coluna, lista)]

def eachIn(list1, list2):

    retorno = []

    for item in list1:

        retorno.append(True if (item in list2) else False)

    return retorno



# retorna as linhas do dataframe cujo valor da coluna esteja contido na lista

def dfRowsEachIn(dataframe, column, value_list):

    return dataframe[eachIn(dataframe[column], value_list)]



# retorna uma lista de True ou False indicando as posições onde um elemento de list1 não está contido em list2

# é útil para retornar todas as linhas de um dataframe cuja coluna list1 nunca assuma qualquer dos valores em list2

# exemplo: dataframe[notIn(dataframe.coluna, lista)]

def notIn(list1, list2):

    retorno = []

    for item in list1:

        retorno.append(False if (item in list2) else True)

    return retorno
candidatos_original = pd.read_csv('../input/dados-dos-candidatos-da-eleio-de-2014/consulta_cand_2014_BRASIL.csv', sep=';', encoding='iso-8859-1', low_memory=False)

candidatos = candidatos_original.copy()

candidatos.head()
candidatos.shape
candidatos.columns
candidatos.describe()
colunas_numericas = candidatos.describe().columns

colunas_numericas
colunas_textuais = np.setdiff1d(candidatos.columns, colunas_numericas)

colunas_textuais
#essa lista vai guardar todas as colunas que não serão utilizadas nesse estudo

colunas_desnecessarias = ['DT_GERACAO', 'HH_GERACAO', 'ANO_ELEICAO', 'CD_TIPO_ELEICAO',

       'CD_ELEICAO', 'DS_ELEICAO', 'DT_ELEICAO', 'NM_UE', 'CD_CARGO',

       'NR_CANDIDATO', 'NM_CANDIDATO', 'NM_URNA_CANDIDATO',

       'NM_SOCIAL_CANDIDATO', 'NR_CPF_CANDIDATO', 'NM_EMAIL',

       'CD_SITUACAO_CANDIDATURA', 'CD_DETALHE_SITUACAO_CAND', 'NR_PARTIDO',

       'NM_PARTIDO', 'SQ_COLIGACAO', 'NM_COLIGACAO', 'DS_COMPOSICAO_COLIGACAO',

       'CD_NACIONALIDADE', 'CD_MUNICIPIO_NASCIMENTO', 'DT_NASCIMENTO',

       'NR_TITULO_ELEITORAL_CANDIDATO', 'CD_GENERO', 'CD_GRAU_INSTRUCAO',

       'CD_ESTADO_CIVIL', 'CD_COR_RACA', 'CD_OCUPACAO', 'CD_SIT_TOT_TURNO',

       'NR_PROTOCOLO_CANDIDATURA', 'NR_PROCESSO ','SQ_CANDIDATO']
candidatos[candidatos.NM_TIPO_ELEICAO=='ELEIÇÃO SUPLEMENTAR'].head()
len(candidatos)
candidatos = candidatos.drop(candidatos[candidatos.NM_TIPO_ELEICAO=='ELEIÇÃO SUPLEMENTAR'].index, axis="rows")

len(candidatos)
colunas_desnecessarias.append('NM_TIPO_ELEICAO')
candidatos.loc[:,'NR_DESPESA_MAX_CAMPANHA'].unique()
candidatos.groupby('NR_DESPESA_MAX_CAMPANHA').NR_DESPESA_MAX_CAMPANHA.count()
#candidatos['SEM_DESPESA_MAX_CAMPANHA'] = [i*1 for i in candidatos['NR_DESPESA_MAX_CAMPANHA']<0]

#candidatos['SEM_DESPESA_MAX_CAMPANHA']
#candidatos['NR_DESPESA_MAX_CAMPANHA'] = [i if i>0 else 0 for i in candidatos['NR_DESPESA_MAX_CAMPANHA']]

#candidatos['NR_DESPESA_MAX_CAMPANHA']
#media = candidatos['NR_DESPESA_MAX_CAMPANHA'].mean()

#desv_pad = candidatos['NR_DESPESA_MAX_CAMPANHA'].std()

#candidatos['NR_DESPESA_MAX_CAMPANHA'] = [(i-media)/desv_pad for i in candidatos['NR_DESPESA_MAX_CAMPANHA']]
#candidatos['NR_DESPESA_MAX_CAMPANHA'].describe()
class Transformer_nr_despesa:

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        self.media = X['NR_DESPESA_MAX_CAMPANHA'].mean()

        self.desv_pad = X['NR_DESPESA_MAX_CAMPANHA'].std()

        return self

        

    def transform(self, X, y=None):

        candidatos = X.copy()



        #adicionando uma coluna para indicar quando o valor de NR_DESPESA_MAX_CAMPANHA não estava disponível

        candidatos['SEM_DESPESA_MAX_CAMPANHA'] = [i*1 for i in candidatos['NR_DESPESA_MAX_CAMPANHA']<0]



        #s valores negativos da coluna NR_DESPESA_MAX_CAMPANHA serão zerados

        candidatos['NR_DESPESA_MAX_CAMPANHA'] = [i if i>0 else 0 for i in candidatos['NR_DESPESA_MAX_CAMPANHA']]



        #Normalizando os valores

        candidatos['NR_DESPESA_MAX_CAMPANHA'] = [(i-self.media)/self.desv_pad for i in candidatos['NR_DESPESA_MAX_CAMPANHA']]

        return candidatos
if(executar_pipeline):

    pipeline = Pipeline(steps=[('nr_despesa',Transformer_nr_despesa())])

    pipeline.fit(candidatos)

    candidatos = pipeline.transform(candidatos)
#verificando resultado da transformação, quando aplicada

candidatos['NR_DESPESA_MAX_CAMPANHA'].describe().apply(lambda x: format(x, 'f'))
#adicionando a transformação no pipeline

pipe_steps = [('nr_despesa',Transformer_nr_despesa())]
#listando os candidatos com idade nula

candidatos[candidatos.NR_IDADE_DATA_POSSE.isna()].SQ_CANDIDATO
candidatos.NR_IDADE_DATA_POSSE.mean()
class Transformer_nr_idade:

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        self.media = X['NR_IDADE_DATA_POSSE'].mean()

        return self

        

    def transform(self, X, y=None):

        candidatos = X.copy()

        candidatos.loc[candidatos.NR_IDADE_DATA_POSSE.isna(),'NR_IDADE_DATA_POSSE'] = self.media

        return candidatos
if(executar_pipeline):

    pipeline = Pipeline(steps=[('nr_idade',Transformer_nr_idade())])

    pipeline.fit(candidatos)

    candidatos = pipeline.transform(candidatos)
#mostrando as idades nulas depois da operação

if(execucao_direta):

    print(f"Idades nulas:{candidatos.NR_IDADE_DATA_POSSE.isna().sum()}")
#adicionando a transformação para o pipeline

pipe_steps.append(('nr_idade',Transformer_nr_idade()))
candidatos.groupby('DS_CARGO').DS_CARGO.count()
candidatos = candidatos.drop(candidatos[candidatos['DS_CARGO'] == '1º SUPLENTE'].index, axis="rows")

candidatos = candidatos.drop(candidatos[candidatos['DS_CARGO'] == '2º SUPLENTE'].index, axis="rows")

candidatos = candidatos.drop(candidatos[candidatos['DS_CARGO'] == 'VICE-GOVERNADOR'].index, axis="rows")

candidatos = candidatos.drop(candidatos[candidatos['DS_CARGO'] == 'VICE-PRESIDENTE'].index, axis="rows")
candidatos.groupby('DS_CARGO').DS_CARGO.count()
candidatos.groupby('DS_COR_RACA').DS_COR_RACA.count()
candidatos.groupby('DS_DETALHE_SITUACAO_CAND').DS_DETALHE_SITUACAO_CAND.count()
candidatos.groupby('DS_ESTADO_CIVIL').DS_ESTADO_CIVIL.count()
candidatos.groupby('DS_GENERO').DS_GENERO.count()
candidatos.groupby('DS_GRAU_INSTRUCAO').DS_GRAU_INSTRUCAO.count()
candidatos.groupby('DS_NACIONALIDADE').DS_NACIONALIDADE.count()
candidatos.DS_OCUPACAO.value_counts()
candidatos.DS_OCUPACAO.describe()
(candidatos.DS_OCUPACAO == None).sum()
# esse transformador modifica as ocupações menos frequentes para OUTROS

class Transformer_ds_ocupacao(BaseEstimator):

    def __init__(self, min_value=1):

        self.min_value = min_value

    

    def fit(self, X, y=None):

        ocupacoes = X.DS_OCUPACAO.value_counts()

        self.ocupacoes_comuns = ocupacoes[ocupacoes>=self.min_value].index

        return self

        

    def transform(self, X, y=None):

        candidatos = X.copy()       

        candidatos.loc[notIn(candidatos.DS_OCUPACAO, self.ocupacoes_comuns),'DS_OCUPACAO'] = "OUTROS"

        return candidatos
if(executar_pipeline):

    pipeline = Pipeline(steps=[('ds_ocupacao',Transformer_ds_ocupacao(min_value=1))])

    pipeline.fit(candidatos)

    candidatos = pipeline.transform(candidatos)
#adicionando a transformação ao pipeline

pipe_steps.append(('ds_ocupacao',Transformer_ds_ocupacao()))
candidatos.groupby('DS_SITUACAO_CANDIDATURA').DS_SITUACAO_CANDIDATURA.count()
inaptos = candidatos[candidatos.DS_SITUACAO_CANDIDATURA=='INAPTO']

inaptos.groupby('DS_SIT_TOT_TURNO').DS_SIT_TOT_TURNO.count()
candidatos.groupby('NM_MUNICIPIO_NASCIMENTO').NM_MUNICIPIO_NASCIMENTO.count()
candidatos.NM_MUNICIPIO_NASCIMENTO.describe()
candidatos.loc[:,['CD_MUNICIPIO_NASCIMENTO','NM_MUNICIPIO_NASCIMENTO']]
candidatos.CD_MUNICIPIO_NASCIMENTO.describe()
colunas_desnecessarias.append('NM_MUNICIPIO_NASCIMENTO')
candidatos.groupby('SG_PARTIDO').SG_PARTIDO.count()
candidatos.groupby('SG_UE').SG_UE.count()
candidatos.groupby('SG_UF').SG_UF.count()
candidatos[candidatos.SG_UE != candidatos.SG_UF].SG_UF.sum()
colunas_desnecessarias.append('SG_UF')
candidatos.groupby('SG_UF_NASCIMENTO').SG_UF_NASCIMENTO.count()
#candidatos.SG_UF_NASCIMENTO = candidatos.SG_UF_NASCIMENTO.replace('ZZ','Não divulgável')
class Transformer_uf_nascimento:

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        candidatos = X.copy()



        #trocando os valores ZZ por um rótulo mais inteligível

        candidatos.SG_UF_NASCIMENTO = candidatos.SG_UF_NASCIMENTO.replace('ZZ','Não divulgável')



        return candidatos
if(executar_pipeline):

    pipeline = Pipeline(steps=[('uf_nascimento',Transformer_uf_nascimento())])

    pipeline.fit(candidatos)

    candidatos = pipeline.transform(candidatos)
candidatos[candidatos.SG_UF_NASCIMENTO=='ZZ'].SG_UF_NASCIMENTO.count()
#colocando a transformação no pipeline

pipe_steps.append(('uf_nascimento',Transformer_uf_nascimento()))
candidatos.groupby('ST_DECLARAR_BENS').ST_DECLARAR_BENS.count()
candidatos.groupby('ST_REELEICAO').ST_REELEICAO.count()
candidatos.groupby('TP_ABRANGENCIA').TP_ABRANGENCIA.count()
len(candidatos)
colunas_desnecessarias.append('TP_ABRANGENCIA')
candidatos.groupby('TP_AGREMIACAO').TP_AGREMIACAO.count()
candidatos.groupby('DS_SIT_TOT_TURNO').DS_SIT_TOT_TURNO.count()
len(candidatos)
#descobrindo o número de candidatos que aparecem mais de uma vez no dataset

contagem = candidatos.SQ_CANDIDATO.value_counts()

repetidos = contagem[contagem==2]

len(repetidos)
#verificando a situação de totalização apenas dos candidatos repetidos

cand_repetidos = candidatos[eachIn(candidatos.SQ_CANDIDATO, repetidos)]

cand_repetidos.DS_SIT_TOT_TURNO.value_counts()
#verificando que candidatos são esses com situação nula

cand_nulo = cand_repetidos[cand_repetidos.DS_SIT_TOT_TURNO=='#NULO#'].SQ_CANDIDATO.tolist()

cand_repetidos[eachIn(cand_repetidos.SQ_CANDIDATO, cand_nulo)]
#atribuindo '2º TURNO' para os candidatos com situação nula

candidatos.loc[7863,'DS_SIT_TOT_TURNO'] = '2º TURNO'

candidatos.loc[12133,'DS_SIT_TOT_TURNO'] = '2º TURNO'
candidatos = candidatos.drop(candidatos[candidatos.DS_SIT_TOT_TURNO=='2º TURNO'].index, axis='rows')
#mostrando o número de candidatos que aparecem duas vezes no dataset

repetidos = candidatos.SQ_CANDIDATO.value_counts()[candidatos.SQ_CANDIDATO.value_counts()==2]

if(execucao_direta):

    print(f"repetidos: {len(repetidos)}")

cand_repetidos = candidatos[eachIn(candidatos.SQ_CANDIDATO, repetidos)]

cand_repetidos.DS_SIT_TOT_TURNO.value_counts()
cand_repetidos.loc[:,['NM_CANDIDATO','SQ_CANDIDATO','DS_CARGO','SG_UF','NM_TIPO_ELEICAO','NR_TURNO','DS_SIT_TOT_TURNO']].sort_values('NM_CANDIDATO')
eliminar = candidatos[(candidatos.SQ_CANDIDATO==90000000608) & (candidatos.NR_TURNO==1)].index

candidatos = candidatos.drop(eliminar, axis="rows")

eliminar = candidatos[(candidatos.SQ_CANDIDATO==90000000637) & (candidatos.NR_TURNO==1)].index

candidatos = candidatos.drop(eliminar, axis="rows")
#mostrando o número de candidatos que aparecem duas vezes no dataset

repetidos = candidatos.SQ_CANDIDATO.value_counts()[candidatos.SQ_CANDIDATO.value_counts()==2]

if(execucao_direta):

    print(f"repetidos: {len(repetidos)}")

cand_repetidos = candidatos[eachIn(candidatos.SQ_CANDIDATO, repetidos)]

cand_repetidos.DS_SIT_TOT_TURNO.value_counts()
colunas_desnecessarias.append('NR_TURNO')
#candidatos['DS_SIT_TOT_TURNO'] = candidatos.DS_SIT_TOT_TURNO.replace('#NULO#','NÃO ELEITO')

#candidatos['DS_SIT_TOT_TURNO'] = candidatos.DS_SIT_TOT_TURNO.replace('SUPLENTE','NÃO ELEITO')

#candidatos['DS_SIT_TOT_TURNO'] = candidatos.DS_SIT_TOT_TURNO.replace('ELEITO POR MÉDIA','ELEITO')

#candidatos['DS_SIT_TOT_TURNO'] = candidatos.DS_SIT_TOT_TURNO.replace('ELEITO POR QP','ELEITO')

#candidatos.groupby('DS_SIT_TOT_TURNO').DS_SIT_TOT_TURNO.count()
candidatos.groupby('DS_SIT_TOT_TURNO').DS_SIT_TOT_TURNO.count()
class Transformer_ds_turno:

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        candidatos = X.copy()



        #Reduzindo todas as possíveis classificações só para ELEITO e NÃO ELEITO

        candidatos = candidatos.replace('#NULO#','NÃO ELEITO')

        candidatos = candidatos.replace('SUPLENTE','NÃO ELEITO')

        candidatos = candidatos.replace('ELEITO POR MÉDIA','ELEITO')

        candidatos = candidatos.replace('ELEITO POR QP','ELEITO')



        return candidatos
if(executar_pipeline):

    pipeline = Pipeline(steps=[('ds_turno',Transformer_ds_turno())])

    pipeline.fit(candidatos)

    candidatos = pipeline.transform(candidatos)
candidatos.groupby('DS_SIT_TOT_TURNO').DS_SIT_TOT_TURNO.count()
#essa transformação não será adicionada ao pipeline porque é aplicada somente sobre a coluna target

#sendo assim, rodo esse transformador manualmente logo antes do pipeline
candidatos['DS_SIT_TOT_TURNO'].value_counts()/len(candidatos)
eleitos = pd.read_csv('../input/dados-dos-candidatos-da-eleio-de-2014/resultado_eleicao.csv', sep=';', encoding='iso-8859-1')

eleitos.head()
#o número de eleitos segundo TSE

len(eleitos)
#O número de eleitos no meu dataset

cand_eleitos = candidatos[candidatos.DS_SIT_TOT_TURNO=='ELEITO']

len(cand_eleitos)
#mostrando os candidatos que considerei eleitos e não estão na lista do TSE

cand_eleitos[notIn(cand_eleitos.NM_CANDIDATO, eleitos['Nome do candidato'].tolist())].loc[:,['NM_CANDIDATO','NR_CPF_CANDIDATO','DS_CARGO','DS_SIT_TOT_TURNO']].sort_values('NM_CANDIDATO')
#mostrando os candidatos eleitos do TSE que não foram achados o meu dataset

eleitos[notIn(eleitos['Nome do candidato'], cand_eleitos.NM_CANDIDATO.tolist())]
#corrigindo o MARQUES BATISTA DE ABREU

candidatos.loc[15903,'DS_SIT_TOT_TURNO'] = 'NÃO ELEITO'
candidatos[candidatos.NM_CANDIDATO=='MÁRCIO JOSÉ MACHADO OLIVEIRA'].loc[:,['NM_CANDIDATO','NR_CPF_CANDIDATO','DS_CARGO','DS_SIT_TOT_TURNO']].sort_values('NM_CANDIDATO')
#corrigindo o MÁRCIO JOSÉ MACHADO OLIVEIRA

candidatos.loc[16296,'DS_SIT_TOT_TURNO'] = 'ELEITO'
candidatos.loc[candidatos.DS_SIT_TOT_TURNO=='ELEITO'].DS_SIT_TOT_TURNO.count()
candidatos = pd.concat([candidatos, candidatos.loc[candidatos.DS_SIT_TOT_TURNO=='ELEITO']], ignore_index=True)

candidatos = pd.concat([candidatos, candidatos.loc[candidatos.DS_SIT_TOT_TURNO=='ELEITO POR QP']], ignore_index=True)

candidatos = pd.concat([candidatos, candidatos.loc[candidatos.DS_SIT_TOT_TURNO=='ELEITO POR MÉDIA']], ignore_index=True)
candidatos.loc[candidatos.DS_SIT_TOT_TURNO=='ELEITO'].DS_SIT_TOT_TURNO.count()
len(candidatos)
nao_eleitos = candidatos.loc[candidatos.DS_SIT_TOT_TURNO=='NÃO ELEITO'].sample(3000, random_state=2) 
eleitos = candidatos.loc[eachIn(candidatos.DS_SIT_TOT_TURNO,['ELEITO','ELEITO POR QP','ELEITO POR MÉDIA'])]
candidatos = pd.concat([nao_eleitos, eleitos])
candidatos.shape
candidatos.columns
#candidatos = candidatos.drop(colunas_desnecessarias, axis="columns")
class Transformer_drop_columns(BaseEstimator):

    def __init__(self, columns):

        self.columns = columns

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        candidatos = X.copy()

        candidatos = candidatos.drop(self.columns, axis="columns")

        return candidatos
if(executar_pipeline):

    pipeline = Pipeline(steps=[('drop_columns',Transformer_drop_columns(colunas_desnecessarias))])

    pipeline.fit(candidatos)

    candidatos = pipeline.transform(candidatos)
candidatos.columns
#adicionando essa transformação no pipeline

pipe_steps.append(('drop_columns',Transformer_drop_columns(colunas_desnecessarias)))
import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.metrics import plot_confusion_matrix

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import  OneHotEncoder

from sklearn import set_config

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.base import TransformerMixin



import matplotlib.pyplot as plt



from IPython.core.display import display, HTML
#mostra erro se o pipeline já foi executado no pré-processamento

if(executar_pipeline):

    stop
#mostra o tamanho do dataset

candidatos.shape
#separa as variáveis independentes

features = candidatos.drop('DS_SIT_TOT_TURNO', axis="columns")

features.shape
#separa as variáveis dependentes

target = candidatos.loc[:,'DS_SIT_TOT_TURNO']

#essa transformação reduz o número de categorias nos resultados para ELEITO e NÃO ELEITO apenas

targetTransformer = Transformer_ds_turno()

targetTransformer.fit(target)

target = targetTransformer.transform(target)

target.shape
#mostra os passos de pré-processamento configurados no outro caderno

pipe_steps
preprocessamento = Pipeline(steps=pipe_steps)
onehotencoder = make_column_transformer((OneHotEncoder(handle_unknown='error', drop='first'), ['DS_CARGO','DS_COR_RACA','DS_DETALHE_SITUACAO_CAND','DS_ESTADO_CIVIL',

                                               'DS_GENERO','DS_GRAU_INSTRUCAO','DS_NACIONALIDADE','DS_OCUPACAO',

                                               'DS_SITUACAO_CANDIDATURA','SG_PARTIDO','SG_UE','SG_UF_NASCIMENTO',

                                               'ST_DECLARAR_BENS','ST_REELEICAO','TP_AGREMIACAO']), remainder='passthrough')
#separação dos dados de teste e treinamento

X_treino, X_teste, y_treino, y_teste = train_test_split(features, target, test_size=0.2, shuffle=True, random_state=123, stratify=target) 
class Redefine_Desconhecido(BaseEstimator, TransformerMixin):

    def __init__(self, mapa=None):

        self.mapa = mapa

    

    def fit(self, X, y=None):

        self.classes = {}

        for coluna in sorted(self.mapa.keys()):

            self.classes[coluna] = X.loc[:,coluna].unique()

        return self

        

    def transform(self, X, y=None):

        X_ = X.copy()   

        for coluna in sorted(self.mapa.keys()):

            X_.loc[notIn(X_.loc[:,coluna], self.classes[coluna]),coluna] = self.mapa[coluna]

        return X_
#função que realiza a criação de um Grid Search com os parâmetros configurados, roda o grid e apresenta uma avaliação preliminar

def avaliacao_grid_search(parameters):

    pipeline = Pipeline(steps=[('preprocessamento',preprocessamento),

                           ('redefine_desconhecido',Redefine_Desconhecido({'DS_DETALHE_SITUACAO_CAND':'INDEFERIDO', 'DS_CARGO':'DEPUTADO ESTADUAL', 'DS_COR_RACA':'BRANCA',

                         'DS_ESTADO_CIVIL':'CASADO(A)', 'DS_GENERO':'MASCULINO', 'DS_GRAU_INSTRUCAO':'SUPERIOR COMPLETO',

                         'DS_NACIONALIDADE':'BRASILEIRA NATA', 'DS_SITUACAO_CANDIDATURA':'APTO', 'SG_PARTIDO':'PT',

                         'SG_UE':'SP', 'SG_UF_NASCIMENTO':'SP', 'ST_DECLARAR_BENS':'N', 'ST_REELEICAO':'N'})),

                          ('onehotencoder', onehotencoder),

                          ('classificador', DecisionTreeClassifier())])

    

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=True, cv=3)

    

    grid_search.fit(X_treino, y_treino)

        

    display(HTML('<font size="4"><br/><bold>Melhores parâmetros</bold></font>'))

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))

        

    display(HTML('<font size="4"><br/><bold>Desempenho nos dados de treinamento</bold></font>'))

    media = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]

    desvio = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    print(f"\tCross-validation: {media} (+/- {desvio})")

        

    display(HTML('<font size="4"><br/><bold>Desempenho nos dados de teste</bold></font>'))

    print(f"\tAcurácia: {grid_search.score(X_teste, y_teste)}")

    print(f"\tF1_score: {metrics.f1_score(y_teste.tolist(), grid_search.predict(X_teste),pos_label='ELEITO')}")

    

    display(HTML('<font size="4"><br/><bold>Matriz de confusão</bold></font>'))

    plot_confusion_matrix(grid_search, X_teste, y_teste);

    

    return grid_search
parameters = {

    'classificador':[LogisticRegression(solver='lbfgs', multi_class='auto')],

    'classificador__max_iter': (2, 50, 100, 1000, 5000, 10000),

    'classificador__random_state': [123]

}
grid_search = avaliacao_grid_search(parameters)
parameters = {

    'classificador':[KNeighborsClassifier()],

    'classificador__n_neighbors': (5, 10, 20, 40, 80),

    'preprocessamento__ds_ocupacao__min_value': (1,2,3,4,5,10,20,100)

}
grid_search = avaliacao_grid_search(parameters)
parameters = {

    'preprocessamento__ds_ocupacao__min_value': [1,2,10,50,100,200,500,1000,2000,4000],

    'classificador':[DecisionTreeClassifier()],

    'classificador__min_samples_split': (5, 10, 20, 40, 80),

    'classificador__max_depth': (None, 3, 4, 5, 6),

    'classificador__random_state': [123]

}
grid_search = avaliacao_grid_search(parameters)
#mostrar a imagem do pipeline

set_config(display='diagram')

grid_search
#gera o nome das colunas de maneira mais legível considerando o One Hot Encoder

onehotencoder_treinado = grid_search.best_estimator_.named_steps['onehotencoder'].transformers_[0][1]

colunas = onehotencoder_treinado.get_feature_names(['CARGO','RACA','DET_SIT_CAND','ESTADO_CIVIL',

                                               'GENERO','INSTRUCAO','NACIONALIDADE','OCUPACAO',

                                               'SIT_CANDIDATURA','PARTIDO','UE','NASCIMENTO',

                                               'BENS_DECLARAR','REELEICAO','AGREMIACAO'])



colunas_originais = ['NR_DESPESA_MAX_CAMPANHA','NR_IDADE_DATA_POSSE', 'SEM_DESPESA_MAX_CAMPANHA']



colunas = colunas.tolist()

colunas.extend(colunas_originais)
#mostra os primeiros nós da árvore de decisão

arvore = grid_search.best_estimator_.named_steps['classificador']

plt.subplots(figsize=(20, 20))

plot_tree(arvore, max_depth=3, filled=True, fontsize=12, feature_names=colunas, class_names=["ELEITO","NÃO ELEITO"])

plt.show()
print("O caderno foi executado com êxito!") 