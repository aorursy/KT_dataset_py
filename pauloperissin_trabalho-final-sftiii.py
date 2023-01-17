import sqlite3 # Declara que vamos utilizar o SQlite para ler dados
import pandas as pd
conexao = sqlite3.connect('../input/amostra05pcsqlite/amostra05pc.sqlite')

# Carrega os dados do CNPJ
df_CNPJ = pd.read_sql_query("SELECT * FROM cnpj_dados_cadastrais_pj", conexao)
# Verificamos o tamanho da nossa base de dados
print(df_CNPJ.shape)
df_desc_cnae = pd.read_sql_query("SELECT cod_secao, nm_secao, cod_cnae FROM tab_cnae", conexao)
df_desc_cnae.head()
df_uf_regiao = pd.read_csv('../input/uf-regiao/uf_regiao.csv')
df_uf_regiao.head()
def tabela_categoricas(df,col,name_col):
    df_1 = pd.DataFrame(df[col].value_counts()).reset_index()
    df_1.columns = [name_col,'Qtd empresas']
    df_1['%Total'] = (df_1['Qtd empresas']/len(df))*100
    return df_1
def distribuicaoNumericas(df,col,nameCol):
    med = pd.DataFrame(columns=[nameCol], index=['Media', 'Mediana', 'Minimo','Maximo'])
    med.loc['Media'][nameCol] = float(df[col].mean())
    med.loc['Mediana'][nameCol] = float(df[col].median())
    med.loc['Minimo'][nameCol] = float(df[col].min())
    med.loc['Maximo'][nameCol] = float(df[col].max())
    return med
df_cnpjs = df_CNPJ.copy()

df_cnpjs = df_cnpjs[(df_cnpjs["situacao_cadastral"].isin(["08","02"]))]
print(df_cnpjs.shape)
df_cnpjs.head()
# Descrição do Porte da Empresa
df_cnpjs['porte_empresa_descr'] = df_cnpjs['porte_empresa'].replace(['00','01','03','05'],['NAO INFORMADO','MICRO EMPRESA','EMPRESA DE PEQUENO PORTE','DEMAIS'])
df_cnpjs['situacao_atividade']= df_cnpjs['situacao_cadastral'].replace(['08','02'],['INATIVA','ATIVA'])
df_cnpjs.head()
df_cnpjs['cnae_fiscal']

# Join para incluir descrição do CNAE
df_cnpjs = pd.merge(df_cnpjs, df_desc_cnae,how='left', left_on=['cnae_fiscal'], right_on=['cod_cnae'])
df_cnpjs = pd.merge(df_cnpjs, df_uf_regiao,how='left', on=['uf'])
df_cnpjs.head()
tabela_categoricas(df_cnpjs,'situacao_atividade', 'Atividade')
df_cnpjs['data_inicio_atividade'] = pd.to_datetime(df_cnpjs['data_inicio_atividade'])
df_cnpjs['data_situacao_cadastral'] = pd.to_datetime(df_cnpjs['data_situacao_cadastral'])

#print(df_cnpjs.dtypes)
# Seleciono empresas que fecharam em 2020, independentemente de quando abriram 
df_cnpjs_filtrado = df_cnpjs[(df_cnpjs['data_situacao_cadastral'] >= "2020-01-01") & 
                            (df_cnpjs['situacao_cadastral'] < "08")]
df_cnpjs_filtrado.shape
# Verifica a distribuição por categorias de atividade na base total
tabela_categoricas(df_cnpjs,'nm_secao','SEÇÃO')
# Verifica a distribuição do tamanho das empresas na base total
tabela_categoricas(df_cnpjs,'porte_empresa_descr','Porte')
# Mostra o porte das empresas que fecharam esse ano (2020)
tabela_categoricas(df_cnpjs_filtrado,'porte_empresa_descr','Porte')
# Mostra o dado tabular das áreas de atuação das empresas que fecharam esse ano
tabela_categoricas(df_cnpjs_filtrado,'nm_secao','SEÇÃO')
df_cnpjs['capital_social_empresa'].hist(grid = False)
distribuicaoNumericas(df_cnpjs,'capital_social_empresa','Capital Social - Empresas')
df_cnpjs[df_cnpjs['capital_social_empresa']==0].shape[0]
df_cnpjs[df_cnpjs['capital_social_empresa'] > 10000000000]
# Separo as empresas ativas das inativas
cnpjAtivos = df_cnpjs[df_cnpjs['situacao_cadastral'] == "02"].copy()
cnpjInativo = df_cnpjs[df_cnpjs['situacao_cadastral'] == "08"].copy()
cnpjAtivos['idade'] = pd.to_datetime("2020-07-04") - cnpjAtivos['data_inicio_atividade']
print(cnpjAtivos.head())
cnpjAtivos['idade'] = pd.to_numeric(cnpjAtivos['idade'].dt.days, downcast = 'integer') / 365
print(cnpjAtivos.head())
distribuicaoNumericas(cnpjAtivos, 'idade' , 'Idade das Empresas Ativas')
cnpjInativo['idade'] = cnpjInativo['data_situacao_cadastral'] - cnpjInativo['data_inicio_atividade']
cnpjInativo['idade'] = pd.to_numeric(cnpjInativo['idade'].dt.days, downcast = 'integer') / 365
distribuicaoNumericas(cnpjInativo, 'idade' , 'Idade das Empresas Inativas')
cnpjAtivos['idade'].hist(grid = False, bins = 10)
cnpjInativo['idade'].hist(grid = False, bins = 10)
cnpjAtivos['anoAbertura'] = cnpjAtivos['data_inicio_atividade'].dt.year
cnpjInativo['anoAbertura'] = cnpjInativo['data_inicio_atividade'].dt.year
cnpjInativo['anoFechamento'] = cnpjInativo['data_situacao_cadastral'].dt.year
df_cnpjs['anoAbertura'] = df_cnpjs['data_inicio_atividade'].dt.year
abert_anos = pd.DataFrame(df_cnpjs['anoAbertura'].value_counts()).reset_index()
abert_anos = abert_anos[abert_anos['index'] > 1984]
abert_anos.columns = ['Ano Abertura' , 'Qtd. Empresas']
abert_anos['Ano Abertura'] = abert_anos['Ano Abertura'].apply(str)
abert_anos = abert_anos.sort_values(by='Ano Abertura')
ax = abert_anos.plot(kind = "bar", x = 'Ano Abertura',
                     title = " Quantidade de Empresas Abertas",figsize = (10,4))
ax.set_xlabel("Ano Abertura")
df_cnpjs_ult_ano = df_cnpjs[(df_cnpjs['anoAbertura'] > 2018) & (df_cnpjs['uf'] == 'DF')]
tabela_categoricas(df_cnpjs_ult_ano, 'nm_secao', 'SEÇÃO')
fech_anos = pd.DataFrame(cnpjInativo['anoFechamento'].value_counts()).reset_index()
fech_anos = fech_anos[fech_anos['index'] > 1984]
fech_anos.columns = ['Ano Fechamento','Qtd Empresas']
fech_anos['Ano Fechamento'] = fech_anos['Ano Fechamento'].apply(str)
fech_anos = fech_anos.sort_values(by='Ano Fechamento')
ax = fech_anos.plot(kind = "bar", x = 'Ano Fechamento',
                    title = " Quantidade de Empresas Fechadas",figsize = (10,4))
ax.set_xlabel("Ano Fechamento")
df_cnpjs_ult_ano = cnpjInativo[(cnpjInativo['anoFechamento'] > 2019)]
tabela_categoricas(df_cnpjs_ult_ano, 'nm_secao', 'SEÇÃO')
uf_fechamento = pd.DataFrame(cnpjInativo['uf'].value_counts()).reset_index()
uf_fechamento.columns = ['Estados', 'Quantidade de Empresas Fechadas']
uf_fechamento = uf_fechamento.sort_values(by='Estados')
ax = uf_fechamento.plot(kind = "bar", x = 'Estados',
                    title = "Quantidade de Empresas Fechadas por UF",figsize = (10,4))
ax.set_xlabel("UF")
uf_fechamento = pd.DataFrame(df_cnpjs_ult_ano['uf'].value_counts()).reset_index()
uf_fechamento.columns = ['Estados', 'Quantidade de Empresas Fechadas']
uf_fechamento = uf_fechamento.sort_values(by='Estados')
ax = uf_fechamento.plot(kind = "bar", x = 'Estados',
                    title = "Quantidade de Empresas Fechadas por UF",figsize = (10,4))
ax.set_xlabel("UF")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


warnings.filterwarnings('ignore')
df_cnpjs.shape
df_cnpjs.head()
df_cnpjs.dtypes
df_cnpjs.isnull().sum().sort_values()
#Criação da variável mes para observar a quantidade de empresas no mes
df_cnpjs['data_situacao_cadastral'] = pd.to_datetime(df_cnpjs['data_situacao_cadastral'])
df_cnpjs['mes'] = df_cnpjs['data_situacao_cadastral'].dt.month
print(df_cnpjs['mes'])
df_cnpjs['data_situacao_cadastral'] = pd.to_datetime(df_cnpjs['data_situacao_cadastral'])
df_cnpjs['dia'] = df_cnpjs['data_situacao_cadastral'].dt.day
print(df_cnpjs['dia'])
df_cnpjs['mes'].hist(grid = True)
#Idade de todas as empresas
df_cnpjs['idade'] = df_cnpjs['data_situacao_cadastral'] - df_cnpjs['data_inicio_atividade']
df_cnpjs['idade'] = pd.to_numeric(df_cnpjs['idade'].dt.days, downcast = 'integer') / 365
distribuicaoNumericas(cnpjInativo, 'idade' , 'Idade das Empresas Inativas')
# Seleciono empresas que fecharam em 2020, independentemente de quando abriram 
fechada = df_cnpjs[(df_cnpjs['data_situacao_cadastral'] >= "2020-01-01") & 
                            (df_cnpjs['situacao_cadastral'] == "08")]
fechada.shape
fechada.head()
fechamento = pd.DataFrame(fechada['uf'].value_counts()).reset_index()
fechamento.columns = ['Estados', 'Quantidade de Empresas Fechadas']
fechamento = fechamento.sort_values(by='Estados')
ax = fechamento.plot(kind = "bar", x = 'Estados',
                    title = "Quantidade de Empresas Fechadas por UF na Pandemia",figsize = (10,4))
ax.set_xlabel("UF")
fechamentos = pd.DataFrame(fechada['nm_secao'].value_counts()).reset_index()
fechamentos.columns = ['Segmento', 'Quantidade de Empresas Fechadas']
fechamentos = fechamentos.sort_values(by='Segmento')
ax = fechamentos.plot(kind = "bar", x = 'Segmento',
                    title = "Quantidade de Empresas Fechadas por Segmento na Pandemia",figsize = (10,4))
ax.set_xlabel("Segmento")
#Mapeamento para alterar o porte da empresa
mapeamento1 = {'00' : 0,'01' : 1, '03' : 3,'05' : 5, '' : 0}

#Criação de nova coluna como número para porte da empresa
df_cnpjs['porte_empresan'] = df_cnpjs['porte_empresa']
df_cnpjs['porte_empresan'] = df_cnpjs['porte_empresan'].replace(mapeamento1).astype(int)
#Alteração do tipo da coluna situacao cadastral para numero para poder ser utilizadas no modelo
df_cnpjs["situacao_cadastral"] = df_cnpjs["situacao_cadastral"].astype(int)

#Alteração do tipo da coluna para numero para poder ser utilizadas no modelo
df_cnpjs["cnae_fiscal"] = df_cnpjs["cnae_fiscal"].astype(int)
df_cnpjs["idade"] = df_cnpjs["idade"].astype(int)
df_cnpjs.dtypes
#Preparando o modelo de regressão linear para verificar a previsão do porte da empresa conforme a situação
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics  import mean_squared_error,r2_score
X = df_cnpjs[['cnae_fiscal','porte_empresan','idade','capital_social_empresa']]
y = df_cnpjs['situacao_cadastral']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=25, random_state=42,shuffle=True )
(X_test, y_test)
(X_train, y_train)
from sklearn.linear_model import LinearRegression
rl = LinearRegression()
%time rl.fit(X_train,y_train)
%time pred_rl = rl.predict(X_test)
rl.coef_
rl.score(X_train, y_train)
# Join para incluir descrição do CNAE
rr = X_train
#Alteração do tipo da coluna situacao cadastral para numero para poder ser utilizadas no modelo
rr["cnae_fiscal"] = rr["cnae_fiscal"].astype(str)
rrcnae = pd.merge(rr, df_desc_cnae,how='left', left_on=['cnae_fiscal'], right_on=['cod_cnae'])
rrcnae.head()
# Verifica a distribuição por categorias de atividade na base total
tabela_categoricas(rrcnae,'nm_secao','idade' )
y_train.head()
X_train.head()
# Criando o arquivo para exportação
y_train.to_csv('ArquivoSituacao.csv', index=False)
# Criando o arquivo para exportação
X_train.to_csv('Arquivo.csv', index=False)
## Correlation
import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = X_train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(X_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
from sklearn.neighbors import KNeighborsRegressor
kNR = KNeighborsRegressor(n_neighbors=3)
%time kNR.fit(X_train,y_train)
%time pred_kNR = kNR.predict(X_test)
kNR.score(X_train, y_train)
kNR.score(X_test, y_test)
df = df_cnpjs[['cnae_fiscal','porte_empresan','idade','capital_social_empresa','situacao_cadastral']]
df = pd.read_csv('../input/datarand1/df1.csv')
# Separando as variáveis para treinamento
# Separando as colunas para treinamento
feats = [c for c in df.columns if c  in ['cnae_fiscal','porte_empresan','idade','capital_social_empresa',]]
# Separar os dataframes
train, test = df[~df['situacao_cadastral'].isnull()], df[df['situacao_cadastral'].isnull()]

train.shape, test.shape
X1 = df_cnpjs[['cnae_fiscal','porte_empresan','idade','capital_social_empresa','situacao_cadastral']]

y1 = df_cnpjs[['cnae_fiscal','porte_empresan','idade','capital_social_empresa','situacao_cadastral']]
y1['situacao_cadastral'] = ''
y1.head()
train.shape, test.shape
# Instanciando o random forest classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)
# Treinando o modelo
rf.fit(train[feats], train['situacao_cadastral'])
# Treinando o modelo
rf.fit(test[feats], train['situacao_cadastral'])
# Vamos verificar as previsões
train['situacao_cadastral'].value_counts(normalize=True)
# Vamos verificar as previsões
test['situacao_cadastral'].value_counts(normalize=True)
# Vamos verificar as previsões
train['cnae_fiscal'].value_counts(normalize=True)
train['cnae_fiscal'].value_counts().sort_values()
#Avaliando a importancia de cada coluna no modelo
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(15,10))
    
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()

# Trabalhando com AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
abc = AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=42)
abc.fit([feats], train['situacao_cadastral'])
accuracy_score(test['situacao_cadastral'], abc.predict(train[feats]))
# Join para incluir descrição do CNAE
rt = train
#Alteração do tipo da coluna situacao cadastral para numero para poder ser utilizadas no modelo
rt["cnae_fiscal"] = rt["cnae_fiscal"].astype(str)
rcnae = pd.merge(rt, df_desc_cnae,how='left', left_on=['cnae_fiscal'], right_on=['cod_cnae'])

rcnae.head()
cnpjAtivos['nm_secao'].hist(grid = False)
