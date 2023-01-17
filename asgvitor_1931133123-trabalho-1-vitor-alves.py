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
# Carregando os dados

df = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')



df.shape, test.shape
## Separando as features:



x_train = df['Id']

x_test = test['Id']

y_train = df['Target'] 

## y_test = test['Target'] Não existe



# Juntando os dataframes

df_all = df.append(test)



df_all.shape
## Gráficos

import matplotlib.pyplot as plt

import seaborn as sns



## Modelos

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV

from sklearn.metrics import classification_report,multilabel_confusion_matrix

def grafico_pizza(labels,var,titulo,legenda):

    sizes = [df[var].value_counts()[0],df[var].value_counts()[1]]

    explode = (0, 0.1)  



    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

    ax1.axis('equal')  

    ax1.set_title(titulo)

    ax1.legend(title=legenda,

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))

    plt.show()
def grafico_barras(variaveis,eixoX,titulo):

    eixoY = []

    for v in variaveis: 

        eixoY.append(df[v].value_counts()[1])

    

    plt.figure(figsize=(20,5))

    sns.barplot(x = eixoX,y = eixoY).set_title(titulo)

    plt.show()
### Super lotação de quartos

labels = 'Não','Sim'

var = 'hacdor'

titulo = 'Superlotação de quartos'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
### Super lotação de espaços

labels = 'Não','Sim'

var = 'hacapo'

titulo = 'Superlotação de espaços'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
### Possui geladeira

labels = 'Não','Sim'

var = 'refrig'

titulo = 'Possui geladeira?'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
### Possui tablet

labels = 'Não','Sim'

var = 'v18q'

titulo = 'Possui tablet?'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
## Material predominante na parte de fora da casa 

variaveis = 'paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc','paredfibras','paredother'

eixoX = ['Bloco ou tijolo','Encaixe','Pré moldado ou Cimento','Resíduo','Madeira','Zinco','Fibras Naturais','Outro']

titulo = 'Material predominante na parte de fora da casa'

grafico_barras(variaveis,eixoX,titulo)
## Material predominante no piso

variaveis = 'pisomoscer','pisocemento','pisoother','pisonatur','pisonotiene','pisomadera'

eixoX = ['Mosaico, Cerâmica ou Terrazo','Cimento','Outro','Natural','Não há piso','Madeira']

titulo = 'Material predominante no piso'

grafico_barras(variaveis,eixoX,titulo)
## Material predominante no teto

variaveis = 'techozinc','techoentrepiso','techocane','techootro'

eixoX = ['Folha de metal ou zinco','Fibro Cimentou ou Mezanino','Fibras naturais','Outro']

titulo = 'Material predominante no teto'

grafico_barras(variaveis,eixoX,titulo)
### Possui Teto

labels = 'Não','Sim'

var = 'cielorazo'

titulo = 'Possui teto?'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
## Abastecimento de água

variaveis = 'abastaguadentro','abastaguafuera','abastaguano'

eixoX = ['Interior da casa','Fora da casa','Não há abastecimento']

titulo = 'Abastecimento de água'

grafico_barras(variaveis,eixoX,titulo)
## Abastecimento de eletricidade

variaveis = 'public','planpri','noelec','coopele'

eixoX = ['CNFL, ICE, ESPH / JASEC','Privada','Sem eletricidade','Cooperativa']

titulo = 'Abastecimento de Eletricidade'

grafico_barras(variaveis,eixoX,titulo)
## Banheiros

variaveis = 'sanitario1','sanitario2','sanitario3','sanitario5','sanitario6'

eixoX = ['Sem banheiro','Banheiro com esgoto','Banheiro com fossa','Banheiro conectado a buraco','Banheiro conectado a outro sistema']

titulo = 'Banheiros'

grafico_barras(variaveis,eixoX,titulo)
# Principal fonte de energia para cozinhar

variaveis = 'energcocinar1','energcocinar2','energcocinar3','energcocinar4'

eixoX = ['Sem cozinha','Elétrica','Gás','Carvão']

titulo = 'Principal fonte de energia para cozinhar'

grafico_barras(variaveis,eixoX,titulo)
# Descarte de lixo

### elimbasu5 sempre 0!

variaveis = 'elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu6'

eixoX = ['Caminhão Tanque','Botânica ou Enterrada','Queima','Terreno Baldio','Outros']

titulo = 'Descarte de lixo'

grafico_barras(variaveis,eixoX,titulo)
# Situação das paredes

variaveis = 'epared1','epared2','epared3'

eixoX = ['Parede ruim','Parede regular','Parede boa']

titulo = 'Situação das paredes'

grafico_barras(variaveis,eixoX,titulo)
# Situação do teto

variaveis = 'etecho1','etecho2','etecho3'

eixoX = ['Teto ruim','Teto regular','Teto bom']

titulo = 'Situacao do teto'

grafico_barras(variaveis,eixoX,titulo)
## Situacao do chão

variaveis = 'eviv1','eviv2','eviv3'

eixoX = ['Chão ruim','Chão regular','Chão bom']

titulo = 'Situação do chão'

grafico_barras(variaveis,eixoX,titulo)
### Pessoa incacitada

labels = 'Não','Sim'

var = 'dis'

titulo = 'Pessoa incapacitada?'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
### Distribuição do sexo

labels = 'Não','Sim'

var = 'male'

titulo = 'Distribuição do sexo'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
titulo = 'Estado civil'

variaveis = 'estadocivil1','estadocivil2','estadocivil3','estadocivil4','estadocivil5','estadocivil6','estadocivil7'

eixoX = ['< 10 anos ','Free','Casado','Divorciado','Separado','viúvo','Solteiro']

grafico_barras(variaveis,eixoX,titulo)
titulo = 'Parentesco'

variaveis = 'parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12'

eixoX = ['Chefe de família','Cônjugue','Filho','Divorciado','Genro/Nora','Neto','Pai','Sogro','Irmão','Cunhada','Outro Familiar','Outro Não Familiar']

grafico_barras(variaveis,eixoX,titulo)
titulo = 'Nível de educação'

variaveis = 'instlevel1','instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9'

eixoX = ['Sem nível de educação','Primário Incompleto','Primário Completo','Secundário Incompleto','Secundário Completo','Técnico Incompleto','Técnico Completo','Graduação','Ensino Superior']

grafico_barras(variaveis,eixoX,titulo)
titulo = 'Tipo de Moradia'

variaveis = 'tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5'

eixoX = ['Casa própria e quitada','Própria e parcelada','Alugada','Precária','Outro (Atribuído / Empresatado)']

grafico_barras(variaveis,eixoX,titulo)
### Possui Computador ?

labels = 'Não','Sim'

var = 'computer'

titulo = 'Possui Computador ?'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
### Possui Televisão ?

labels = 'Não','Sim'

var = 'television'

titulo = 'Possui Televisão ?'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
### Possui telefone Celular ?

labels = 'Não','Sim'

var = 'mobilephone'

titulo = 'Possui telefone Celular ?'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
titulo = 'Região'

variaveis = 'lugar1','lugar2','lugar3','lugar4','lugar5','lugar6'

eixoX = ['Central','Chorotega','Pacífico central','Brunca','Huetar Atlântica','Huetar Norte']

grafico_barras(variaveis,eixoX,titulo)
### Zonas

labels = 'Rural','Urbana'

var = 'area1'

titulo = 'Zona Urbana / Rural'

legenda = 'Legenda'

grafico_pizza(labels,var,titulo,legenda)
## Separando as variáveis

varNumericas = ['v2a1','rooms','v18q','v18q1','r4h1','r4h2','r4h3','r4m1','r4m2','r4m3','r4t1','r4t2','r4t3','tamhog','tamviv','escolari','rez_esc','hhsize','hogar_nin','hogar_adul','hogar_mayor','hogar_total','dependency','edjefe','edjefa','meaneduc','bedrooms','overcrowding','qmobilephone','age','SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','agesq']
## Correlações

plt.figure(figsize=(20,20));

sns.heatmap(df[varNumericas].corr(), square=True ,annot=True, linewidths=1,vmin=-1,vmax=1,cmap='RdYlGn')
naoUsar = ['idhogar','Id','Target'] # ID

naoUsarNumericas = ['tamhog','hogar_total','agesq','hhsize'] ## Correlação 1
## Tirando as variáveis

varNumericas = np.setdiff1d(varNumericas,naoUsarNumericas)

## Correlações

plt.figure(figsize=(20,20));

sns.heatmap(df[varNumericas].corr(), square=True ,annot=True, linewidths=1,vmin=-1,vmax=1,cmap='RdYlGn')
df[varNumericas].describe().transpose()
## Análise da Variável Alvo

df['Target'].value_counts().sort_values()
eixoX = ['Pobreza Extrema','Pobreza Moderada','Famílias Vulneráveis','Famílias Não Vulneráveis']

plt.figure(figsize=(20,5))

sns.barplot(x = eixoX,y = df['Target'].value_counts().sort_values()).set_title(titulo)

plt.show()
# Quais colunas do dataframe são do tipo object

df_all.select_dtypes('object').head()
# Analisando os dados da coluna edjefa

df_all['edjefa'].value_counts()
# Analisando os dados da coluna edjefe

df_all['edjefe'].value_counts()
## Analisando a coluna dependency

df_all['dependency'].value_counts()
# Transformar 'yes' em 1 e 'no' em 0



mapeamento = {'yes': 1, 'no': 0}

df_all['edjefa'] = df_all['edjefa'].replace(mapeamento).astype(int)

df_all['edjefe'] = df_all['edjefe'].replace(mapeamento).astype(int)

df_all['dependency'] = df_all['dependency'].replace(mapeamento).astype(float)
# Quais colunas do dataframe são do tipo object

df_all.select_dtypes('object').head()
# Visualizando do comando info

df_all.info()
# Verificando os valores nulos

df_all.isnull().sum()
# Prenchendo com -1 os valores nulos de v2a1

df_all['v2a1'].fillna(-1, inplace=True)

# Prenchendo com 0 os valores nulos de v18q1

df_all['v18q1'].fillna(0, inplace=True)

# Prenchendo com -1 os valores nulos de SQBmeaned, meaneduc e rez_esc

df_all['SQBmeaned'].fillna(-1, inplace=True)

df_all['meaneduc'].fillna(-1, inplace=True)

df_all['rez_esc'].fillna(-1, inplace=True)
# Verificando os valores nulos novamente

df_all.isnull().sum()
# Separando as colunas para treinamento

feats = [c for c in df_all.columns if c not in ['Id', 'idhogar', 'Target']]
# Separar os dataframes

train, test = df_all[~df_all['Target'].isnull()], df_all[df_all['Target'].isnull()]



train.shape, test.shape
# Instanciando o random forest classifier

rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)
# Treinando o modelo

rf.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões

test['Target'].value_counts(normalize=True)
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission_1.csv', index=False)
fig=plt.figure(figsize=(15, 20))



# Avaliando a importancia de cada coluna (cada variável de entrada)

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
varNaoUtilizadas = ['Id', 'idhogar', 'Target'] ## Ids e alvo

varNaoUtilizadasCat = ['female','area2'] ## Duplicadas

varNaoUtilizadasNum = ['tamhog','hogar_total','hhsize'] ## Correlação = 1

varNaoUtilizadasSQ = ['SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','agesq']

varNaoUtilizadas = varNaoUtilizadas + varNaoUtilizadasCat + varNaoUtilizadasNum + varNaoUtilizadasSQ

varNaoUtilizadas
# Separando as colunas para treinamento

feats = [c for c in df_all.columns if c not in varNaoUtilizadas]
# Treinando o modelo

rf.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões

test['Target'].value_counts(normalize=True)
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)

# 0.36832 contra 0.36781 da primeira
## Usar || Nao Usar 

## 'techozinc' || 'techoentrepiso','techocane','techootro'

## 'abastaguadentro' || 'abastaguafuera','abastaguano'

## 'public' || 'planpri','noelec','coopele'

## 'sanitario3' || 'sanitario1','sanitario2', 'sanitario5','sanitario6'

## 'energcocinar2' || 'energcocinar1', 'energcocinar3','energcocinar4'

## 'elimbasu1'  ||  'elimbasu2','elimbasu3','elimbasu4','elimbasu6'

## 'tipovivi1'  || 'tipovivi2','tipovivi3','tipovivi4','tipovivi5'

varNaoUtilizadas = ['Id', 'idhogar', 'Target'] ## Ids e alvo

varNaoUtilizadasCat = ['female','area2'] ## Duplicadas

varNaoUtilizadasNum = ['tamhog','hogar_total','hhsize'] ## Correlação = 1

varNaoUtilizadasSQ = ['SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','agesq']

varPoucosRegistros = ['techoentrepiso','techocane','techootro','abastaguafuera','abastaguano','sanitario1','sanitario2', 'sanitario5','sanitario6','energcocinar1', 'energcocinar3','energcocinar4','elimbasu2','elimbasu3','elimbasu4','elimbasu6','tipovivi2','tipovivi3','tipovivi4','tipovivi5']

varNaoUtilizadas = varNaoUtilizadas + varNaoUtilizadasCat + varNaoUtilizadasNum + varNaoUtilizadasSQ + varPoucosRegistros

varNaoUtilizadas
# Separando as colunas para treinamento

feats = [c for c in df_all.columns if c not in varNaoUtilizadas]
# Treinando o modelo

rf.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões

test['Target'].value_counts(normalize=True)
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)

# 0.35910
rf = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')
# Separando as colunas para treinamento

feats = [c for c in df_all.columns if c not in varNaoUtilizadas]
# Treinando o modelo

rf.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões

test['Target'].value_counts(normalize=True)
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)

# 0.42693
rf.get_params().keys()
param_grid = {'max_depth': [None,5,10],

             'max_leaf_nodes': [None,2,6],

             'min_impurity_decrease' : [1,1e-3],

             'n_jobs': [-1],

             'min_samples_leaf': [2,4],

             'n_estimators': [100,300,700],

             'class_weight' : [None,'balanced']}



grid = GridSearchCV(rf,param_grid=param_grid,cv=4,scoring='f1_macro')
grid.fit(train[feats], train['Target'])
grid_df = pd.DataFrame(grid.cv_results_)

grid_df
## Modelo com melhores parâmetros

grid_df.sort_values('rank_test_score',ascending=True).iloc[0,:]
## acessando os melhores parametros

grid.best_params_
rf = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=100,max_leaf_nodes=None,

                            min_impurity_decrease=0.001, min_samples_leaf=4,

                            verbose=0, class_weight='balanced')
# Separando as colunas para treinamento

feats = [c for c in df_all.columns if c not in varNaoUtilizadas]
# Treinando o modelo

rf.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões

test['Target'].value_counts(normalize=True)
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)

# 0.42100
fig=plt.figure(figsize=(15, 20))



# Avaliando a importancia de cada coluna (cada variável de entrada)

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()