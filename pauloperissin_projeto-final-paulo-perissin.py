# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Set a few plotting defaults

%matplotlib inline

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['patch.edgecolor'] = 'k'
def plotar(variaveis,eixoX,titulo):

    eixoY = []

    for v in variaveis: 

        eixoY.append(df[v].value_counts()[1])

    

    plt.figure(figsize=(20,5))

    sns.barplot(x = eixoX,y = eixoY).set_title(titulo)

    plt.show()
#Carregar os dados dos datasets

df = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')



df.shape, test.shape
# Chefes de família

heads = df.loc[df['parentesco1'] == 1].copy()







# Variáveis para treinamento

train_campos = df.loc[(df['Target'].notnull()) & (df['parentesco1'] == 1), ['Target', 'idhogar']]



# Quantidade de chefes conforme a calissificação

l_counts = train_campos['Target'].value_counts().sort_index()



l_counts

Parentes = 'Parentesco'

variaveis = 'parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12'

eixoX = ['Chefe de família','Cônjugue','Filho','Divorciado','Genro/Nora','Neto','Pai','Sogro','Irmão','Cunhada','Outro Familiar','Outro Não Familiar']

plotar(variaveis,eixoX,Parentes)
# Verificação de famílias onde os indivíduos no mesmo domicílio têm um nível de pobreza diferente na base de treino

# Agrupa as famílias para verificar os valores únicos

all_equal = df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Famílias onde as metas não são todas iguais

not_equal = all_equal[all_equal != True]

print('Encontramos {} Individuos na mesma familia no mesmo domícilio que possuem um nível de probreza diferente e precisamos corrigir.'.format(len(not_equal)))
# Iteração com cada familia

for household in not_equal.index:

    # Localizar a classificação correta do chefe para cada familia

    true_target = int(df[(df['idhogar'] == household) & (df['parentesco1'] == 1.0)]['Target'])

    

    # Definindo a target correta para todos os membros da família

    df.loc[df['idhogar'] == household, 'Target'] = true_target

    

    

# Agrupando a famíliapara decobrir o número de valores únicos

all_equal = df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Famílias onde as metas não são todas iguais

not_equal = all_equal[all_equal != True]

print('Encontramos {} familias no mesmo domícilio que possuem um nível de probreza diferente.'.format(len(not_equal)))
households_leader = df.groupby('idhogar')['parentesco1'].sum()



# verificação de familias sem chefe

households_no_head = df.loc[df['idhogar'].isin(households_leader[households_leader == 0].index), :]



print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))







# Verificação das famílias sem chefe e com classificação diferente

households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

print('{} Households with no head have different labels.'.format(sum(households_no_head_equal == False)))
df_all = df.append(test)



df_all.shape
df_all.drop('area2', axis = 1, inplace = True)
# Criando a variável com as caracteristicas das paredes da casa

df_all['walls'] = np.argmax(np.array(df_all[['epared1', 'epared2', 'epared3']]),

                           axis = 1)

df_all = df_all.drop(columns = ['epared1', 'epared2', 'epared3'])

df_all['walls'].value_counts()
# Roof ordinal variable

#df_all['roof'] = np.argmax(np.array(df_all[['etecho1', 'etecho2', 'etecho3']]),

#                           axis = 1)

#df_all = df_all.drop(columns = ['etecho1', 'etecho2', 'etecho3'])
#df_all['roof'].value_counts()
# Floor ordinal variable

#df_all['floor'] = np.argmax(np.array(df_all[['eviv1', 'eviv2', 'eviv3']]),

 #                          axis = 1)

#df_all = df_all.drop(columns = ['eviv1', 'eviv2', 'eviv3'])
#df_all['floor'].value_counts()
# Create new feature

#df_all['walls+roof+floor'] = df_all['walls'] + df_all['roof'] + df_all['floor']
#df_all['walls+roof+floor'].value_counts()
# Verificando os valores nulos

df_all.isnull().sum().sort_values()
df_all.select_dtypes('object').head()
# Analisando os dados da coluna edjefa

df_all['edjefa'].value_counts()
# Analisando os dados da coluna edjefe

df_all['edjefe'].value_counts()
mapeamento = {'yes': 1, 'no': 0}



df_all['edjefa'] = df_all['edjefa'].replace(mapeamento).astype(int)

df_all['edjefe'] = df_all['edjefe'].replace(mapeamento).astype(int)
# Verificando a sobreposição dos dados para o masculino

df_all['edjefe'].value_counts()
# Verificando a sobreposição dos dados para o Feminino

df_all['edjefa'].value_counts()
# COntinuação da verificação das variaveis que são do tipo Object

df_all.select_dtypes('object').head()
# Verificando a coluna dependence

df_all['dependency'].value_counts().sort_values()
# Sobrepondo os dados da coluna dependency conforme o mapeamento e tranformando em tipo float

df_all['dependency'] = df_all['dependency'].replace(mapeamento).astype(float)
df_all['dependency'].value_counts()
df_all['dependency'].isnull().sum()
# COntinuação da verificação das variaveis que são do tipo Object

df_all.select_dtypes('object').head()
# verificando as informações do data set

df_all.info()
# Verificando os valores nulos

df_all.isnull().sum().sort_values()
df_all['Target'].value_counts()
# Verificando os valores de aluguel (v2a1) para os chefes de familia (parentesco = 1)

df_all[df_all['parentesco1'] == 1]['v2a1'].isnull().sum()
# Verificando os dados de  v2a1 - pagamento do valor de aluguel

df_all['v2a1'].value_counts()
df_all['v2a1'].isnull().sum()
# Verificando os dados de v18q

df_all['v18q'].value_counts()
df_all['v18q'].isnull().sum()
# Verificando os dados de v18q1 

df_all['v18q1'].value_counts()
df_all['v18q1'].isnull().sum()
# Prenchendo com -1 os valores nulos de v2a1

df_all['v2a1'].fillna(-1, inplace=True)
# Prenchendo com 0 os valores nulos de v18q1

df_all['v18q1'].fillna(0, inplace=True)
# Prenchendo com -1 os valores nulos de SQBmeaned, meaneduc e rez_esc

df_all['SQBmeaned'].fillna(-1, inplace=True)

df_all['meaneduc'].fillna(-1, inplace=True)

df_all['rez_esc'].fillna(-1, inplace=True)
df_all.isnull().sum().sort_values()
# Feature Engineering



# Vamos criar novas colunas para valores percapita

df_all['phone-pc'] = df_all['qmobilephone'] / df_all['tamviv']

df_all['tablets-pc'] = df_all['v18q1'] / df_all['tamviv']

df_all['rooms-pc'] = df_all['rooms'] / df_all['tamviv']

df_all['rent-pc'] = df_all['v2a1'] / df_all['tamviv']
# Separando as colunas para treinamento

feats = [c for c in df_all.columns if c not in ['Id', 'idhogar', 'Target']]



# Separar os dataframes

train, test = df_all[~df_all['Target'].isnull()], df_all[df_all['Target'].isnull()]



train.shape, test.shape
# Instanciando o random forest classifier



from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)
# Treinando o modelo

rf.fit(train[feats], train['Target'])

df_all['Target'].value_counts().sort_values()
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf.predict(test[feats]).astype(int)
# Vamos verificar as previsões

test['Target'].value_counts(normalize=True)
# Trabalhando com AdaBoost

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

abc = AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=42)

abc.fit(train[feats], train['Target'])

accuracy_score(test['Target'], abc.predict(test[feats]))
# Trabalhando com GBM

from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42)

gbm.fit(train[feats], train['Target'])

accuracy_score(test['Target'], gbm.predict(test[feats]))
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)
#Avaliando a importancia de cada coluna

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(25,30))

    

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Limitando o treinamento ao chefe da familia



# Criando um novo dataframe para treinar

heads = train[train['parentesco1'] == 1]
# Feature Engineering



# Vamos criar novas colunas para valores percapita

heads['hsize-pc'] = heads['hhsize'] / heads['tamviv']

heads['phone-pc'] = heads['qmobilephone'] / heads['tamviv']

heads['tablets-pc'] = heads['v18q1'] / heads['tamviv']

heads['rooms-pc'] = heads['rooms'] / heads['tamviv']

heads['rent-pc'] = heads['v2a1'] / heads['tamviv']
# Criando um novo modelo

rf2 = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')
# Treinando o modelo

rf2.fit(heads[feats], heads['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf2.predict(test[feats]).astype(int)
# Vamos verificar as previsões

test['Target'].value_counts(normalize=True)
# Trabalhando com AdaBoost

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

abc = AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=42)

abc.fit(heads[feats], heads['Target'])

accuracy_score(test['Target'], abc.predict(test[feats]))
# Trabalhando com GBM

from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42)

gbm.fit(heads[feats], heads['Target'])

accuracy_score(test['Target'], gbm.predict(test[feats]))
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)
#Avaliando a importancia de cada coluna

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(25,30))

    

pd.Series(rf2.feature_importances_, index=feats).sort_values().plot.barh()
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix

import itertools



# Imprimir a matriz de confusão no modelo de test

print(classification_report(Id, Target))





cmat = confusion_matrix(Id, Target)





plt.figure(figsize = (10,7))

sns.set(font_scale=1.4) # for label size

sns.heatmap(cmat, annot=True, fmt="d") # font size





print('Verdade Negativo {}'.format(cmat[0,0]))

print('Falso Positivo {}'.format(cmat[0,1]))

print('Falso Negativo {}'.format(cmat[1,0]))

print('Verdadeiro Positivo {}'.format(cmat[1,1]))

print('Acurácia: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))

print('Classificação: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))



error_rate = []

acc = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(Id, Target)



    acc.append(knn.score(Id, Target))



    # Plotando o erro



plt.figure(figsize=(10,4))

plt.plot(range(1,40), acc, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Accuracia vs. K-Valores')

plt.xlabel('K-Valores')

plt.ylabel('Accuracia')

plt.show()
cm = confusion_matrix('Target', test['prediction'])