import numpy as np

import pandas as pd



#Visualização gráfica

import matplotlib.pyplot as plt

import seaborn as sns



#Definindo os parâmetros dos gráficos

%matplotlib inline

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['patch.edgecolor'] = 'k'

plt.rcParams['font.sans-serif']=['SimHei']

plt.rcParams['axes.unicode_minus'] = False



#importando base do kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# mostrando o máximo de colunas

pd.options.display.max_columns = 150



#Lendo os dados

train = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')

##Importanto bibliotecas para gerar relátorio da análise exploratória 

from pandas_profiling import ProfileReport

import requests

import pandas_profiling

import matplotlib.pyplot as plt
## Instalando biblioteca

!pip install -U pandas_profiling
#Gerando relatório da análise exploratória



profile = ProfileReport(train, title='Exploratória Pobreza da Costa Rica',html={'style':{'full_width':True}})
##Gerando visualização do relatório da análise exploratória        

profile.to_notebook_iframe()
#visão geral do dataframe de treino e teste

train.shape, test.shape
# Adicionando uma coluna target vazia ao conjunto de teste 

test['Target'] = np.nan

test
#obtendo informações do dataframe de treino

train.info()

# Quais colunas do dataframe de treino são do tipo object

train.select_dtypes('object').head()

# Quais colunas do dataframe de treino são do tipo inteiro

train.select_dtypes('int64').head()

# Quais colunas do dataframe de treino são do tipo float

train.select_dtypes('float').head()
#agregando os valores inteiros do dataframe de treino

train.select_dtypes(include=['int64']).nunique().value_counts().sort_index().plot.bar(color = 'blue', figsize = (8, 6),edgecolor = 'k', linewidth = 2)

plt.xlabel('Distribuição de valores individuais')

plt.ylabel('Quantidade')

plt.title('O número de valores únicos na coluna inteira')
from collections import OrderedDict



plt.figure(figsize = (20, 16))

plt.style.use('fivethirtyeight')



# Dicionário de cores



colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})

poverty_mapping = OrderedDict({1: 'Pobreza extrema', 2: 'Pobreza moderada', 3: 'Famílias vuneráveis', 4: 'Famílias não vuneráveis'})



# Iterar sobre números de ponto flutuante

for i, col in enumerate(train.select_dtypes(include=['floating'])):

    ax = plt.subplot(4, 2, i + 1)

    # Níveis de pobreza transversais

    for poverty_level, color in colors.items():

        # Desenhando uma linha para cada nível de pobreza

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(),ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} - Distribuição'); plt.xlabel(f'{col}'); plt.ylabel('Densidade')



plt.subplots_adjust(top = 2)
#juntando os dataframes

data = train.append(test, ignore_index = True)

data.shape
# Olhando a coluna dependency

data['dependency'].value_counts()
# Analisando os dados da coluna edjefa

data['edjefa'].value_counts()
# Analisando os dados da coluna edjefe

data['edjefe'].value_counts()
#Verificando a coluna dependency com valor 'yes'.

(data['dependency'] == 'yes').value_counts()

#Verificando a coluna dependency com valor 'no'

(data['dependency'] == 'no').value_counts()
#Verificando a coluna edjefa com valor 'yes'

(data['edjefa'] == 'yes').value_counts()
#Verificando a coluna edjefa com valor 'no'

(data['edjefa'] == 'no').value_counts()
#Verificando a coluna edjefe com valor 'yes'

(data['edjefe'] == 'yes').value_counts()
#Verificando a coluna edjefe com valor 'no'

(data['edjefe'] == 'no').value_counts()
#Mapeando onde tem 'yes' e substituindo por 1 e 'no' por 0.

data["edjefe"] = data["edjefe"].apply(lambda x: 1 if x == "yes" else x).apply(lambda x: 0 if x == "no" else x)



data["edjefa"] = data["edjefa"].apply(lambda x: 1 if x == "yes" else x).apply(lambda x: 0 if x == "no" else x)



data["dependency"] = data["dependency"].apply(lambda x: 1 if x == "yes" else x).apply(lambda x: 0 if x == "no" else x)



data["edjefe"] = pd.to_numeric(data["edjefe"])



data["edjefa"] = pd.to_numeric(data["edjefa"])



data["dependency"] = pd.to_numeric(data["dependency"])



data[['dependency', 'edjefa', 'edjefe']].describe()
plt.figure(figsize = (16, 12))



# Iterar sobre números de ponto flutuante

for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):

    ax = plt.subplot(3, 1, i + 1)

    # Níveis de pobreza transversais

    for poverty_level, color in colors.items():

        # Desenhando uma linha para cada nível de pobreza

        sns.kdeplot(data.loc[data['Target'] == poverty_level, col].dropna(),ax = ax, color = color, label = poverty_mapping[poverty_level])

      

    plt.title(f'{col.capitalize()} - Distribuição'); 

    plt.xlabel(f'{col}'); 

    plt.ylabel('Densidade')



plt.subplots_adjust(top = 2)
data.info()
# Verificando a composição do nível de pobreza das famílias



heads = data.loc[data['parentesco1'] == 1].copy()



# tags em treinamento

train_labels = data.loc[(data['Target'].notnull()) & (data['parentesco1'] == 1), ['Target', 'idhogar']]



# Valores da váriavel Target

label_counts = train_labels['Target'].value_counts().sort_index()



# Gráfico de barras que aparece para cada tag

label_counts.plot.bar(figsize = (8, 6), color = colors.values(),edgecolor = 'k', linewidth = 2)



# formato

plt.xlabel('Nível de pobreza'); plt.ylabel('Quantidade'); 

plt.xticks([x - 1 for x in poverty_mapping.keys()], 

           list(poverty_mapping.values()), rotation = 60)

plt.title('Composição do nível de pobreza');



label_counts
#Agrupando os membros da família pelo nível de pobreza e para identificar se coincide a target para todos

all_igual = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Famílias com niveis de pobrezas diferentes

not_igual = all_igual[all_igual != True]

print('Existem {} membros da família que têm diferentes valores de níveis de pobreza'.format(len(not_igual)))
#Exemplo de uma família com níveis de pobreza diferentes

data[data['idhogar'] == not_igual.index[0]][['idhogar', 'parentesco1', 'Target']]
#identificando a quantidade de chefe de familia na base de treinamento

chefe_familia = data.groupby('idhogar')['parentesco1'].sum()

chefe_familia.sum()
chefe_familia = data.groupby('idhogar')['parentesco1'].sum()



# Encontrando famílias sem chefe

familia_sem_chefe = data.loc[data['idhogar'].isin(chefe_familia[chefe_familia == 0].index), :]



print('Existem {} famílias sem chefe'.format(familia_sem_chefe['idhogar'].nunique()))
# Localizando famílias sem chefe com niveis de pobrezas diferente

familia_sem_chefe_iguais = familia_sem_chefe.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

print('Exite {} famílias sem chefe com niveis de pobrezas diferente'.format(sum(familia_sem_chefe_iguais == False)))
# Percorrendo por todas as famílias

for familia in not_igual.index:

    # Encontrando a target correta para o chefe da família

    true_target = int(data[(data['idhogar'] == familia) & (data['parentesco1'] == 1.0)]['Target'])

    

    # Definindo rótulos corretos para todos os membros da família

    data.loc[data['idhogar'] == familia, 'Target'] = true_target

    

    

# Agrupando por família e encontrando o número de valores únicos

all_igual = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Famílias com niveis de pobrezas diferentes

not_igual = all_igual[all_igual != True]

print('Existem {} famílias no total, e nem todos os membros da família têm os mesmos níveis de pobreza.'.format(len(not_igual)))
# selecionando apenas a linha do chefe da família.

def plot_value_counts(df, col, apenas_chefe = False):

    # Selecionando o chefe da família

    if apenas_chefe:

        df = df.loc[df['parentesco1'] == 1].copy()

        

    plt.figure(figsize = (8, 6))

    df[col].value_counts().sort_index().plot.bar(color = 'red',

                                                 edgecolor = 'k',

                                                 linewidth = 2)

    plt.xlabel(f'{col}')

    plt.title(f'{col}- Contagens de valor')

    plt.ylabel('Contagem')

    plt.show()
#Plotando por chefe da familia que possui tablet

plot_value_counts(heads, 'v18q1')
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
#preenchendo com zero os valores nulos para a coluna v18q1

data['v18q1'] = data['v18q1'].fillna(0)
# Defina as variáveis de propriedade da casa

prop_var = [x for x in data if x.startswith('tipo')]



# Gráfico para mostrar os aluguéis não pagos na casa

data.loc[data['v2a1'].isnull(), prop_var].sum().plot.bar(figsize = (10, 8),color = 'green',edgecolor = 'k', linewidth = 2)

plt.xticks([0, 1, 2, 3, 4],['comprado e pago', 'Propriedade e pagando', 'Alugado', 'Instável', 'outro (atribuído, emprestado)'],rotation = 60)

plt.title('Status de propriedade de casa para famílias que faltam pagamentos de aluguel', size = 18)
# Preenchendo as famílias que possuem a casa com zero pagamento de aluguel

data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0



#encontrando a idade máxima para pessoas que estão na escola

data.loc[data['rez_esc'].notnull()]['age'].describe()
#Encontrando a idade de pessoas que não estão na escola

data.loc[data['rez_esc'].isnull()]['age'].describe()

# Definindo como zero, se o indivíduo tiver mais de 19 anos ou menos de 7 anos e estiver faltando anos atrás

data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0





# Numeros de valores faltantes de cada coluna

val_ausentes = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})



# Percentual de valores faltantes em cada coluna

val_ausentes['Percentual'] = val_ausentes['total'] / len(data)



val_ausentes.sort_values('Percentual', ascending = False).head(10).drop('Target')
#localizando valores maiores que 5 na coluna 'rez_esc'e substituindo por 5, pois valores maiores que isso, são considerados outliers

data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5
#Visualizando que a coluna não valores maiores que cinco.

(train['rez_esc'] > 5).value_counts()
# Verificando os valores nulos

data.isnull().sum()
 # Verificando os valores de aluguel (v2a1) para os chefes/as de familia (parentesco1 = 1)

data[data['parentesco1'] == 1]['v2a1'].isnull().sum()
# Qual a cara dos dados de v18q

data['v18q'].value_counts()
# Prenchendo com -1 os valores nulos de v2a1 para não serem utilizados no modelo

data['v2a1'].fillna(-1, inplace=True)
# Prenchendo com 0 os valores nulos de v18q1

data['v18q1'].fillna(0, inplace=True)
# Verificando os valores nulos

data.isnull().sum().sort_values()
# Prenchendo com -1 os valores nulos de SQBmeaned, meaneduc e rez_esc para não serem utilizados no modelo

data['SQBmeaned'].fillna(-1, inplace=True)

data['meaneduc'].fillna(-1, inplace=True)

data['rez_esc'].fillna(-1, inplace=True)
# Separando as colunas para treinamento

feats = [c for c in data.columns if c not in ['Id', 'idhogar', 'Target']]
# Separar os dataframes

train, test = data[~data['Target'].isnull()], data[data['Target'].isnull()]



train.shape, test.shape
data['Target'].value_counts().sort_values()
# Instanciando o random forest classifier

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)





# Treinando o modelo

rf.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf.predict(test[feats]).astype(int)



#Verificando as previsões

test['Target'].value_counts(normalize = True)
# Vamos verificar as previsões nos últimos registros

test[['Target', 'Id']].tail()
# melhorando o modelo com AdaBoost e verificando a acurácia 

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

abc = AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=42)

abc.fit(train[feats], train['Target'])

accuracy_score(test['Target'], abc.predict(test[feats]))
# verificando a acurácia com o modelo Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=42)

gbc.fit(train[feats], train['Target'])

accuracy_score(test['Target'], gbc.predict(test[feats]))
# Criando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)
import matplotlib.pyplot as plt



fig=plt.figure(figsize=(15, 20))



# Avaliando a importancia de cada coluna (cada variável de entrada)

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()