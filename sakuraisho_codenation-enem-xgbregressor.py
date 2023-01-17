import matplotlib.pyplot as plt

plt.style.use('ggplot') 

import pandas as pd

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from scipy import stats

from xgboost import XGBRegressor

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Realiza a leitura da base (csv) e carrega na mémoria

enemTrain = pd.read_csv('../input/codenation-enem2/train.csv')

enemTest = pd.read_csv('../input/codenation-enem2/test.csv')
#verificando o tamanho da Base

enemTrain.shape
#Verificando os nomes das colunas

enemTrain.columns
import qgrid

qgrid.show_qgri(enemTrain)
import pandas_profiling

profile = enemTrain.profile_report(title="Enem Dataset")
profile
#vendo os 5 primeiros registros

enemTrain.head()
#verificando a tipagem dos dados

enemTrain.dtypes
#Verificando valores nulos

total = enemTrain.isnull().sum().sort_values(ascending=False)

percent = (enemTrain.isnull().sum()/enemTrain.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#Verificando os valores nulos e mostrando a quantidade por coluna

for i in enemTrain.columns:

    nulls_value = enemTrain[i].isna().sum()

    message = "Column {} has {} nulls".format(i, nulls_value)

    print(message)
#Realizando algumas analises

df = enemTrain.groupby('TP_SEXO').size()



df.plot(kind='pie', subplots=True, figsize=(8, 4))

plt.title("Pie Chart Genero")

plt.ylabel("")

plt.show()





df = enemTrain.groupby('TP_ENSINO').size()



df.plot(kind='pie', subplots=True, figsize=(8, 4))

plt.title("Pie Chart TP_ENSINO")

plt.ylabel("")

plt.show()



df = enemTrain.groupby('TP_COR_RACA').size()



df.plot(kind='pie', subplots=True, figsize=(8, 4))

plt.title("Pie Chart TP_COR_RACA")

plt.ylabel("")

plt.show()
import squarify 



df_raw = enemTrain



df = df_raw.groupby('SG_UF_RESIDENCIA').size().reset_index(name='counts')

labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)

sizes = df['counts'].values.tolist()

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]



plt.figure(figsize=(12,8), dpi= 80)

squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)



plt.title('Treemap of Vechile Class')

plt.axis('off')

plt.show()
sns.set(rc={'figure.figsize':(9,4)})

sns.distplot(enemTrain['IN_CEGUEIRA']);
sns.set(rc={'figure.figsize':(9,4)})

sns.distplot(enemTrain['IN_SURDEZ']);
sns.set(rc={'figure.figsize':(9,4)})

sns.distplot(enemTrain['IN_DEFICIENCIA_FISICA']);
sns.set(rc={'figure.figsize':(9,4)})

sns.distplot(enemTrain['IN_DEFICIENCIA_MENTAL']);
#Análise estatística

enemTrain['NU_NOTA_MT'].describe()
#Frequencia 

enemTrain.plot.hist(y='NU_NOTA_MT')
#Relação entre Nota de matemáticas e outras matérias

enemTrain.plot.scatter(x='NU_NOTA_CN', y='NU_NOTA_MT')

enemTrain.plot.scatter(x='NU_NOTA_CH', y='NU_NOTA_MT')

enemTrain.plot.scatter(x='NU_NOTA_LC', y='NU_NOTA_MT')

enemTrain.plot.scatter(x='NU_NOTA_REDACAO', y='NU_NOTA_MT')


df_counts = enemTrain.groupby(['NU_NOTA_MT', 'NU_IDADE']).size().reset_index(name='counts')



# Desenhar Stripplot

fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    

sns.stripplot(df_counts.NU_IDADE, df_counts.NU_NOTA_MT, size=df_counts.counts*2, ax=ax)





plt.title('Relação entre Idade e Nota em Matemática', fontsize=22)

plt.show()
#Para vermos as colunas que possuem maior correlação

aux = enemTrain.copy()

aux2 = enemTrain.copy()



aux = aux.loc[:, enemTest.columns]

aux['NU_NOTA_MT'] = aux2.NU_NOTA_MT



c = aux.corr()

c.NU_NOTA_MT.sort_values()
#Separando para trabalhar só com o que importa - tem maior relação

new_vector_training = [

    'NU_NOTA_COMP1',

    'NU_NOTA_COMP2',

    'NU_NOTA_COMP4',

    'NU_NOTA_COMP5',

    'NU_NOTA_COMP3',

    'NU_NOTA_REDACAO',

    'NU_NOTA_LC',

    'NU_NOTA_CH',

    'NU_NOTA_CN',

    'NU_NOTA_MT'

]



new_vector_test = [

    'NU_INSCRICAO',

    'NU_NOTA_COMP1',

    'NU_NOTA_COMP2',

    'NU_NOTA_COMP4',

    'NU_NOTA_COMP5',

    'NU_NOTA_COMP3',

    'NU_NOTA_REDACAO',

    'NU_NOTA_LC',

    'NU_NOTA_CH',

    'NU_NOTA_CN'

]



enemTrain_data = enemTrain.copy()

enemTrain_data = enemTrain_data.loc[:, new_vector_training]

enemTrain_data.dropna(subset=['NU_NOTA_MT'], inplace=True)

enemTrain_data.head()
y = enemTrain_data.NU_NOTA_MT

X = enemTrain_data.drop(['NU_NOTA_MT'], axis=1)



enem_validation_data = enemTest.copy()

enem_validation_data_1 = enem_validation_data.loc[:, new_vector_test]

enem_validation_data_2 = enem_validation_data.loc[:, new_vector_test]
enem_train_X, enem_validation_X, enem_train_y, enem_validation_y = train_test_split(X, y, random_state = 0)
model = XGBRegressor(n_estimators=200, learning_rate=0.1)

model.fit(enem_train_X, enem_train_y, early_stopping_rounds=5, eval_set=[(enem_validation_X, enem_validation_y)], verbose=False)



enem_validation_data_1.drop(['NU_INSCRICAO'], axis=1, inplace=True)
predicted_nota = model.predict(enem_validation_data_1)

result_df = pd.DataFrame({'NU_INSCRICAO': enem_validation_data_2['NU_INSCRICAO'], 'NU_NOTA_MT': predicted_nota})

result_df.head()
#Verifica se tem valores nulos no result

result_df.isnull().any().any()
result_df['NU_NOTA_MT'].describe()
result_df_final = result_df.loc[: , ['NU_INSCRICAO', 'NU_NOTA_MT']]

result_df.to_csv('answer.csv', index=False)
