import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
df=pd.read_csv('../input/CanaisAtend.csv', encoding='utf-8', sep=';')

df.sample(5)
# Verificando os tipos das colunas

print('==== Colunas por Tipo ====') ,

print('                          '),

print(df.dtypes.value_counts()),

print('                          '),

print('==========================')

print('==  Colunas por Tipo[%]  ==')

print('                          ')

print(df.dtypes.value_counts(normalize=True).apply("{:.2%}".format))

print('                          ')

print('==========================')

print('==========Shape===========')

print('                          ')

print(df.shape)

print('                          ')

print('==========================')

print('======== Colunas =========')

print('                          ')

print(df.columns)

print('                          ')

print('==========================')
df['entrevistados']=df['entrevistados'].astype(float)

df['Md_Ponderada']=df['Md_Ponderada'].str.replace(',','.').convert_objects(convert_numeric=True)

df['Nota_0']=df['Nota_0'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_1']=df['Nota_1'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_2']=df['Nota_2'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_3']=df['Nota_3'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_4']=df['Nota_4'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_5']=df['Nota_5'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_6']=df['Nota_6'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_7']=df['Nota_7'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_8']=df['Nota_8'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_9']=df['Nota_9'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['Nota_10']=df['Nota_10'].str.replace(',','.').convert_objects(convert_numeric=True)/100 

df['NS_NR']=df['NS_NR'].str.replace(',','.').convert_objects(convert_numeric=True)/100

df['nota']=df['Md_Ponderada']

df=df.drop(columns=['Md_Ponderada','Md_Indicadores'])

df.columns = map(str.lower,df.columns)
# Verificando os tipos das colunas

print('Verificando os tipos das colunas após transformações')

print('                          ')

print('==== Colunas por Tipo ====')

print('                          ')

print(df.dtypes.value_counts())

print('                          ')

print('==========================')

print('==  Colunas por Tipo[%]  ==')

print('                          ')

print(df.dtypes.value_counts(normalize=True).apply("{:.2%}".format))

print('                          ')

print('==========================')

print('==========Shape===========')

print('                          ')

print(df.shape)

print('                          ')

print('==========================')

print('======== Colunas =========')

print('                          ')

print(df.columns)

print('                          ')

print('==========================')
df.sample(3)
df[(df["estado"]=='Total') & (df['questao']=='A3A4A5')].sample(5)
unique_counts = pd.DataFrame.from_records([(col, df[col].nunique()) for col in df.columns],

                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])



unique_counts.T
print(df['estado'].unique())
df[(df['ano']!=2018) & (df['estado']!='Total') & (df['questao']!='A3A4A5') & (df['operadora']!='TOTAL BRASIL')]
# Criando dataframe temp para novas colunas

temp=df[(df['questao']!='A3A4A5') & (df['estado']!='Total') & (df['operadora']!='TOTAL BRASIL')]
# avaliando as colunas que serão convertidas em categorias

unique_counts = pd.DataFrame.from_records([(col, temp[col].nunique()) for col in temp.columns],

                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])



unique_counts
#Verificando dos dados das colunas

print('=======================')

print(temp['servico'].unique())

print('=======================')

print(temp['questao'].unique())

print('=======================')

print(temp['tema'].unique())

print('=======================')

print(temp['ano'].unique())

print('=======================')

print(temp['operadora'].unique())

print('=======================')

print(np.sort(temp['estado'].unique()))

print('=======================')
# colunas escolhidas

temp['ct_questao'] = temp['questao'].astype('category').cat.codes

temp['ct_operadora']=temp['operadora'].astype('category').cat.codes

temp['ct_estado']=temp['estado'].astype('category').cat.codes
#Avaliando o resultado

temp[['ct_questao','questao',

     'ct_operadora','operadora',

     'ct_estado','estado']]
# Verificando os tipos das colunas com as novas infromações

print('==== Colunas por Tipo ====')

print('                          ')

print(temp.dtypes.value_counts())

print('                          ')

print('==========================')

print('==  Colunas por Tipo[%]  ==')

print('                          ')

print(temp.dtypes.value_counts(normalize=True).apply("{:.2%}".format))

print('                          ')

print('==========================')

print('==========Shape===========')

print('                          ')

print(temp.shape)

print('                          ')

print('==========================')

print('======== Colunas =========')

print('                          ')

print(temp.columns)

print('                          ')

print('==========================')
df2=temp[(temp['ano']!=2018) & 

         (temp['questao']!='A3A4A5') & 

         (temp['estado']!='Total') & 

         (temp['operadora']!='TOTAL BRASIL')]



df_2018=temp[(temp['ano']==2018) & 

             (temp['questao']!='A3A4A5') & 

             (temp['estado']!='Total') & 

             (temp['operadora']!='TOTAL BRASIL')]



print(df2['ano'].unique())

print(df_2018['ano'].unique())

#Entrevistados por ano

df2.groupby(['questao','ano'])[['entrevistados']].sum().unstack()
#Abrangência da Operadora

df2.groupby(['operadora','ano'])['estado'].nunique().unstack().reset_index()
#Entrevistados por Estado, Questao e Ano

df2.groupby(['estado','questao','ano'])['entrevistados'].sum().unstack().reset_index()
df2.dtypes
#Criando a métrica que será prevista

df_2018pred = df_2018.drop(columns=['nota'])
df3=df2.append(df_2018pred)
#Criando as bases de treino, validação e teste



from sklearn.model_selection import train_test_split as tts



train, valid = tts(df2,random_state=42)

test = df3[df3['nota'].isnull()]



train.shape, valid.shape, test.shape

# Colunas que não serão usadas no modelo

removed_cols=['servico','operadora','estado','questao','tema']
cols = []

for c in train.columns:

    if c not in removed_cols:

        cols.append(c)

        

cols
# 

feats = [c for c in train.columns if c not in removed_cols]
# Importando os MODELOS

from sklearn.ensemble import RandomForestRegressor as rfr

from sklearn.ensemble import ExtraTreesRegressor as etr

from sklearn.ensemble import AdaBoostRegressor as abr

from sklearn.ensemble import GradientBoostingRegressor as gbr

from sklearn.tree import DecisionTreeRegressor as dtr

from sklearn.linear_model import LinearRegression as lnr

from sklearn.neighbors import KNeighborsRegressor as knr

from sklearn import neighbors

from math import sqrt

from sklearn.svm import SVR as svr
# Dicionário de Modelos

models = {'RandomForest': rfr(random_state=42),

         'ExtraTrees': etr(random_state=42),

         'GradientBoosting': gbr(random_state=42),

         'DecisionTree': dtr(random_state=42),

         'AdaBoost': abr(random_state=42),

         'KNN 11': knr(n_neighbors=11),

         'SVR': svr(),

         'Linear Regression': lnr()}
# Importando métrica

from sklearn.metrics import mean_squared_error
# Função para treinamento dos modelos



def run_model (model, train, valid, feats, y_name):

    model.fit(train[feats], train[y_name])

    preds = model.predict(valid[feats])

    return mean_squared_error(valid[y_name], preds)**(1/2)
#Executando os modelos

scores = []

for name, model in models.items():

    score = run_model(model, train.fillna(-1), valid.fillna(-1), feats, 'nota')

    scores.append(score)

    print(name, ':', score)
from sklearn.model_selection import GridSearchCV



#Selecionando os K

params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}



knn = neighbors.KNeighborsRegressor()



model = GridSearchCV(knn, params, cv=5, n_jobs=-1)

model.fit(train[feats], train[feats])

model.best_params_

rmse_val = [] #to store rmse values for different k

for K in range(20):

    K = K+1

    model = neighbors.KNeighborsRegressor(n_neighbors = K)



    model.fit(train[feats], train[feats])  #fit the model

    pred=model.predict(valid[feats]) #make prediction on test set

    error = sqrt(mean_squared_error(valid[feats],pred)) #calculate rmse

    rmse_val.append(error) #store rmse values

    print('RMSE value for k= ' , K , 'is:', error)
#plotting the rmse values against k values

curve = pd.DataFrame(rmse_val) #elbow curve 

curve.plot()