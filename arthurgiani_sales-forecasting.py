# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Carregando os datasets



df_train = pd.read_csv('../input/dataset_treino.csv')

df_test = pd.read_csv('../input/dataset_teste.csv')

df_stores = pd.read_csv('../input/lojas.csv')
#Realizando a união dos datasets originais com as informações complementares das lojas





df_train = df_train.merge(df_stores, on = 'Store', how = 'left')

df_test = df_test.merge(df_stores, on = 'Store', how = 'left')



print(df_train.shape, df_test.shape)
df_train.head()
df_test.head()
#Isolando variável ID para posterior submissão ao Kaggle



var_id = df_test['Id']



var_id.head()



#Excluíndo Variável ID 



df_test = df_test.drop(['Id'], axis = 1)



df_test.head()
#Excluíndo a variável costumers, que está no df_train mas não no df_test



df_train = df_train.drop(['Customers'], axis = 1)



df_train.head()
#Movendo a variável target (Sales) para a última coluna para facilitar o slicing quando necessário.



df_train['Sales1'] = df_train['Sales']

df_train = df_train.drop(['Sales'], axis = 1)



df_train.head()
#Junção dos datasets treino e teste



df_test['Sales1'] = -1

df_united = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)



df_united.head()
print('shapes: ', df_united[df_united['Sales1'] != -1].shape, df_united[df_united['Sales1'] == -1].shape)
#Verificando valores nulos



pd.DataFrame(df_united.isnull().sum().sort_values(ascending=False))
#Excluíndo colunas com aprox 50% de dados missing



df_united = df_united.drop(['Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval'], axis = 1)



df_united.head()

#Verificação do tipo dos dados. Ajustes serão feitos mais tarde



df_united.dtypes

#Extraíndo mês, dia e ano da colunas de datas



df_united['Date'] = pd.to_datetime(df_united['Date'])



df_united['Year'] = df_united['Date'].apply(lambda data: data.year)

df_united['Month'] = df_united['Date'].apply(lambda data: data.month)

df_united['Day'] = df_united['Date'].apply(lambda data: data.day)

df_united.head()
#Recolocando a variável target na última coluna



df_united['Sales Amount'] = df_united['Sales1']

df_united = df_united.drop(['Sales1'], axis = 1)



df_united.head()
#Reorganizando a ordem das colunas



df_united = df_united[['Store',

 'DayOfWeek',

 'Date',

 'Year',

 'Month',

 'Day',

 'Open',

 'Promo',

 'Promo2',

 'StateHoliday',

 'SchoolHoliday',

 'StoreType',

 'Assortment',

 'CompetitionDistance',

 'CompetitionOpenSinceMonth',

 'CompetitionOpenSinceYear',

 'Sales Amount']]

df_united.head()
#Tipo de cada coluna



df_united.dtypes
#Verificando correlação preliminar dos dados numéricos



dataset_num = df_united._get_numeric_data()



import seaborn as sns



corr = dataset_num.corr()

_ , ax = plt.subplots( figsize =( 30 , 30 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })



pd.DataFrame(df_united.isnull().sum().sort_values(ascending=False))
#Exclusão das colunas CompetitionOpenSinceMonth e CompetitionOpenSinceYear pois possuem quantidade significativa

#de dados missing e não apresentam forte correlação com as vendas.



df_united = df_united.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis = 1)



df_united.head()



#Os dados missing das variáveis 'CompetitionDistance' e 'Open' serão tratados mais tarde.
#Determinando o tipo de cada variável



convert_dict = {'Open':object, 'Promo':object, 'Promo2':object, 'SchoolHoliday':object, 'Year': object, 'Store': object

               , 'DayOfWeek': object, 'Month':object, 'Day': object}





df_united = df_united.astype(convert_dict) 

print(df_united.dtypes)
#Análise Exploratória das variáveis numéricas



dataset_num = df_united._get_numeric_data()



columns=dataset_num.columns

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in zip(columns,range(length)):

    plt.subplot((length/2),2,j+1)

    plt.subplots_adjust(wspace=0.4,hspace=0.1)

    dataset_num[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
#Análise das variáveis categóricas



#Excluíndo a coluna 'date' que não será mais necessária



df_united = df_united.drop(['Date'], axis = 1)

df_united.head()
#Pré processamento



#Tratamento nos valores missing de 'CompetitionDistance' e 'Promo'



df_united['CompetitionDistance'].fillna(df_united['CompetitionDistance'].mean(), inplace=True)



df_united['CompetitionDistance'] = df_united['CompetitionDistance'].astype(int)



df_united.head()
df_united.dtypes
df_united['Open'] = df_united['Open'].fillna(0)



df_united['Open'] = df_united['Open'].astype('object')



df_united.head()
pd.DataFrame(df_united.isnull().sum().sort_values(ascending=False))



#Tudo ok
#Preparação dos dados para o algoritmo de ML



df_united.dtypes
#Transformation in 'StoreType' ,'Assortment' and 'StateHoliday' columns (string to number)



def StoreType_numeric(x):

    if x=='a':

        return 1

    if x=='b':

        return 2

    if x=='c':

        return 3

    if x=='d':

        return 4

    

    

df_united['StoreType'] = df_united['StoreType'].apply(StoreType_numeric)

df_united['StoreType'] = df_united['StoreType'].astype(object)

df_united.head()

def Assortment_numeric(x):

    if x=='a':

        return 1

    if x=='b':

        return 2

    if x=='c':

        return 3

    

    

df_united['Assortment'] = df_united['Assortment'].apply(StoreType_numeric)

df_united['Assortment'] = df_united['Assortment'].astype(object)

df_united.head()
pd.DataFrame(df_united.isnull().sum().sort_values(ascending=False))
def Holiday_numeric(x):

    if x=='0':

        return 0

    if x==0:

        return 0

    if x=='a':

        return 1

    if x=='b':

        return 2

    if x=='c':

        return 3

      

    

df_united['StateHoliday'] = df_united['StateHoliday'].apply(Holiday_numeric)

df_united['StateHoliday'] = df_united['StateHoliday'].astype(object)

df_united.head()
#Splitting dataframe in 3 parts



#df_norm = numerical set (normalization process)

#df_categ = categorical features

#target = target variable(Sales Amount)





df_norm = df_united['CompetitionDistance']

df_categ = df_united.select_dtypes(include=[object])

target = df_united['Sales Amount']







#Adjusting df_norm and target



df_norm = pd.DataFrame(df_norm)

target = pd.DataFrame(target)



print(df_norm.shape, target.shape)



from sklearn import preprocessing



x = df_norm



min_max_scaler = preprocessing.MinMaxScaler()



x_scaled = min_max_scaler.fit_transform(x)



df_1 = pd.DataFrame(x_scaled)



df_1.head()
#Concatenate datasets again



df_final = pd.concat([df_categ, df_norm, target], axis = 1)



df_final.head()



#Machine Learning Process

#Dataset split

train_set = df_final[df_final['Sales Amount'] != -1]

test_set = df_final[df_final['Sales Amount'] == -1]



print(train_set.shape, test_set.shape)
#Drop artificial information from 'Sales amount' on teste set from the beggining of script



x_test = test_set.drop('Sales Amount', axis = 1)

y_test = test_set['Sales Amount'] #Just an artificial information to use on cross_val_predict in ML process



test_set.head()
#X and Y train for cross validation process



x_train = train_set.iloc[:,0:13] 

y_train = train_set['Sales Amount']
x_train.head()
from sklearn.ensemble import RandomForestRegressor



# Treinar o modelo com 20 árvores de decisão

model = RandomForestRegressor(n_estimators = 100, random_state = 7)



# Treinando o modelo com o dataset de treino com as variáveis importantes



model.fit(x_train, y_train)



#Calculando a precisão do modelo

R2 = model.score(x_train, y_train)



print("Accuracy:" %R2)



R2




pred = np.round(model.predict(x_test)).astype(int)

pred[pred < 0] = 0





submission = pd.DataFrame({'Id': var_id, 'Sales': pred })

submission.to_csv('submission.csv',index=False)
