#Atividade 2 - Aprendizado de Máquina - FACENS

#Bruno Silva

#Denise Leite

#Milena Rocha
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importando o arquivo de treino

df_train = pd.read_csv('../input/train.csv')

df_train.describe()
#Verificando se existem dados duplicados

print('Antes:', df_train.shape)

df_train.drop_duplicates()

print('Depois:', df_train.shape)
#Importante o arquivo de teste

df_test = pd.read_csv('../input/test.csv')

df_test.describe()
#Verificando se existem dados duplicados

print('Antes:', df_test.shape)

df_test.drop_duplicates()

print('Depois:', df_test.shape)
#Verificando o tamanho do dataset de treino e teste

print('Train: ', df_train.shape)

print('Test:  ', df_test.shape)
#Separando os dataset de treino para criar o ds meta dos dados

data = []

for f in df_train.columns:

    # definindo o uso (entre rótulo, id e atributos)

    if f == 'target':

        role = 'target' # rótulo

    elif f == 'id':

        role = 'id'

    else:

        role = 'input' # atributos

         

    # definindo o tipo do dado

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f == 'id':

        level = 'nominal'

    elif df_train[f].dtype == float:

        level = 'interval'

    elif df_train[f].dtype == int:

        level = 'ordinal'

        

    # mantem keep como verdadeiro pra tudo, exceto id

    keep = True

    if f == 'id':

        keep = False

    

    # cria o tipo de dado

    dtype = df_train[f].dtype

    

    # cria dicionário de metadados

    f_dict = {

        'varname': f,

        'role': role,

        'level': level,

        'keep': keep,

        'dtype': dtype

    }

    data.append(f_dict)

    

meta_train = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

meta_train.set_index('varname', inplace=True)
#Visualizando o meta de treino

meta_train
#Separando os dataset de teste para criar o ds meta dos dados

data = []

for f in df_test.columns:

    # definindo o uso (entre rótulo, id e atributos)

    if f == 'target':

        role = 'target' # rótulo

    elif f == 'id':

        role = 'id'

    else:

        role = 'input' # atributos

         

    # definindo o tipo do dado

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f == 'id':

        level = 'nominal'

    elif df_test[f].dtype == float:

        level = 'interval'

    elif df_test[f].dtype == int:

        level = 'ordinal'

        

    # mantem keep como verdadeiro pra tudo, exceto id

    keep = True

    if f == 'id':

        keep = False

    

    # cria o tipo de dado

    dtype = df_test[f].dtype

    

    # cria dicionário de metadados

    f_dict = {

        'varname': f,

        'role': role,

        'level': level,

        'keep': keep,

        'dtype': dtype

    }

    data.append(f_dict)

    

meta_test = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

meta_test.set_index('varname', inplace=True)
meta_test
#Filtrando as variáveis a serem mantidas DS de Treino

meta_train[(meta_train.level == 'nominal') & (meta_train.keep)].index
#Filtrando as variáveis a serem mantidas DS Teste

meta_test[(meta_test.level == 'nominal') & (meta_test.keep)].index
#Dataset de Treino

pd.DataFrame({'count' : meta_train.groupby(['role', 'level'])['role'].size()}).reset_index()
#Datast de Teste

pd.DataFrame({'count' : meta_test.groupby(['role', 'level'])['role'].size()}).reset_index()
#Analisando o Dataset de Treino

atributos_missing = []



for f in df_train.columns:

    missings = df_train[df_train[f] == -1][f].count()

    if missings > 0:

        atributos_missing.append(f)

        missings_perc = missings/df_train.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))

        

print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
#Analisando o Dataset de Teste

atributos_missing_test = []



for f in df_test.columns:

    missings_test = df_test[df_test[f] == -1][f].count()

    if missings_test > 0:

        atributos_missing_test.append(f)

        missings_perc_test = missings_test/df_test.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings_test, missings_perc_test))

        

print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing_test)))
# removendo ps_car_03_cat e ps_car_05_cat que tem muitos valores faltantes

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

train = df_train.drop(vars_to_drop, axis=1)

test = df_test.drop(vars_to_drop, axis=1)

meta_train.loc[(vars_to_drop),'keep'] = False  # atualiza os metadados para ter como referência (processar o test depois)

meta_test.loc[(vars_to_drop),'keep'] = False
from sklearn.preprocessing import Imputer



media_imp = Imputer(missing_values=-1, strategy='mean', axis=0)

moda_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

train['ps_reg_03'] = media_imp.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_12'] = media_imp.fit_transform(train[['ps_car_12']]).ravel()

train['ps_car_14'] = media_imp.fit_transform(train[['ps_car_14']]).ravel()

train['ps_car_11'] = moda_imp.fit_transform(train[['ps_car_11']]).ravel()



test['ps_reg_03'] = media_imp.fit_transform(test[['ps_reg_03']]).ravel()

test['ps_car_12'] = media_imp.fit_transform(test[['ps_car_12']]).ravel()

test['ps_car_14'] = media_imp.fit_transform(test[['ps_car_14']]).ravel()

test['ps_car_11'] = moda_imp.fit_transform(test[['ps_car_11']]).ravel()
v = meta_train[(meta_train.level == 'nominal') & (meta_train.keep)].index



for f in v:

    dist_values = train[f].value_counts().shape[0]

    print('Atributo {} tem {} valores distintos'.format(f, dist_values))
v = meta_test[(meta_test.level == 'nominal') & (meta_test.keep)].index



for f in v:

    dist_values = train[f].value_counts().shape[0]

    print('Atributo {} tem {} valores distintos'.format(f, dist_values))
v = meta_train[(meta_train.level == 'nominal') & (meta_train.keep)].index

print('Antes do one-hot encoding tinha-se {} atributos'.format(train.shape[1]))

train = pd.get_dummies(train, columns=v, drop_first=True)

print('Depois do one-hot encoding tem-se {} atributos'.format(train.shape[1]))



print('Antes do one-hot encoding tinha-se {} atributos'.format(test.shape[1]))

test = pd.get_dummies(test, columns=v, drop_first=True)

print('Depois do one-hot encoding tem-se {} atributos'.format(test.shape[1]))



missing_cols = set( train.columns ) - set( test.columns )



print(missing_cols)

for c in missing_cols:

    test[c] = 0

    

train, test = train.align(test, axis=1)
print(train.shape)

print(test.shape)
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']



X_test  = test.drop(['id', 'target'], axis=1)

y_test  = test['target']



from sklearn.linear_model import LogisticRegression



model = LogisticRegression(class_weight='balanced')



model.fit(X_train, y_train)

model.score(X_test, y_test)
#Creating Submission file

y_pred = model.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'target':y_pred})

#submit.head() 

submit.to_csv('submission_log.csv',index=False) 
from sklearn import linear_model

from sklearn import metrics



model = linear_model.LassoCV(alphas=[1e-3, 1e-2, 1e-1, 1])

model.fit(X_train, y_train)



erro_treino = metrics.mean_squared_error(y_train,model.predict(X_train))

print('RMSE no treino:', erro_treino)



erro_teste = metrics.mean_squared_error(y_test,model.predict(X_test))

print('RMSE no teste:', erro_teste)
#Creating Submission file

y_pred = model.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'target':y_pred})

#submit.head() 

submit.to_csv('submission_lin.csv',index=False) 