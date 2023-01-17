import pandas as pd

myfile = pd.read_csv("../input/train.csv")

test_X = pd.read_csv("../input/test.csv")
test_X = test_X[['exp_vida','anos_estudo_empreendedor','idhm', 'perc_pop_econ_ativa']]
test_X = test_X.fillna(test_X.mean())
a = myfile[['densidade_dem']]

b = myfile[['area']]

c = myfile[['codigo_mun']]

d = myfile[['ranking_igm']]

e = myfile[['comissionados_por_servidor']]



myfile['densidade_dem'] = a.apply(lambda x: x.str.replace(',',''))

myfile['area'] = b.apply(lambda x: x.str.replace(',',''))

myfile['codigo_mun'] = c.apply(lambda x: x.str.replace('ID_',''))

myfile['ranking_igm'] = d.apply(lambda x: x.str.replace('º',''))

myfile['comissionados_por_servidor'] = e.apply(lambda x: x.str.replace('%',''))

e = myfile[['comissionados_por_servidor']]

myfile['comissionados_por_servidor'] = e.apply(lambda x: x.str.replace('#DIV/0!',''))





myfile['densidade_dem'] = pd.to_numeric(myfile['densidade_dem'])

myfile['area'] = pd.to_numeric(myfile['area'])

myfile['codigo_mun'] = pd.to_numeric(myfile['codigo_mun'])

myfile['ranking_igm'] = pd.to_numeric(myfile['ranking_igm'])

myfile['comissionados_por_servidor'] = pd.to_numeric(myfile['comissionados_por_servidor'])
a = myfile['regiao']

regioes = pd.get_dummies(a)

myfile = pd.concat([myfile, regioes], axis=1)
a = myfile['estado']

estado = pd.get_dummies(a)

myfile = pd.concat([myfile, estado], axis=1)
a = myfile['porte']

porte = pd.get_dummies(a)

myfile = pd.concat([myfile, porte], axis=1)
myfile.isnull().sum()
import matplotlib.pyplot as plt

plt.hist(myfile['nota_mat'])

plt.show()
notas = myfile[['nota_mat','NORTE','NORDESTE','SUL','SUDESTE','CENTRO-OESTE']]

normalized_notas=(notas-notas.mean())/notas.std()



from string import ascii_letters

import numpy as np

import seaborn as sns



# Compute the correlation matrix

corr = normalized_notas.corr()



sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns, annot = True)
notas = myfile[['nota_mat','Grande porte', 'Médio porte', 'Pequeno porte 1', 'Pequeno porte 2']]

normalized_notas=(notas-notas.mean())/notas.std()



from string import ascii_letters

import numpy as np

import seaborn as sns



# Compute the correlation matrix

corr = normalized_notas.corr()



sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns, annot = True)
notas = myfile[['nota_mat','AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS',

       'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC',

       'SE', 'SP', 'TO']]

normalized_notas=(notas-notas.mean())/notas.std()



from string import ascii_letters

import numpy as np

import seaborn as sns



# Compute the correlation matrix

corr = normalized_notas.corr()



f, ax = plt.subplots(figsize=(16, 13))

sns.heatmap(corr,annot=True,annot_kws={"size":8})
corr = myfile[['nota_mat','exp_vida','populacao','pib','pib_pc','taxa_empreendedorismo','anos_estudo_empreendedor',

              'indice_governanca','idhm','gasto_pc_educacao','hab_p_medico','exp_anos_estudo','gasto_pc_saude',

              'jornada_trabalho','perc_pop_econ_ativa','comissionados','servidores','participacao_transf_receita',

              'densidade_dem', 'area', 'codigo_mun','ranking_igm', 'comissionados_por_servidor']]

cor_normalize =(corr-corr.mean())/corr.std()



from string import ascii_letters

import numpy as np

import seaborn as sns



# Compute the correlation matrix

corr = cor_normalize.corr()



f, ax = plt.subplots(figsize=(14, 11))

sns.heatmap(corr,annot=True,annot_kws={"size":8})
train_df = myfile[['nota_mat','exp_vida',

        'anos_estudo_empreendedor','indice_governanca','idhm', 'perc_pop_econ_ativa',

        'ranking_igm']]



cor_normalize =(train_df-train_df.mean())/train_df.std()



from string import ascii_letters

import numpy as np

import seaborn as sns



# Compute the correlation matrix

corr = cor_normalize.corr()



f, ax = plt.subplots(figsize=(14, 11))

sns.heatmap(corr,annot=True,annot_kws={"size":8})
train_df = myfile[['nota_mat','exp_vida',

        'anos_estudo_empreendedor','idhm', 'perc_pop_econ_ativa']]

train_df.dropna(inplace=True)



#create a dataframe with all training data except the target column

train_X = train_df.drop(columns=['nota_mat'])



#check that the target variable has been removed

train_X.head()
#create a dataframe with only the target column

train_y = train_df[['nota_mat']]



#view dataframe

train_y.head()
from keras.models import Sequential

from keras.layers import Dense

#create model

model = Sequential()



#get number of columns in training data

n_cols = train_X.shape[1]



#add model layers

model.add(Dense(10, activation='relu', input_shape=(n_cols,)))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))
#compile model using mse as a measure of model performance

model.compile(optimizer='adam', loss='mean_squared_error')
from keras.callbacks import EarlyStopping

#set early stopping monitor so the model stops training when it won't improve anymore

early_stopping_monitor = EarlyStopping(patience=3)

#train model

model.fit(train_X, train_y, validation_split=0.1, epochs=30, callbacks=[early_stopping_monitor])
#example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called 'test_X').

test_y_predictions = model.predict(test_X)
test_y_predictions = pd.DataFrame(test_y_predictions)
test_X = pd.read_csv("../input/test.csv")

b = test_X[['codigo_mun']]

b['codigo_mun'] = b.apply(lambda x: x.str.replace('ID_ID_',''))

b['codigo_mun'] = (b['codigo_mun']).astype(int)

a = pd.concat([b, test_y_predictions], axis=1)

a.columns = ['codigo_mun', 'nota_mat']
a.to_csv('Maria_de_Fatima.csv', index = False)
a