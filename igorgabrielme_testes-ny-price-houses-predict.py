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
df_raw_train = pd.read_csv('../input/train.csv')
df_raw_train.head()
df_raw_train.describe()
df_raw_train.info()
df_raw_train.columns
# Convertendo land_square_feet e gross_square_feet para numerico

df_clean_train = df_raw_train

df_clean_train['land_square_feet'] = pd.to_numeric(df_clean_train['land_square_feet'], downcast='integer', errors='coerce')

df_clean_train['gross_square_feet'] = pd.to_numeric(df_clean_train['gross_square_feet'], downcast='integer', errors='coerce')

# df_clean_train.info()
# Contando a quantidade de celulas com valores nulos

df_clean_train.isna().sum()
print(df_clean_train['land_square_feet'].shape)

print(df_clean_train['gross_square_feet'].shape)
# Adicionando 0 nos valores nulos 

df_clean_train['land_square_feet'].fillna(0, inplace=True)

df_clean_train['gross_square_feet'].fillna(0, inplace=True)

# df_clean_train['land_square_feet'].fillna(df_clean_train['land_square_feet'].mean(), inplace=True)

# df_clean_train['gross_square_feet'].fillna(df_clean_train['gross_square_feet'].mean(), inplace=True)

# df_clean_train = df_clean_train[df_clean_train['land_square_feet'].notnull()] 

# df_clean_train = df_clean_train[df_clean_train['gross_square_feet'].notnull()] 
# print(df_clean_train['land_square_feet'].shape)

# print(df_clean_train['gross_square_feet'].shape)

df_clean_train.tax_class_at_present.unique()
df_clean_train.isna().sum()
# df_clean_train.describe()

# df_clean_train.dtypes
#escolhendo as colunas que serão utilizadas no treinamento

columns = ['borough', 'residential_units', 'commercial_units', 'total_units', 'land_square_feet', 'gross_square_feet', 

           'year_built', 'tax_class_at_present']



data_model = df_clean_train.loc[:,columns]
data_model
# Categorizando borough

one_hot_encoder = pd.get_dummies(data_model['borough'])

one_hot_encoder.info()
# Dropando a coluna borough

data_model=data_model.drop(['borough'], axis=1)



# Juntando as colunas borough categorizada

data_model = pd.concat([data_model, one_hot_encoder], axis=1)
# data_model.info()
# Categorizando tax_class_at_present

one_hot_tax_clss = pd.get_dummies(data_model['tax_class_at_present'])

one_hot_tax_clss.info()
# Dropando a coluna tax_class_at_present

data_model=data_model.drop(['tax_class_at_present'], axis=1)



# Juntando as colunas tax_class_at_present categorizada

data_model = pd.concat([data_model, one_hot_tax_clss], axis=1)
data_model.info()
x_train = data_model

y_train = df_clean_train.sale_price
x_train
# Regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier
# Treinando o modelo random forest

rf_reg = RandomForestRegressor()

rf_reg.fit(x_train, y_train)
# Treinando o modelo decisiontree 

# rf_reg = DecisionTreeClassifier(max_leaf_nodes=100)

# rf_reg.fit(x_train, y_train)
# Carregando os csv de validação e teste e juntando-os

df_valid = pd.read_csv('../input/valid.csv')

test = pd.read_csv('../input/test.csv')

df_test_valid = df_valid.append(test)
# coluna = ['borough', 'residential_units', 'commercial_units','total_units', 'land_square_feet', 'gross_square_feet', 'year_built']

df_test_valid.info()
#Convertendo para numerico

df_cle_test_valid = df_test_valid

df_cle_test_valid['land_square_feet'] = pd.to_numeric(df_cle_test_valid['land_square_feet'], downcast='integer', errors='coerce')

df_cle_test_valid['gross_square_feet'] = pd.to_numeric(df_cle_test_valid['gross_square_feet'], downcast='integer', errors='coerce')
# df_cle_test_valid['land_square_feet']

# df_cle_test_valid['gross_square_feet']

df_cle_test_valid.info()
# Colocando o valor zero onde os valores são nulos

df_cle_test_valid['land_square_feet'].fillna(0, inplace=True)

df_cle_test_valid['gross_square_feet'].fillna(0, inplace=True)

# df_cle_test_valid['land_square_feet'].fillna(df_cle_test_valid['land_square_feet'].mean(), inplace=True)

# df_cle_test_valid['gross_square_feet'].fillna(df_cle_test_valid['gross_square_feet'].mean(), inplace=True)

# df_cle_test_valid['gross_square_feet']
valid_x = df_cle_test_valid[columns]



#Categorizando borough

one_hot_encoder_val = pd.get_dummies(valid_x['borough'])

one_hot_encoder_val.info()
valid_x
# valid_x.tax_class_at_present.unique()
#Dropando borough

valid_x = valid_x.drop(['borough'], axis=1)



#Adicionando borough categorizado

valid_x = pd.concat([valid_x, one_hot_encoder_val], axis=1)

# valid_x.describe()
# Categorizando tax class

one_hot_bd_clss_val = pd.get_dummies(valid_x['tax_class_at_present'])

one_hot_bd_clss_val.info()
#Dropando tax class

valid_x = valid_x.drop(['tax_class_at_present'], axis=1)



# Adicionando tax class categorizado

valid_x = pd.concat([valid_x, one_hot_bd_clss_val], axis=1)
# valid_x.info()
#Predizendo os dados



# valid_x = df_cle_test_valid[coluna]



predicted_prices = rf_reg.predict(valid_x)



print(predicted_prices)
# Gerando o csv para envio

submission = pd.DataFrame({'sale_id': df_test_valid.sale_id, 'sale_price': predicted_prices})

submission.to_csv('submission.csv', index=False)
# print("R^2: {}".format(rf_reg.score(X_test_s, y_test_s)))