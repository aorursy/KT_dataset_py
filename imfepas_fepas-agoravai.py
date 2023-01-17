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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

valid = pd.read_csv('../input/valid.csv')
list(train.columns.values)
train = train.drop(['neighborhood', 'building_class_at_present','block', 'lot', 'ease-ment', 'address', 'apartment_number', 'zip_code', 'building_class_category', 'tax_class_at_time_of_sale', 'tax_class_at_present', 'building_class_at_time_of_sale' ],axis=1)

test = test.drop(['neighborhood', 'building_class_at_present','block', 'lot', 'ease-ment', 'address', 'apartment_number', 'zip_code', 'building_class_category', 'tax_class_at_time_of_sale','tax_class_at_present',  'building_class_at_time_of_sale' ],axis=1)

valid = valid.drop(['neighborhood', 'building_class_at_present','block', 'lot', 'ease-ment', 'address', 'apartment_number', 'zip_code', 'building_class_category', 'tax_class_at_time_of_sale','tax_class_at_present',  'building_class_at_time_of_sale' ],axis=1)
train.shape
test.shape
valid.shape
list(valid.columns.values)
#BOROUGH: Código sobre onde a propriedade é localizada: Manhattan (1), Bronx (2), Brooklyn (3), Queens (4) e Staten Island (5).



train['borough'][train['borough'] == 1] = 'manhattan'

train['borough'][train['borough'] == 2] = 'bronx'

train['borough'][train['borough'] == 3] = 'brooklyn'

train['borough'][train['borough'] == 4] = 'queens'

train['borough'][train['borough'] == 5] = 'staten island'



test['borough'][test['borough'] == 1] = 'manhattan'

test['borough'][test['borough'] == 2] = 'bronx'

test['borough'][test['borough'] == 3] = 'brooklyn'

test['borough'][test['borough'] == 4] = 'queens'

test['borough'][test['borough'] == 5] = 'staten island'



valid['borough'][valid['borough'] == 1] = 'manhattan'

valid['borough'][valid['borough'] == 2] = 'bronx'

valid['borough'][valid['borough'] == 3] = 'brooklyn'

valid['borough'][valid['borough'] == 4] = 'queens'

valid['borough'][valid['borough'] == 5] = 'staten island'
test.head(10)
test.dtypes
#As metricas de metros quadrados devem ser numéricas



test['land_square_feet'] = pd.to_numeric(test['land_square_feet'], errors='coerce')

test['gross_square_feet']= pd.to_numeric(test['gross_square_feet'], errors='coerce')



train['land_square_feet'] = pd.to_numeric(train['land_square_feet'], errors='coerce')

train['gross_square_feet']= pd.to_numeric(train['gross_square_feet'], errors='coerce')



valid['land_square_feet'] = pd.to_numeric(valid['land_square_feet'], errors='coerce')

valid['gross_square_feet']= pd.to_numeric(valid['gross_square_feet'], errors='coerce')



train.head(100)
train.dtypes
#testando se há dados duplicados

sum(train.duplicated(train.columns))
variables = train.columns



count = []



for variable in variables:

    length = train[variable].count()

    count.append(length)

    

count_pct = np.round(100 * pd.Series(count) / len(train), 2)
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(20,6))

plt.bar(variables, count_pct)

plt.title('Numero de dados em relação ao número de colunas', fontsize=15)

plt.show()
train.describe()
train['land_square_feet'].fillna(train['land_square_feet'].mode()[0], inplace=True)

train['gross_square_feet'].fillna(train['gross_square_feet'].mode()[0], inplace=True)



test['land_square_feet'].fillna(test['land_square_feet'].mode()[0], inplace=True)

test['gross_square_feet'].fillna(test['gross_square_feet'].mode()[0], inplace=True)



valid['land_square_feet'].fillna(valid['land_square_feet'].mode()[0], inplace=True)

valid['gross_square_feet'].fillna(valid['gross_square_feet'].mode()[0], inplace=True)



#Existem propriedades com 0 como ano de construção, então irei retirar esses dados do treino

train['year_built'].fillna(train['year_built'].mode()[0], inplace=True)

test['year_built'].fillna(test['year_built'].mode()[0], inplace=True)

valid['year_built'].fillna(valid['year_built'].mode()[0], inplace=True)



#Transformando a variavel de ano de construção em idade.

train['building_age'] = 2019 - train['year_built']

test['building_age'] = 2019 - test['year_built']

valid['building_age'] = 2019 - valid['year_built']



#Transformando data para datetime

train['sale_date'] = pd.to_datetime(train['sale_date'])

test['sale_date'] = pd.to_datetime(test['sale_date'])

valid['sale_date'] = pd.to_datetime(valid['sale_date'])



#Criando tabela sale_year

train['sale_year'] = train['sale_date'].dt.strftime('%Y')

test['sale_year'] = test['sale_date'].dt.strftime('%Y')

valid['sale_year'] = valid['sale_date'].dt.strftime('%Y')



train['sale_year'] = train['sale_year'].astype('int64')

test['sale_year'] = test['sale_year'].astype('int64')

valid['sale_year'] = valid['sale_year'].astype('int64')



#Criando tabela sale_age

train['sale_age'] = 2019 - train['sale_year']

test['sale_age'] = 2019 - test['sale_year']

valid['sale_age'] = 2019 - valid['sale_year']





train = train.drop(['year_built'],axis=1)

test = test.drop(['year_built'],axis=1)

valid = valid.drop(['year_built'],axis=1)



train = train.drop(['sale_year'],axis=1)

test = test.drop(['sale_year'],axis=1)

valid = valid.drop(['sale_year'],axis=1)



train = train.drop(['sale_date'],axis=1)

test = test.drop(['sale_date'],axis=1)

valid = valid.drop(['sale_date'],axis=1)
train.describe()
train.dtypes
train.head(50)
train_encoded = pd.get_dummies(train['borough'])

test_encoded = pd.get_dummies(test['borough'])

valid_encoded = pd.get_dummies(valid['borough'])
train = pd.concat([train, train_encoded], axis=1)

test = pd.concat([test, test_encoded], axis=1)

valid = pd.concat([valid, valid_encoded], axis=1)
train = train.drop(['borough'], axis=1)

test = test.drop(['borough'], axis=1)

valid = valid.drop(['borough'], axis=1)
train.head(5)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

train_1, train_2 = train_test_split(train, test_size=0.5)
list(train.columns.values)
train_1_y=train_1.sale_price

train_2_y=train_1.sale_price



predictor_cols=['residential_units',

 'commercial_units','land_square_feet','gross_square_feet', 'building_age', 'bronx', 'brooklyn', 'manhattan', 'queens', 'staten island' ]



train_1_x=train_1[predictor_cols]

train_2_x=train_2[predictor_cols]



model=RandomForestRegressor(n_jobs=-1, n_estimators= 100,  min_samples_split=2000,)



model.fit(train_1_x,train_1_y)

model.score(train_1_x,train_1_y)
model.fit(train_1_x,train_1_y)

model.score(train_2_x,train_2_y)
model.fit(train_2_x,train_2_y)

model.score(train_1_x,train_1_y)
train_final = train

train_final_y=train_final.sale_price

train_final_x=train_final[predictor_cols]



test_final_x_with_id = pd.concat([valid, test])



test_final_x = test_final_x_with_id[predictor_cols]



test_final_x_with_id.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

linreg = LinearRegression()





linreg.fit(train_final_x,train_final_y)



test_final_y = linreg.predict(test_final_x)





my_submission=pd.DataFrame({'sale_id': test_final_x_with_id['sale_id'], 'sale_price': test_final_y})

my_submission.to_csv('fepas_submission.csv',index=False)

my_submission.head(25)
# from sklearn.model_selection import RandomizedSearchCV

# # Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

# max_depth.append(None)

# # Minimum number of samples required to split a node

# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]

# # Method of selecting samples for training each tree

# bootstrap = [True, False]

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}

# print(random_grid)
# # Use the random grid to search for best hyperparameters

# # First create the base model to tune

# rf = RandomForestRegressor()

# # Random search of parameters, using 3 fold cross validation, 

# # search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model

# rf_random.fit(train_1_x,train_1_y)
# rf_random.best_params_
train.shape
test.shape

valid.shape