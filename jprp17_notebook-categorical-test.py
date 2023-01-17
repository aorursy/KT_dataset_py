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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')
X.dropna(axis = 0, subset=['SalePrice'], inplace = True)

#X_2 não possui coluna SalePrice
#X_2.dropna(axis=0, subset=['SalePrice'], inplace = True)

y=X['SalePrice']
X.drop(['SalePrice'],axis=1,inplace = True)

cols_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_missing, axis = 1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def scores_mae(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators = 100, random_state=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_absolute_error(y_test,pred)
#remove columns with categorical data
remove_X_train = X_train.select_dtypes(exclude=['object'])
remove_X_test = X_test.select_dtypes(exclude=['object'])
print('MAE para remoção das colunas com valores categóricos:')
print(scores_mae(remove_X_train, remove_X_test, y_train, y_test))
#All categorical columns
object_columns = [col for col in X_train.columns if X_train[col].dtype =='object']
good_labels = [col for col in object_columns if set(X_train[col]) == set(X_test[col])]
bad_labels = list(set(object_columns) - set(good_labels))
print('Categorical columns que serão utilizadas: \n', good_labels,'\n')
print('Categorical columns que não serão utilizadas: \n', bad_labels,'\n')
#LabelEncoder
from sklearn.preprocessing import LabelEncoder

label_X_train = X_train.drop(bad_labels, axis = 1)
label_X_test = X_test.drop(bad_labels, axis = 1)

label_encode = LabelEncoder()
for col in good_labels:
    label_X_train[col] = label_encode.fit_transform(X_train[col])
    label_X_test[col] = label_encode.transform(X_test[col])
        
print('MAE para remoção das colunas com valores categóricos, usando método LabelEncoder:')
print(scores_mae(label_X_train, label_X_test, y_train, y_test))
#Lista de quantas variáveis categoricas existem em cada uma das colunas
object_nunique = list(map(lambda col: X_train[col].nunique(), object_columns))
d = dict(zip(object_columns, object_nunique))
sorted(d.items(),key=lambda x:x[1])
#One-hot encoding

low_cardinality_cols = [col for col in object_columns if X_train[col].nunique() < 10]
high_cardinality_cols = list(set(object_columns)-set(low_cardinality_cols))
print('Categorical columns que serão One-hot encoding: \n', low_cardinality_cols,'\n')
print('Categorical columns que não serão One-hot encoding: \n', high_cardinality_cols,'\n')
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols])) 
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))

#O método HotEncode retira os indices, adicionando-os novemante
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

#removendo categoricals columns
num_X_train= X_train.drop(object_columns, axis=1)
num_X_test= X_test.drop(object_columns, axis =1)

#adicionando as HotEncode columns
OH_X_train = pd.concat([num_X_train, OH_cols_train],axis =1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis =1)

print('MAE para remoção das colunas com valores categóricos, usando método HotEncoder:')
print(scores_mae(OH_X_train, OH_X_test, y_train, y_test))
#Full Test

rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(OH_X_train,y_train)

X_2 = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')
cols_missing = [col for col in X.columns if X[col].isnull().any()]
X_2.drop(cols_missing, axis = 1, inplace = True)
#All categorical columns
object_columns = [col for col in X_train.columns if X_train[col].dtype =='object']
good_label = [col for col in object_columns if set(X_train[col]) == set(X_2[col])]
bad_label = list(set(object_columns) - set(good_label))
print('Categorical columns que serão utilizadas: \n', good_labels,'\n')
print('Categorical columns que não serão utilizadas: \n', bad_labels,'\n')
full_model = RandomForestRegressor(n_estimators = 100, random_state=1)
full_model.fit(OH_X_train,y_train)
pred = full_model.predict(full_test)
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)