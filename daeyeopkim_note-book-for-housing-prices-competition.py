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
from sklearn.model_selection import train_test_split



data_file_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

data = pd.read_csv(data_file_path, index_col='Id')



#features=['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','GarageYrBlt','GarageArea']

#X = data[features]

y = data.SalePrice

data.drop(['SalePrice'], axis=1 , inplace = True)

X = data.select_dtypes(exclude = ['object'])



train_X, val_X, train_y, val_y = train_test_split(X,y,train_size=0.8, test_size = 0.2)
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]



dropped_train_X = train_X.drop(cols_with_missing, axis=1)

dropped_val_X = val_X.drop(cols_with_missing, axis=1)

                          

print(dropped_train_X.shape)

print(train_X.shape)

from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()



imputed_train_X= pd.DataFrame(my_imputer.fit_transform(train_X))

imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))



imputed_train_X.columns = train_X.columns

imputed_val_X.columns = val_X.columns



print(imputed_train_X.shape)
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



def get_mae(train_X,train_y,val_X,val_y,n_estimators=10):

    model = RandomForestRegressor(random_state=0, n_estimators = n_estimators)

    model.fit(train_X,train_y)

    pred = model.predict(val_X)

    mae = mean_absolute_error(val_y, pred)

    return mae
import matplotlib.pyplot as plt



def get_best_size(train_X,train_y,val_X,val_y):

    mae_list = []

    scales = np.linspace(10,100,91,dtype="int16")

    for i in scales:

        mae = get_mae(train_X,train_y,val_X,val_y,n_estimators=i)

        mae_list.append(mae)

    min_mae = min(mae_list)

    best_size = scales[mae_list.index(min_mae)]

    print(best_size)

    plt.plot(mae_list)

    plt.show()

    return best_size

    
train_X_plus = train_X.copy()

val_X_plus = val_X.copy()



for col in cols_with_missing:

    train_X_plus[col+"was_missing"] = train_X_plus[col].isnull()

    val_X_plus[col+"was_missing"] = val_X_plus[col].isnull()

    

imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))

imputed_val_X_plus = pd.DataFrame(my_imputer.transform(val_X_plus))



print(imputed_train_X_plus.shape)
best_size = get_best_size(dropped_train_X, train_y , dropped_val_X , val_y)
mae_droped = get_mae(dropped_train_X, train_y , dropped_val_X , val_y ,best_size)

mae_imputed = get_mae(imputed_train_X , train_y , imputed_val_X , val_y,best_size)

mae_extension = get_mae(imputed_train_X_plus , train_y , imputed_val_X_plus , val_y,best_size)



print(f"drop :{mae_droped} \nimputation :{mae_imputed} \nextension : {mae_extension}")
test_file_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

test_data = pd.read_csv(test_file_path,index_col = 'Id').select_dtypes(exclude = ['object'])

test_data.drop(cols_with_missing, axis = 1 , inplace = True)

imputed_test_data = pd.DataFrame(my_imputer.fit_transform(test_data))



dropped_X = X.drop(cols_with_missing, axis = 1)





my_model = RandomForestRegressor(random_state = 1, n_estimators = 100)

my_model.fit(dropped_X, y)

pred = my_model.predict(imputed_test_data)

sample_file_path = '/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv'

sample = pd.read_csv(sample_file_path, index_col="Id")



output = pd.DataFrame({'Id': test_data.index,

                       'SalePrice': pred})

output.to_csv('submission.csv', index=False)


