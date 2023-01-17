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

train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col = 'Id')
test_data  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col = 'Id')

train_data.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis = 1, inplace = True)

X_train_full, X_val_full, y_train, y_val = train_test_split(train_data,
                                                 y,
                                                 train_size = 0.8,
                                                 test_size = 0.2,
                                                 random_state = 0)
X_train_full.shape, X_val_full.shape
X_train_full.head(5)
categorical_cols = [cname for cname in X_train_full.columns if
                   X_train_full[cname].nunique() < 10 and
                   X_train_full[cname].dtype == 'object']

numerical_cols = [cname for cname in X_train_full.columns if
                 X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[numerical_cols].copy()
X_val = X_val_full[numerical_cols].copy()
X_test = test_data[numerical_cols].copy()
X_train.head()
X_train.shape, X_val.shape
mean = X_train.mean(axis = 0)
X_train -= mean
std = X_train.std(axis = 0)
X_train /= std

X_val -= mean
X_val /= std

X_test -= mean
X_test /= std


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(X_train)
X_train = imp.transform(X_train)
X_val = imp.transform(X_val)
X_test = imp.transform(X_test)

import tensorflow as tf
from keras import models
from keras import layers

def build_model():
    model = models.Sequential([
        layers.Dense(512, activation = 'relu', input_shape = (X_train.shape[1],), ),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 0.000001), loss = 'mse',
                 metrics = ['mae'])
    return model
model = build_model()
model.fit(X_train, 
          y_train,
         epochs = 100,
          batch_size = 128,
         validation_data = (X_val, y_val))
pred = model.predict(X_test)
pred = pred.reshape(pred.shape[0],)
output = pd.DataFrame({'Id': test_data.index,
                       'SalePrice': pred})
output.to_csv('submission.csv', index=False)