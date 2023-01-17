# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

X_test_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')



#print(len(X))

# 1460



X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X['SalePrice']

X.drop(columns=['SalePrice'], inplace=True)



from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, 

									train_size=0.8, test_size=0.2,

									random_state = 0)

low_cardinality_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]
print("Total columns in dataset: {}\nColumns I'll use: {}".

      format(len(X.columns), len(low_cardinality_cols + numerical_cols)))
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder



# Num

mean_imp = SimpleImputer(strategy='mean')



# Cat

mf_imp = SimpleImputer(strategy='most_frequent')

oh_enc = OneHotEncoder(handle_unknown='ignore',sparse=False)

cat_pipe = Pipeline(steps=[

    ('impute', mf_imp), ('one hot', oh_enc)

])



# Common

preprocess = ColumnTransformer(transformers=[

    ('num',mean_imp, numerical_cols), ('cat', cat_pipe, low_cardinality_cols)

])
# Transform



data_full = pd.concat([X_train_full, X_valid_full, X_test_full])[numerical_cols + low_cardinality_cols]

preprocess.fit(data_full)

X_train =  preprocess.transform(X_train_full[numerical_cols + low_cardinality_cols])

X_valid =  preprocess.transform(X_valid_full[numerical_cols + low_cardinality_cols])

X_test =  preprocess.transform(X_test_full[numerical_cols + low_cardinality_cols])
from xgboost import XGBRegressor



xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05)

xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])
validation_prediction = xgb_model.predict(X_valid)

print(mean_absolute_error(validation_prediction,y_valid))
preds_test_xgb = xgb_model.predict(X_test)



output_nn = pd.DataFrame({'Id': X_test_full['Id'],

                       'SalePrice': preds_test_xgb.reshape(-1)})

output_nn.to_csv('xgb_submission.csv', index=False)
from tensorflow.keras.layers import Input, Dense, Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam



entry = Input(X_train.shape[1])

nn = Dense(32, activation='relu')(entry)

nn = Dense(16, activation='relu')(entry)

nn = Dense(8, activation='relu')(nn)

#nn = Dropout(0.1)(nn)

nn = Dense(1, activation='linear')(nn)



nn = Model(inputs=entry, outputs=nn)

opt = Adam(lr=0.5, beta_1=0.9, beta_2=0.999, decay=0.01)

nn.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
history = nn.fit(X_train, y_train, epochs=150, batch_size= 32, verbose=1, validation_data=(X_valid, y_valid))
nn_validation_pred = nn.predict(X_valid)

print(mean_absolute_error(nn_validation_pred, y_valid))
# Plot loss

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
preds_test_nn = nn.predict(X_test)
output_nn = pd.DataFrame({'Id': X_test_full['Id'],

                       'SalePrice': preds_test_nn.reshape(-1)})

output_nn.to_csv('nn_submission.csv', index=False)
from lightgbm import LGBMRegressor
lgbm_model = LGBMRegressor(max_depth=-1,

                               n_estimators=500,

                               learning_rate=0.1,

                               colsample_bytree=0.2,

                               objective='regression', 

                               n_jobs=-1)

                               

lgbm_model.fit(X_train, y_train, eval_metric='mae', eval_set=[(X_valid, y_valid)], 

              verbose=100, early_stopping_rounds=100)
lgbm_preds = lgbm_model.predict(X_test)

output_lgbm = pd.DataFrame({'Id': X_test_full['Id'],

                       'SalePrice': lgbm_preds.reshape(-1)})

output_lgbm.to_csv('lgbm_submission.csv', index=False)
output_lgbm.head()