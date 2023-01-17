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
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')
len(test)
train_Y = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder



label_X_train = train.copy()

label_X_test = test.copy()

labelencoder=LabelEncoder()
s= (train.dtypes=='object')

object_cols=list(s[s].index)

print("categorical Columns")

print(object_cols)
print(s)
# drop_X_train=label_X_train.select_dtypes(exclude='object')

# drop_X_test=label_X_test.select_dtypes(exclude='object')

# numerical_columns=list(drop_X_train.columns.values)
# from sklearn.ensemble import RandomForestRegressor

# from sklearn.metrics import mean_absolute_error

# model = RandomForestRegressor(n_estimators=100, random_state=0)

# model.fit(drop_X_train, train_Y)

# preds = model.predict(drop_X_test)
# for col in object_cols:

#     label_X_train[col] = label_encoder.fit_transform(X_train[col])

#     label_X_valid[col] = label_encoder.transform(X_valid[col])
# ONCE HOT ENCODING
# from sklearn.compose import ColumnTransformer

# from sklearn.pipeline import Pipeline

# from sklearn.impute import SimpleImputer

# from sklearn.preprocessing import OneHotEncoder



# # Preprocessing for numerical data

# numerical_transformer = SimpleImputer(strategy='constant')



# # Preprocessing for categorical data

# categorical_transformer = Pipeline(steps=[

#     ('imputer', SimpleImputer(strategy='most_frequent')),

#     ('labelencoder', LabelEncoder())

# ])



# # Bundle preprocessing for numerical and categorical data

# preprocessor = ColumnTransformer(

#     transformers=[

#         ('num', numerical_transformer, numerical_columns),

#         ('cat', categorical_transformer, object_cols)

#     ])
# model = RandomForestRegressor(n_estimators=100, random_state=0)

# clf = Pipeline(steps=[('preprocessor', preprocessor),

#                       ('model', model)

#                      ])
# clf.fit(label_X_train, train_Y)
import category_encoders as ce

target_enc = ce.TargetEncoder(cols=object_cols)

target_enc.fit(label_X_train[object_cols],  train_Y)

label_X_train.iloc[:,7:16]
train_tar = label_X_train.join(target_enc.transform(label_X_train[object_cols]).add_suffix('_target'))

test_tar = label_X_test.join(target_enc.transform(label_X_test[object_cols]).add_suffix('_target'))
train_tar.head()
count_enc = ce.CountEncoder(cols=object_cols)
count_enc.fit(label_X_train[object_cols])
train_tar_count = train_tar.join(count_enc.transform(label_X_train[object_cols]).add_suffix('_count'))

test_tar_count = test_tar.join(count_enc.transform(label_X_test[object_cols]).add_suffix('_count'))
train_tar_count.columns
# need to join both target and count 

# categorical column changing for label encoding 

# remove the five columns and then encode original columns and then fit with random forest regressor 

train_dropped_five=train.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1)

test_dropped_five = test.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1)

train.head()

object_cols
object_cols.remove('nom_5')

object_cols.remove('nom_6')

object_cols.remove('nom_7')

object_cols.remove('nom_8')

object_cols.remove('nom_9')

print("hello",object_cols)
for col in object_cols:

    label_X_train[col]=labelencoder.fit_transform(train_dropped_five[col])

    label_X_test[col] = labelencoder.fit_transform(test_dropped_five[col])

    #label_X_test[col]=labelencoder.transform(test[col])
label_X_train.head()
len(label_X_test_deleted)
label_X_train_deleted=label_X_train.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1)

label_X_test_deleted=label_X_test.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1)

label_X_train_deleted.head()
# join label_X_train_deleted and train_tar_count  .Remove original columns in train_tar_count

train_tar_count.columns
label_X_train_deleted.columns
train_tar_count_dropped=train_tar_count.drop(['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2',

                     'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0',

                     'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month'],axis=1)

test_tar_count_dropped=test_tar_count.drop(['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2',

                     'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0',

                     'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month'],axis=1)

       

       
len(label_X_test_deleted),len(test_tar_count_dropped)
train_consolidated=label_X_train_deleted.join(train_tar_count_dropped)

test_consolidated=label_X_test_deleted.join(test_tar_count_dropped)
from sklearn.ensemble import RandomForestRegressor
train_consolidated.columns
# model = RandomForestRegressor(n_estimators=30,learning_rate=0.3,n_job=2, random_state=0)

# model.fit(train_consolidated, train_Y)



#label_X_test,learning_rate=0.3,n_job=2
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=510,learning_rate=0.05,n_job=2, random_state=0)

model.fit(train_consolidated, train_Y)
len(test_consolidated)




from sklearn.impute import SimpleImputer



# Fill in the lines below: imputation



imputer=  SimpleImputer(strategy='median')

test_cons =pd.DataFrame(imputer.fit_transform(test_consolidated))
test_cons.columns = test_consolidated.columns
#test_cons=test_consolidated.fillna(0)
test_cons.head()
len(test_consolidated)
preds = model.predict(test_cons)
preds
pd.DataFrame({"id": test_id, "target": preds}).to_csv("submission.csv", index=False)