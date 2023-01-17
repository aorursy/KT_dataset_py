import numpy as np

import pandas

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn import preprocessing
def get_train_data():

    # load dataset

    dataset_train = pandas.read_csv("../input/train.csv")

    dataset_test = pandas.read_csv("../input/test.csv")

    #['id' 'molecule_name' 'atom_index_0' 'atom_index_1' 'type' 'scalar_coupling_constant']

    cat_columns =['molecule_name', 'type']

    label_encoders = {}

    for col in cat_columns:

        new_le = LabelEncoder()

        dataset_train[col] = new_le.fit_transform(dataset_train[col])

        dataset_test[col] = new_le.fit_transform(dataset_test[col])

    X_train = pandas.DataFrame(dataset_train, columns=['molecule_name','atom_index_0','atom_index_1','type'])

    X_test = pandas.DataFrame(dataset_test, columns=['id','molecule_name','atom_index_0','atom_index_1','type'])

    Y_train = dataset_train['scalar_coupling_constant']

    return X_train,X_test,Y_train



min_max_scaler = preprocessing.MinMaxScaler()

X_train,X_test_with_id,Y_train = get_train_data()

X_train = min_max_scaler.fit_transform(X_train)

X_test = pandas.DataFrame(X_test_with_id, columns=['molecule_name','atom_index_0','atom_index_1','type'])

X_test = min_max_scaler.fit_transform(X_test)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

print(x_train[0:5])

print(x_test[0:5])
# Regression

'''

from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  

regressor.fit(x_train, y_train) 



y_pred = regressor.predict(x_test)

print(regressor.score(x_test,y_test))

y_out = y_pred = regressor.predict(X_test)

'''
## Random_forest

'''

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(x_train, y_train)

print(forest_model.score(x_test,y_test))

melb_preds = forest_model.predict(x_test)

y_out = forest_model.predict(X_test)

'''
'''

from sklearn.ensemble import GradientBoostingRegressor

model=GradientBoostingRegressor()

model= model.fit(x_train,y_train)

print(model.score(x_test,y_test))

y_out = model.predict(X_test)

'''
#Parameters for LightGBM Model

params_lgb = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'regression',

          'max_depth': 9,

          'learning_rate': 0.1,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          "bagging_seed": 47,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1302650970728192,

          'reg_lambda': 0.3603427518866501,

          'colsample_bytree': 1.0,

          'n_estimators':10000

         }





import lightgbm as lgb

lgtrain = lgb.Dataset(x_train, label=y_train)

lgval = lgb.Dataset(x_test, label=y_test)

model_lgb = lgb.train(params_lgb, lgtrain, 10000, 

                  valid_sets=[lgtrain, lgval], 

                  verbose_eval=500)

y_out = model_lgb.predict(X_test)

'''

#CatBoost Model

from catboost import CatBoostRegressor



model_cat = CatBoostRegressor(iterations=15000,

                             learning_rate=0.03,

                             depth=2,

                             eval_metric='MAE',

                             random_seed = 47,

                             od_wait=5000)

model_cat.fit(x_train,y_train, verbose=500)

y_out = model_cat.predict(X_test)

'''
y_out = model_cat.predict(X_test)
my_submission = pandas.DataFrame({'id': X_test_with_id.id, 'scalar_coupling_constant': y_out})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)