import numpy as np 

import pandas as pd

from matplotlib import pyplot as pp

%matplotlib inline

import itertools

from sklearn.preprocessing import OneHotEncoder

from category_encoders import CountEncoder, TargetEncoder, CatBoostEncoder

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split 

from sklearn.feature_selection import SelectFromModel

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/financial-inclusion-data/323388_649935_compressed_Train_v2.csv/Train_v2.csv')



train_data.head()
train_data.drop(['year'], axis = 1, inplace = True)

train_data.head()
def change_series(data):

    listy = []

    for i in data['bank_account']:

        if i == 'Yes':

            listy.append(1)

        else:

            listy.append(0)

            

    return listy



train_data['bank_account'] = change_series(train_data)
train_data.isna().sum()
len(train_data)
train_data.dtypes
pp.hist((train_data['household_size']), range = (0,(train_data['household_size'].max())))
pp.hist(train_data['age_of_respondent'], range = (0,train_data['age_of_respondent'].max()))
y = train_data['bank_account']

train_data.drop(['bank_account'], axis =1 , inplace = True)



interactions = list(itertools.combinations(train_data.columns,2))



for i in interactions:

    train_data[str(i[0])+'_'+ str(i[1])] = train_data[i[0]].astype(str) + '_' + train_data[i[1]].astype(str)



categorical_columns = [col for col in train_data.columns if train_data[col].dtypes == 'object']

numerical_columns = [col for col in train_data.columns if train_data[col].dtypes == 'int64' or train_data[col].dtypes == 'float64']
categorical_columns 
numerical_columns
#valid_fraction = 0.8

#val = valid_fraction*(len(train_data)+1)

#X_train = train_data[:int(val)]

#X_val = train_data[int(val):]



X_train, X_val, y_train, y_val = train_test_split(train_data, y, train_size = 0.8, test_size = 0.2, 

                                                  random_state = 0)
#encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)



#X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_columns]))

#X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_columns]))



#X_train_encoded.index = X_train.index

#X_val_encoded.index = X_val.index



#num_train = X_train.drop(categorical_columns, axis = 1)

#num_test = X_val.drop(categorical_columns, axis =1)



#final_train = pd.concat([X_train_encoded, num_train], axis = 1)

#final_test = pd.concat([X_val_encoded, num_test], axis = 1)

encoder = CountEncoder(cols = categorical_columns)



X_train_encoded = encoder.fit_transform(X_train[categorical_columns])

X_val_encoded = encoder.transform(X_val[categorical_columns])



num_train = X_train[numerical_columns]

num_val = X_val[numerical_columns]



final_train = pd.concat([X_train_encoded, num_train],axis = 1)

final_val = pd.concat([X_val_encoded, num_val], axis = 1)
#model = RandomForestClassifier(random_state = 0)

#model.fit(final_train, y_train)

#preds = model.predict(final_val)





#model = XGBClassifier()

#model.fit(final_train, y_train)

#preds = model.predict(final_val)





model = LGBMClassifier(random_state = 0, objective = 'binary', n_jobs = 4)

model.fit(final_train, y_train)
first_score = cross_val_score(model, final_val, y_val, cv = 5, scoring = 'roc_auc')

first_score.mean()
#selection = SelectFromModel(model, threshold = 0.15)

for i in zip(train_data.columns, model.feature_importances_):

    print(i)
model.get_params()
param_grid = {

    'max_depth': [20,25],

    'n_estimators': [700,1000],

    'learning_rate': [0.01],

    'num_leaves': [100, 200],

    'colsample_bytree': [0.7, 0.8],

    'reg_alpha': [1.1, 1.2, 1.3],

    'reg_lambda': [1.1, 1.2, 1.3],

    'min_split_gain': [0.3, 0.4],

    'subsample': [0.7, 0.8, 0.9],

    'subsample_freq': [20]

}





from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



search = RandomizedSearchCV(model, param_grid, cv = 5, scoring = 'roc_auc', n_jobs = 4)



search.fit(final_train, y_train)
search.best_params_
final_model = LGBMClassifier(random_state = 0, objective = 'binary', n_jobs = 4, 

                            learning_rate = 0.01, n_estimators = 700, num_leaves = 100, 

                             max_depth = 25, subsample_freq = 20, subsample = 0.8, reg_lambda = 1.3,

                            reg_alpha = 1.1, mni_split_gain = 0.4, colsamaple_bytree = 0.7)

final_model.fit(final_train, y_train)
final_score = cross_val_score(final_model, final_val, y_val, cv = 5, scoring = 'roc_auc')

final_score.mean()