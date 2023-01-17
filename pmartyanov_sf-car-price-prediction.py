import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pp

import sys



from tqdm.notebook import tqdm

from catboost import CatBoostRegressor



import ast

import itertools

from itertools import combinations



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MultiLabelBinarizer



from datetime import datetime

from datetime import date



import unicodedata



from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, StratifiedKFold, KFold





from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor 

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, AdaBoostRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.base import clone

from pprint import pprint



from collections import Counter



from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
VERSION    = 12

DIR_TRAIN  = '../input/auto-ru-parsed/'

DIR_TEST = '../input/cpptest/'

DIR_SAMPLE = '../input/cppsample/'
RANDOM_SEED = 42 #42

VAL_SIZE   = 0.33   #0.33

N_FOLDS    = 10 #10



# CATBOOST

ITERATIONS = 2000 #4000

LR         = 0.05 #0.05
pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 50)
train = pd.read_csv(DIR_TRAIN+'train.csv')

test = pd.read_csv(DIR_TEST+'test.csv')

sample_submission = pd.read_csv(DIR_SAMPLE+'sample_submission.csv')
train.drop(['url'], axis=1, inplace=True)

train.columns = ['bodyType', 'brand', 'color', 'fuelType', 'modelDate', 'name',

       'numberOfDoors', 'productionDate', 'vehicleConfiguration',

       'vehicleTransmission', 'engineDisplacement', 'enginePower',

       'description', 'mileage', 'Комплектация', 'Привод', 'Руль', 'Состояние',

       'Владельцы', 'ПТС', 'Таможня', 'Владение', 'price']
train.dropna(subset = ['Владельцы'], inplace=True)
display(train.head(3))

display(test.head(3))
train.describe(include = 'all').T
test.describe(include = 'all').T
pp.ProfileReport(train)
pp.ProfileReport(test.drop(['description'], axis=1))
def preproc_data(df_input):

    df_output = df_input.copy()

    

    df_output.drop(['Состояние', 'Таможня', 'Руль', 'brand'], axis=1, inplace=True,)

        

    if 'id' in df_input.columns:

        df_output.drop(['id'], axis=1, inplace=True)

        

#     if 'price' in df_input.columns:

#         df_output.drop(['price'], axis=1, inplace=True,)

    

    

    # ################### fix ############################################################## 

    # Переводим признаки из float в int (иначе catboost выдает ошибку)

    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate']:

        df_output[feature]=df_output[feature].astype('int32')

    



    # ################### Clean #################################################### 

    # убираем признаки, которые не могут быть использованы для модели

    df_output.drop(['Комплектация', 'description', 'Владение'], axis=1, inplace=True,)

    

    

    # меняем формат признаков мощности и объема двигателя

    df_output['enginePower'] = df_output['enginePower'].apply(lambda x: int(x.replace(' N12','')))

    df_output['engineDisplacement'] = df_output['engineDisplacement'].apply(\

                                    lambda x: int(float(x.replace(' LTR','').replace('undefined','2'))*1000))

    

    

    # убираем лишние признаки 

    df_output.drop(['vehicleConfiguration'], axis=1, inplace=True,)

#     df_output.drop(['vehicleConfiguration', 'modelDate', 'numberOfDoors'], axis=1, inplace=True,)

    

#     df_output.drop(['modelDate', 'numberOfDoors'], axis=1, inplace=True,)

    

#     df_output.dropna(inplace=True)

    return df_output
def get_fuel_type(row):

    if row == 'бензин':

        return ''

    elif row == 'дизель':

        return 'd'

    elif row == 'гибрид':

        return 'hyb'

    elif row == 'электро':

        return 'Electro'



def get_transmission(row):

    if row == 'автоматическая':

        return 'AT'

    elif row == 'механическая':

        return 'MT'

    elif row == 'роботизированная':

        return 'AMT'

    

def get_drive(row):

    if row == 'полный':

        return '4WD'

    else:

        return ''

    

def get_mod_name(row):

    words = row.split()

    

    if 'drive' in words[-1].lower():

        return words[-2] + ' ' + words[-1]

    else:

        return words[-1]
def get_models(df_input):

    df_output = df_input.copy()

    df_output = df_output[['fuelType', 'name', 'vehicleTransmission', 

                           'engineDisplacement', 'enginePower', 'Привод', 'modelDate']]

    

    df_output['fuelType'] = df_output['fuelType'].apply(get_fuel_type)

    df_output['vehicleTransmission'] = df_output['vehicleTransmission'].apply(get_transmission)

    df_output['engineDisplacement'] = df_output['engineDisplacement'].apply(lambda x: str(x.replace(' LTR','')))

    df_output['enginePower'] = df_output['enginePower'].apply(lambda x: str(x.replace(' N12','')))

    df_output['Привод'] = df_output['Привод'].apply(get_drive)

    df_output['mod_name'] = df_output['name'].apply(get_mod_name)

    df_output['modelDate'] = df_output['modelDate'].apply(lambda x: str(int(x)))

    

    df_output['short_name'] = df_output['modelDate'] + ' ' + df_output['mod_name'] + ' ' \

    + df_output['engineDisplacement'] + df_output['fuelType'] + ' ' \

    + df_output['vehicleTransmission'] + ' (' + df_output['enginePower'] + ' л.с.)' + ' ' \

    + df_output['Привод']

    

    df_output['short_name'] = df_output['short_name'].str.strip()

    

    df_output = df_output[['short_name','name']]

    df_output.drop_duplicates(subset='short_name',keep='first',inplace=True)

    df_output.columns = ['short_name', 'full_name']



    return df_output
names = get_models(train)

names
test['short_name'] = test.apply(lambda x: str(int(x['modelDate'])) + ' ' + x['name'], axis=1)
print(test.shape)

test = test.merge(names,on='short_name',how='left')

print(test.shape)
test['name'] = test.apply(lambda x: x['name'] 

                          if pd.isnull(x['full_name']) 

                          else x['full_name'], axis=1)
# Отдельные строки, для которых в тесте не было названий модификации

test['name'] = test['name'].apply(lambda x: 'BMW X5 M I (E70)' 

                                  if x == '4.4 AT (555 л.с.) 4WD' else x)

test['name'] = test['name'].apply(lambda x: 'BMW X5 M II (F85)' 

                                  if x == '4.4 AT (575 л.с.) 4WD' else x)

test['name'] = test['name'].apply(lambda x: 'BMW M5 V (F10)' 

                                  if x == '4.4 AMT (560 л.с.)' else x)

test['name'] = test['name'].apply(lambda x: 'BMW M3 IV (E90)' 

                                  if x == '4.0 AMT (420 л.с.)' else x)

test['name'] = test['name'].apply(lambda x: 'BMW M4 F82/F83' 

                                  if x == '3.0 AMT (431 л.с.)' else x)

test['name'] = test['name'].apply(lambda x: 'BMW M5 VI (F90)' 

                                  if x == '4.4 AT (600 л.с.) 4WD' else x)

test.drop(['full_name', 'short_name'], axis=1, inplace=True)
def get_model_series(row):

    words = row.split()

    

    # только серия

    if words[0] == 'BMW':

            

        if 'X' in words[1] or 'M' in words[1]:

            name = words[1]

        else:

            name = words[1] + ' ' + words[2]

            

        if words[-1] == 'xDrive':

            name = name + ' xDrive'

            

        return name

    

    else:

        return(row)
def roman_to_int(s):

    rom_val = {'I': 1, 'V': 5}

    

    int_val = 0

    for i in range(len(s)):

        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:

            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]

        else:

            int_val += rom_val[s[i]]

    return int_val
def get_generation(row):

    

    words = row.split()

    

    gen = 1

    

    if words[0] == 'BMW':

        for word in words:

            if word in ['I','II','III','IV','V','VI','VII']:

                gen = roman_to_int(word)

    

    

    return gen
train['gen'] = train['name'].apply(get_generation)

test['gen'] = test['name'].apply(get_generation)
test['series_name'] = test['name'].apply(get_model_series)

train['series_name'] = train['name'].apply(get_model_series)
test['name'] = test['series_name']

test.drop(['series_name'], axis=1, inplace=True)
train['name'] = train['series_name']

train.drop(['series_name'], axis=1, inplace=True)
train['bodyType'] = train['bodyType'].apply(lambda x: str(x).split()[0])

test['bodyType'] = test['bodyType'].apply(lambda x: str(x).split()[0])
missing_count = len(test[pd.isnull(test['Владение'])].index)

total_count = len(test.index)



print("В тесте владение не заполнено для {} строк из {}".format(missing_count, total_count))

test[pd.isnull(test['Владение'])]['Владельцы'].value_counts()
missing_count = len(train[pd.isnull(train['Владение'])].index)

total_count = len(train.index)



print("В трейне владение не заполнено для {} строк из {}".format(missing_count, total_count))



train[pd.isnull(train['Владение'])]['Владельцы'].value_counts()
test['Владение'].value_counts()
def owning_time_to_months(row):

    if not pd.isnull(row):

        words = row.split()

        

        if len(words) == 2:

            if words[1] == 'год' or words[1] == 'лет' or words[1] == 'года':

                months = int(words[0]) * 12

            else:

                months = int(words[0])

        else:

            months = (int(words[0]) * 12) + (int(words[3]))

        

        return(months)
def diff_month(d1, d2):

    return (d1.year - d2.year) * 12 + d1.month - d2.month
def fill_null(df,col1,col2,col3):

    

    months_median = df.groupby(col2).agg({col1: lambda x: x.median()}).reset_index()

    

#     df[col3] = df.apply(lambda x: diff_month(date.today(), datetime(int(x['productionDate']),6,1))

#                         if x['Владельцы'][0] == '1' and pd.isnull(x[col1])

#                         else months_median[months_median[col2] == x[col2]].iloc[0,1]

#                         , axis=1)

    

    df[col3] = df.apply(lambda x: diff_month(date.today(), datetime(int(x['productionDate']),6,1))

                        if x['Владельцы'][0] == '1' and pd.isnull(x[col1])

                        else (months_median[months_median[col2] == x[col2]].iloc[0,1]

                              if pd.isnull(x[col1]) else x[col1])

                        , axis=1)
# train['owning_months'] = train['Владение'].apply(owning_time_to_months)

# fill_null(train,'owning_months','Владельцы','owning_months')
# test['owning_months'] = test['Владение'].apply(owning_time_to_months)

# fill_null(test,'owning_months','Владельцы','owning_months')
def get_equipment_list(row):

    

    if len(row) == 2:

        return row

#         return np.nan

    else:

        dicts = ast.literal_eval(row[2:-2])

        

        full_list = []

        

        for d in dicts:

            res = [[i for i in d[x]] for x in d.keys()]

            

            for r in res[1:][0]:

                full_list.append(r)

#         print(ress[1:][0])

    

        return full_list
missing_count = len(train[train['Комплектация'] == '[]'].index)

total_count = len(train.index)



print("Комплектация не заполнена для {} строк из {}".format(missing_count, total_count))
train.drop(train[train['Комплектация'] == '[]'].index, inplace=True)
missing_count = len(test[test['Комплектация'] == '[]'].index)

total_count = len(test.index)



print("Комплектация не заполнена для {} строк из {}".format(missing_count, total_count))
train['Комплектация'] = train['Комплектация'].apply(get_equipment_list)
s = train['Комплектация']



mlb = MultiLabelBinarizer()

dummies = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=train.index)
# Полный список опций

options_full_list = list(dummies.columns)

options_full_list
dummies['price'] = train['price']

dummies
options_corr = dummies.corr()

options_corr = options_corr['price'].to_frame().reset_index()

options_corr.columns = ['option', 'coeff']

options_corr
options_corr = options_corr.sort_values(by=['coeff'], inplace=False, ascending=False)

options_corr = options_corr[(options_corr['coeff']>0.3) & 

                            (options_corr['option'] != 'price') &

                            (options_corr['option'] != 'Премиальная аудиосистема')]

options_corr
# Список опций с высоким влиянием на цену

options_short_list =[] 

  

for index, rows in options_corr.iterrows(): 

    # Create list for the current row 

    my_list = rows.option

      

    # append the list to the final list 

    options_short_list.append(my_list) 



options_short_list
def add_cols(target_df, source_df, cols_list):

    

    target_df['options_count'] = target_df.apply(lambda x: len(x['Комплектация']), axis=1)

    

    for col in source_df.columns:

        if col in cols_list:

            target_df[col] = source_df[col].astype(int)
add_cols(train, dummies, options_short_list)
train.head(3)
test['Комплектация'] = test['Комплектация'].apply(get_equipment_list)
s = test['Комплектация']



mlb = MultiLabelBinarizer()

dummies = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=test.index)
pd.Series(options_short_list).isin(dummies.columns).all()
add_cols(test, dummies, options_short_list)
test.head(3)
missing_count = len(test[pd.isnull(test['description'])].index)

total_count = len(test.index)



print("В тесте описание не заполнено для {} строк из {}".format(missing_count, total_count))
missing_count = len(train[pd.isnull(train['description'])].index)

total_count = len(train.index)



print("В трейне описание не заполнено для {} строк из {}".format(missing_count, total_count))
train['description'] = train['description'].str.lower()

test['description'] = test['description'].str.lower()
def add_descr_columns(df):

    df['desc_dealer']= df['description'].apply(lambda x: 1 if ('рольф' or 'салон'

                                                               or 'avilon' or 'inchcape'

                                                               or 'автодом' or 'асц'

                                                               or 'favorit') in x else 0)

  
# add_descr_columns(train)
# add_descr_columns(test)
# descr_columns = ['desc_dealer']
train_preproc = preproc_data(train)

# train_preproc = train_encoded

X_sub = preproc_data(test)

# X_sub = test_encoded
display(train_preproc.head(3))

# print(train.columns)

display(X_sub.head(3))
X = train_preproc.drop(['price'], axis=1,)

y = train_preproc.price.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
def mape(y_true, y_pred):

    return np.mean(np.abs((y_pred-y_true)/y_true))
train_preproc.head(5)
cat_features_ids = []



for col in X:

    if col not in ['modelDate', 'productionDate', 'enginePower', 

                   'mileage', 'engineDisplacement', 'options_count']:

        cat_features_ids.append(col)

        

# cat_features_ids
# cat_features_ids = cat_features_ids + options_short_list + descr_columns

cat_features_ids
model = CatBoostRegressor(iterations = ITERATIONS,

                          learning_rate = LR,

                          random_seed = RANDOM_SEED,

                          eval_metric='MAPE',

                          custom_metric=['R2', 'MAE']

                         )

model.fit(X_train, y_train,

         cat_features=cat_features_ids,

         eval_set=(X_test, y_test),

         verbose_eval=100,

         use_best_model=True,

         plot=True

         )
def cat_model(y_train, X_train, X_test, y_test):

    model = CatBoostRegressor(iterations = ITERATIONS,

                              learning_rate = LR,

                              eval_metric='MAPE',

                              random_seed = RANDOM_SEED,)

    model.fit(X_train, y_train,

              cat_features=cat_features_ids,

              eval_set=(X_test, y_test),

              verbose=False,

              use_best_model=True,

              plot=False)

    

    return(model)
submissions = pd.DataFrame(0,columns=["sub_1"], index=sample_submission.index) # куда пишем предикты по каждой модели

score_ls = []

splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))



for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS,):

    # use the indexes to extract the folds in the train and validation data

    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]

    # model for this fold

    model = cat_model(y_train, X_train, X_test, y_test,)

    # score model on test

    test_predict = model.predict(X_test)

    test_score = mape(y_test, test_predict)

    score_ls.append(test_score)

    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")

    # submissions

    submissions[f'sub_{idx+1}'] = model.predict(X_sub)

    model.save_model(f'catboost_fold_{idx+1}.model')

    

print(f'Mean Score: {np.mean(score_ls):0.3f}')

print(f'Std Score: {np.std(score_ls):0.4f}')

print(f'Max Score: {np.max(score_ls):0.3f}')

print(f'Min Score: {np.min(score_ls):0.3f}')
submissions.head(10)
submissions['blend'] = (submissions.sum(axis=1))/len(submissions.columns)

sample_submission['price'] = submissions['blend'].values

sample_submission.to_csv(f'submission_blend_v{VERSION}.csv', index=False)

sample_submission.head(10)
def encode_labels(df, cols):

    for col in cols:

        le = LabelEncoder()

        le.fit(df[col])

        df[col] = le.transform(df[col])

#         df['test'] = le.transform(df[col])
def encode_data(df_input):

    # productionDate, enginePower, mileage

    # engineDisplacement

    # Владельцы

    

#     df_input = df_input.copy()

    df_input = preproc_data(df_input)

    

#     df_input['Владельцы'].replace({"1 владелец": 1,"2 владельца": 2,"3 или более": 3},inplace=True)

    encode_labels(df_input, ['Владельцы'])

    

    dummy_cols = ['bodyType', 'color', 'fuelType', 'name', 'vehicleTransmission', 'Привод', 'ПТС']

    dummies = pd.get_dummies(df_input[dummy_cols])

    

    

    df_output = df_input[['productionDate', 'enginePower', 'mileage', 'engineDisplacement', 'Владельцы']]

    df_output = pd.concat([df_output, dummies], axis=1)

    

    options = df_input[options_short_list]

    df_output = pd.concat([df_output, options], axis=1)

    

    if 'price' in df_input.columns:

        df_output['price'] = df_input['price']

    

    return df_output

    
train_encoded = encode_data(train)

train_encoded

# train
test_encoded = encode_data(test)

test_encoded
cols = list(test_encoded.columns.difference(train_encoded.columns))

# cols
test_encoded.drop(cols, axis=1, inplace=True)

# test_encoded
cols = list(train_encoded.columns.difference(test_encoded.columns))

# cols
train_encoded.drop(cols[:-1], axis=1, inplace=True)

# train_encoded
print(len(train_encoded.columns))

print(len(test_encoded.columns))
train_cols = train_encoded.columns

test_cols = test_encoded.columns



common_cols = train_cols.intersection(test_cols)

train_not_test = train_cols.difference(test_cols)

   

print(common_cols)

print(train_not_test)
train_preproc = train_encoded

X_sub = test_encoded
X = train_preproc.drop(['price'], axis=1,)

y = train_preproc.price.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

# linear_reg = LogisticRegression(C=0.001, penalty='l1', solver='liblinear', max_iter=5000)

linear_reg.fit(X_train, y_train)

y_pred= linear_reg.predict(X_test)



print(mape(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor



rf_reg = RandomForestRegressor()

rf_reg.fit(X_train, y_train)

y_pred= rf_reg.predict(X_test)





print(mape(y_test, y_pred))

# print("Accuracy on Traing set: ",rf_reg.score(X_train,y_train))

# print("Accuracy on Testing set: ",rf_reg.score(X_test,y_test))
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 5, verbose=2, random_state=RANDOM_SEED, n_jobs = -1)

rf_random.fit(X_train, y_train)



rf_standard = RandomForestRegressor(random_state = RANDOM_SEED)

rf_standard.fit(X_train,y_train)

y_pred_st = rf_standard.predict(X_test)

# mse_st = round(mean_squared_error(Y_val,y_pred_st),2)



print(mape(y_test, y_pred_st))

# print(mse_st)

print(rf_random.best_params_)
rf_tunned = RandomForestRegressor(random_state = RANDOM_SEED,

                                  n_estimators=rf_random.best_params_['n_estimators'], 

                                  min_samples_split=rf_random.best_params_['min_samples_split'], 

                                  min_samples_leaf = rf_random.best_params_['min_samples_leaf'],

                                  max_features = rf_random.best_params_['max_features'], 

                                  max_depth = rf_random.best_params_['max_depth'], 

                                  bootstrap = rf_random.best_params_['bootstrap'])

rf_tunned.fit(X_train,y_train)

y_pred_tunned = rf_tunned.predict(X_test)





# test_predict = model.predict(X_test)

print(mape(y_test, y_pred_tunned))
from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor()



# dt_reg = RandomForestRegressor()

decision_tree.fit(X_train, y_train)

y_pred= decision_tree.predict(X_test)





print(mape(y_test, y_pred))
from sklearn.ensemble import GradientBoostingClassifier



def hyperopt_gb_score(params):

    clf = GradientBoostingRegressor(**params)

    current_score = cross_val_score(clf, X_train, y_train, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_gb = {

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            

        }

 

best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)

print('best:')

print(best)
params = space_eval(space_gb, best)

params
gradient_boosting = GradientBoostingRegressor(**params)

gradient_boosting.fit(X_train, y_train)



y_pred = gradient_boosting.predict(X_test)



print(mape(y_test, y_pred))
bagging = BaggingRegressor()

bagging.fit(X_train, y_train)



y_pred= bagging.predict(X_test)



print(mape(y_test, y_pred))
etr = ExtraTreesRegressor()

etr.fit(X_train, y_train)



y_pred= etr.predict(X_test)



print(mape(y_test, y_pred))
def cat_model(y_train, X_train, X_test, y_test):

    model = CatBoostRegressor(iterations = ITERATIONS,

                              learning_rate = LR,

                              eval_metric='MAPE',

                              random_seed = RANDOM_SEED,)

    model.fit(X_train, y_train,

              cat_features=cat_features_ids,

              eval_set=(X_test, y_test),

              verbose=False,

              use_best_model=True,

              plot=False)

    

    return(model)
cat_features_ids = []



for col in X:

    if col not in ['modelDate', 'productionDate', 'enginePower', 

                   'mileage', 'engineDisplacement', 'options_count']:

        cat_features_ids.append(col)

        

# cat_features_ids
def rfr_model(y_train, X_train, X_test, y_test):

    

    model = RandomForestRegressor(random_state = RANDOM_SEED,

                                  n_estimators=rf_random.best_params_['n_estimators'], 

                                  min_samples_split=rf_random.best_params_['min_samples_split'], 

                                  min_samples_leaf = rf_random.best_params_['min_samples_leaf'],

                                  max_features = rf_random.best_params_['max_features'], 

                                  max_depth = rf_random.best_params_['max_depth'], 

                                  bootstrap = rf_random.best_params_['bootstrap'])

    model.fit(X_train,y_train)

    

    return(model)
def gb_model(y_train, X_train, X_test, y_test):

    model = GradientBoostingRegressor(**params)



    model.fit(X_train,y_train)



    return(model)
def bagging_model(y_train, X_train, X_test, y_test):

    model = BaggingRegressor()

    

    model.fit(X_train,y_train)



    return(model)
def etr_model(y_train, X_train, X_test, y_test):

    model = ExtraTreesRegressor()

    

    model.fit(X_train,y_train)



    return(model)
def compute_meta_feature(clf, X_train, X_test, y_train, cv):



    X_meta_train = np.zeros_like(y_train, dtype = np.float32)

    X_meta_test = np.zeros(len(X_test), dtype=np.float32)

    for train_fold_index, predict_fold_index in cv.split(X_train):

        X_fold_train, X_fold_predict = X_train.iloc[train_fold_index], X_train.iloc[predict_fold_index]

        y_fold_train = y_train[train_fold_index]

        folded_clf = clone(clf)

        

        if type(clf).__name__ == 'CatBoostRegressor':

            folded_clf.fit(X_fold_train, y_fold_train, cat_features=cat_features_ids, verbose_eval = 100)

        else:

            folded_clf.fit(X_fold_train, y_fold_train)

            

        X_meta_train[predict_fold_index] = folded_clf.predict(X_fold_predict)

#         print_regression_metrics(X_meta_train[predict_fold_index], y_train.iloc[predict_fold_index])

        X_meta_test += folded_clf.predict(X_test)

    X_meta_test = X_meta_test / cv.n_splits



    return X_meta_train, X_meta_test







def generate_meta_features(classifiers, X_train, X_test, y_train, cv):

    features = [

        compute_meta_feature(clf, X_train, X_test, y_train, cv)

        for clf in tqdm(classifiers)

    ]



    stacked_features_train = np.stack([

        features_train for features_train, features_test in features

        ],axis=-1)



    stacked_features_test = np.stack([

        features_test for features_train, features_test in features

        ],axis=-1)



    return stacked_features_train, stacked_features_test
# np.random.seed(42)



cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)



model_cat = cat_model(y_train, X_train, X_test, y_test,)

model_gb = gb_model(y_train, X_train, X_test, y_test,)

model_rfr = rfr_model(y_train, X_train, X_test, y_test,)

model_bagging = bagging_model(y_train, X_train, X_test, y_test,)

model_etr = etr_model(y_train, X_train, X_test, y_test,)
# stacked_features_train, stacked_features_test = generate_meta_features([

#     model_cat, model_rfr, model_gb], X_train, X_test, y_train, cv)

stacked_features_train, stacked_features_test = generate_meta_features([

    model_cat, model_rfr, model_gb, model_bagging, model_etr], X_train, X_sub, y_train, cv)
stacked_features_train[0]

# train
from sklearn.linear_model import Ridge

model_stacked = Ridge(alpha=20)



model_stacked.fit(stacked_features_train, y_train)

y_pred = model_stacked.predict(stacked_features_test)
predict_submission = model_stacked.predict(stacked_features_test)
len(predict_submission)
sample_submission['price'] = predict_submission

# sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)

# sample_submission.head(10)
sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)

sample_submission.head(10)