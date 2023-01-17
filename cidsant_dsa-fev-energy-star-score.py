# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/dataset_treino.csv', na_values=['Not Available', 'NaN'])
def filter_percent(df, perc = 0.9):

    lst = [] 

    for col in df.columns:

        value = df[col].value_counts(normalize=True).iloc[0]

        if value >= perc:

            lst.append(col)

    return lst        

    

def print_summary(df, name, max_list_values = 3, sm_list = None):

    len_values = len(df[name].unique())

    print(name, ':', len_values)

    if sm_list is not None:

        sm_list.append(df[name].value_counts(normalize=True))

    if len_values <= max_list_values:

        print(df[name].value_counts(normalize=True))

    print('----------------------------------')

    

def print_most_important_variables(est, feature_list, min_value_to_return = 0.0):

    # Get numerical feature importances

    importances = list(est.feature_importances_)



    # List of tuples with variable and importance

    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]



    # Sort the feature importances by most important first

    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)



    # Print out the feature and importances 

    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    return [f for f, v  in feature_importances if v >= min_value_to_return]

    

def compare_value(x):

    st = str(x).replace('\u200b', '1').strip()

    try:

        return int(st[:1] if x else 0) 

    except:

        return 0 

    

def zero_to_mean(df, name, mean = None, min_value = 0):

    if mean is None:

        mean = data[name].mean()

    df[name]  = df[name].apply(lambda x: mean if x <= min_value else x )



def fill_na_values(df, name, mean = None):

         

    if mean is None:

        if df[name].dtype == 'object':

            mean = data[name].mode().iloc[0]

        else:

            mean = data[name].mean()

    #if df[name].dtype == 'object':

    #    df[name] = df[name].apply(lambda x: x if type(x) == type("") else None)

        

    df[name].fillna(mean, inplace=True)

        

    return df



    

def remove_outliers(df, names, low=0.1, high=0.9):

    df_filter = data_train[names]

    df_ret = df_filter.quantile([low, high])

    return df_filter.apply(lambda x: x[(x > df_ret.loc[low, x.name]) & (x < df_ret.loc[high, x.name])], axis=0)



def transform_missing(df):

    data_missing = df.apply(lambda x: sum(x.isnull()),axis=0)



    missing_columns = []

    for k, v in data_missing.iteritems():

        if v > 0:

            #print(k, v)

            missing_columns.append({'k': k, 'type': 'str' if df[k].dtype == 'O' else 'int'})



    return missing_columns



def apply_for_missing(df, missing_columns, verbose=0):

    for field in missing_columns:

        name = field['k']

        if field['type'] == 'int':

            mean = data[name].mean()

        else:

            mean = data[name].mode().iloc[0]



        if verbose > 0:

            print(name, ':', mean)

            

        fill_na_values(df, name, mean)

    return df



def apply_log(df, cols):

    for col in cols:

        df[col] = np.log(df[col])
data['BBL'] = data['BBL - 10 digits'].apply(compare_value)
CST_PREDICT = 'ENERGY STAR Score'



CST_COLOUMNS = ['ENERGY STAR Score', 'BBL', 'Borough', 'DOF Gross Floor Area', 'Primary Property Type - Self Selected',

                'Largest Property Use Type - Gross Floor Area (ft²)', 'Year Built', 

                #'Number of Buildings - Self-reported',

                'Fuel Oil #1 Use (kBtu)', 'Fuel Oil #2 Use (kBtu)', 'Fuel Oil #4 Use (kBtu)',

                'Diesel #2 Use (kBtu)',

                #'District Steam Use (kBtu)', 

                'Occupancy', 

                #'Metered Areas (Energy)', 

               # 'Metered Areas  (Water)', 

                'Site EUI (kBtu/ft²)',

                'Weather Normalized Site EUI (kBtu/ft²)', 'Weather Normalized Site Electricity Intensity (kWh/ft²)', 

                'Weather Normalized Site Natural Gas Intensity (therms/ft²)', 'Weather Normalized Source EUI (kBtu/ft²)', 

                'Natural Gas Use (kBtu)', 'Weather Normalized Site Natural Gas Use (therms)', 

                'Electricity Use - Grid Purchase (kBtu)', 'Weather Normalized Site Electricity (kWh)', 

                'Total GHG Emissions (Metric Tons CO2e)', 'Direct GHG Emissions (Metric Tons CO2e)', 

                'Indirect GHG Emissions (Metric Tons CO2e)', 'Property GFA - Self-Reported (ft²)', 

                'Water Use (All Water Sources) (kgal)', 'Water Intensity (All Water Sources) (gal/ft²)', 

                'Source EUI (kBtu/ft²)', 

                'Water Required?', 

                #'DOF Benchmarking Submission Status', 

                'Community Board', 'Council District', 'Census Tract']
data_train = data[CST_COLOUMNS].copy()

missing_columns = transform_missing(data_train)

apply_for_missing(data_train, missing_columns, verbose=0).head(1)


def apply_primary_property_type(x):

    if x == 'Multifamily Housing':

        return 1

    elif x == 'Office':

        return 2

    elif x == 'Hotel':

        return 3

    else:

        return 0

    

    

def apply_borough(x):

    if x == 'Manhattan':

        return 1

    elif x == 'Brooklyn':

        return 2

    elif x == 'Queens':

        return 3

    elif x == 'Bronx':

        return 4

    else:

        return 0

    

def apply_water_required(x):

    if x == 'Yes':

        return 1    

    else:

        return 0

    

dict_apply = {'Borough': apply_borough, 

             'Water Required?':apply_water_required, 

            'Primary Property Type - Self Selected':apply_primary_property_type}
data_train['Borough'].value_counts()
cat_data = data_train.select_dtypes(include='object').copy()

cat_data.head(2)
def apply_to_numeric_data(df):    

    cat_data = df.select_dtypes(exclude='object').copy()

    if CST_PREDICT in cat_data.columns.tolist():

        col_auts = cat_data.drop(CST_PREDICT, axis=1).columns

    else:

        col_auts = cat_data.columns



    for col in col_auts:

        zero_to_mean(df, name = col, mean = None, min_value = 0)

    

    apply_log(df, col_auts)

    del cat_data

    

    

from sklearn.preprocessing import LabelEncoder 

def to_numeric(x):

    try:

        if int(x.strip()) > -1:

            return True

    except:

        pass

    return False



def encoder_categorical_data2(df):    

    cat_data = df.select_dtypes(include='object').copy()

    le = LabelEncoder() 

    for i in cat_data.columns:    

        #print(df[i])

        df[i] = le.fit_transform(df[i]) 

        

        

    del cat_data

    

def encoder_categorical_data(df):    

    cat_data = df.select_dtypes(include='object').copy()

    for col in cat_data.columns.tolist():

        if col in dict_apply:

            print(col)

            df[col] = df[col].apply(dict_apply[col])            

        

    del cat_data
use_apply_to_numeric = True

use_apply_to_categorical = True
if use_apply_to_numeric:

    apply_to_numeric_data(data_train)

    

if use_apply_to_categorical:

    encoder_categorical_data(data_train)
data_train.head()
from sklearn.preprocessing import MinMaxScaler

def format_dataset(df, columns = None, scaler_X = None,scaler_y = None ):

    sc_data = df.select_dtypes(exclude='object').copy()

    if columns is not None:

        sc_data = sc_data[columns]

        cols = columns

    else:

        cols = sc_data.iloc[:, 1:len(sc_data.columns)].columns

    if scaler_X is None:

        scaler_X = MinMaxScaler()

        scaler_X = scaler_X.fit(sc_data[cols])

        

    X = scaler_X.transform(sc_data[cols])

    

    if scaler_y is None:

        scaler_y = MinMaxScaler(feature_range=(0, 1))

        scaler_y = scaler_y.fit(data_train[CST_PREDICT].values.reshape(-1, 1))

        

    y = scaler_y.transform(data_train[CST_PREDICT].values.reshape(-1, 1))

    

    return X, y, scaler_X, scaler_y, cols
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from math import sqrt
from sklearn.ensemble import RandomForestRegressor



def create_boosting(X_train, y_train, X_test, y_test, n_estimators = 500):

    """

    Cria o GradientBoost

    """

    

    

    est = GradientBoostingRegressor(n_estimators=n_estimators,

                                    #0.094

                                    learning_rate= 0.005, max_depth= 4, 

                                    max_features= 'sqrt', min_samples_leaf= 5,

                                    

                                    #learning_rate= 0.01, max_depth= 8, max_features= 'sqrt', 

                                    #min_samples_leaf= 5,                                                    

                                    random_state=0)



    #cria modelo

    est.fit(X_train, y_train)



    #regressão

    test_best_values(est, X_test, y_test)

    

    #grid de parametros

    param_grid_cv = {

        "max_features":["sqrt", "log2", None, 1, 2],

        'learning_rate':[0.05, 0.01, 0.001, 0.005],

        'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True),

        'max_depth':[1, 2, 3, 4, 6, 8, 10, None],        

    }



    return est, param_grid_cv



def create_forest(X_train, y_train, X_test, y_test, n_estimators = 500):

    """

    Cria o GradientBoost

    """

    

    max_features = list(range(1, X_train.shape[1] // 2))

    max_features.append('auto')

    max_features.append('log2')

    #grid de parametros

    param_grid_cv = {

        "max_features":max_features,

        #'learning_rate':[0.05, 0.01, 0.001, 0.005],

        #'min_samples_split':[ 2, 3, 4, 5, 6],

        'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),

        #'max_depth':[None, 1, 2, 3, 4, 6, 8, 10],   

        'n_estimators':[100,200,300,500,1000,5000]

    }



    est = RandomForestRegressor(bootstrap=True, 

            criterion='mae',

            max_depth=None, max_features='auto',

            min_samples_leaf=1,

            n_estimators=n_estimators, n_jobs=-1,

            oob_score=False, random_state=52, verbose=0)



    est.fit(X_train, y_train)

    #regressão

    test_best_values(est, X_test, y_test)

    

    return est, param_grid_cv



def create_best_estimator(est, X, y, param_grid_cv, scoring =None, n_folds= None,verbose=0):

    """

    Cria o GridSearchCV 

    """

    

    gs_cv = GridSearchCV(est, param_grid=param_grid_cv, scoring=scoring, 

                         n_jobs=-1, cv=n_folds, verbose=verbose, refit=scoring).fit(X, y)

    #imprime os melhores parametros

    print('Melhores Hyperparametros: %r' % gs_cv.best_params_)

    est.set_params(**gs_cv.best_params_)

    est.fit(X, y)

    return gs_cv

def mae(y_true, y_pred):

    return np.mean(abs(y_true - y_pred))

    

    

def test_best_values(model, X_test, y_test):

    #previsão das classes (labesl)

    pred = model.predict(X_test)

    mae_ = mean_absolute_error(y_true=y_test, y_pred=pred)

    print('MAE: %.3f' % mae_)

    print('MAE: %.3f' % mae(y_test, pred))

    mse= mean_squared_error(y_true=y_test, y_pred=pred)

    print('MSE: %.3f' % mse)

    print('RMSE: %.3f' % sqrt(mse))

    print('R2: %.3f' % r2_score(y_true=y_test, y_pred=pred))

    return mae_

    

    #calculando a previsção de probabilidade de cada classe

    #print(model.predict_proba(X_test)[0])
X, y, scaler_X, scaler_y, cols = format_dataset(df = data_train, columns = None, scaler_X = None,scaler_y = None )

feature_importances = None
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)
use_forest = False


if use_forest:

    est, param_grid_cv = create_forest(X_train, y_train, X_test, y_test,n_estimators = 5000)

else:

    est, param_grid_cv = create_boosting(X_train, y_train, X_test, y_test,n_estimators = 5000)
print(len(est.feature_importances_))

print(len(data_train.columns.values))

feature_importances = print_most_important_variables(est, cols, 0.01)
X, y, scaler_X, scaler_y, cols = format_dataset(df = data_train, columns = feature_importances, scaler_X = None,scaler_y = None )

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)
len(cols), len(feature_importances)
#gsc = create_best_estimator(est, X, y, param_grid_cv, 

#                            scoring ='neg_mean_absolute_error', n_folds= 5,verbose=1)
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam, SGD, RMSprop



def build_model(units, X, opt, dropout=True):    

    model = Sequential()

    model.add(Dense(units[0], input_dim=X.shape[1], activation='relu', kernel_initializer='uniform'))

    

    if dropout:

        model.add(Dropout(0.2))

        

    for i in range(1, len(units)):

        model.add(Dense(units[i], activation='relu', kernel_initializer='uniform'))

        if dropout:

            model.add(Dropout(0.2))

            

    model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))

    #model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mae'])

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

    

    return model
X_train.shape
from keras.callbacks import EarlyStopping, ModelCheckpoint



import datetime

import time

start_time = time.time()

print(datetime.datetime.now())





verbose = 1



filepath_model_checkpoint="weights.best.hdf5"

#opt = RMSprop(lr=0.01, decay=0)

#opt = Adam(lr=0.001)

#model = build_model([150, 50], X, opt, dropout=True)



#opt = Adam(lr=0.001, decay=1e-6)

#model = build_model([1000,600,200,10], X, opt, dropout=True)

opt = Adam(lr=0.001)

#adam-0.001-200.100.hdf5

model = build_model([200,100], X, opt, dropout=True)

model_checkpoint = ModelCheckpoint(filepath=filepath_model_checkpoint, monitor='val_mean_absolute_error', verbose=verbose, mode='min', save_best_only=True)

monitor = EarlyStopping(monitor = 'val_mean_absolute_error', min_delta = 1e-3, patience = 50, verbose = verbose, mode = 'min')

callbacks_list = [model_checkpoint]





model.fit(X, y, batch_size=100, epochs=2000, validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=verbose)

model.load_weights(filepath_model_checkpoint)



test_best_values(model, X_test, y_test)



print(datetime.datetime.now())

print(time.time() - start_time )
test_best_values(model, X_test, y_test)
#model_list = do_train(X, y, verbose=0)

#mae: 0.0958  name: weights.best-adam.decay-0.001-200.100.hdf5
def load_test(df_train, scaler_X, scaler_y, cols, verbose= 0):

    data  = pd.read_csv('../input/dataset_teste.csv', na_values=['Not Available', 'NaN'])

    

    data['BBL'] = data['BBL - 10 digits'].apply(compare_value)

    

    if verbose > 0:

        for value in CST_COLOUMNS[1:]:

            print_summary(data, value, 6)



    df_test = data[CST_COLOUMNS[1:]].copy()





    data_missing = transform_missing(df_test)





    apply_for_missing(df_test, missing_columns, verbose=verbose)

    

    if  use_apply_to_numeric:

        apply_to_numeric_data(df_test)

    

    if use_apply_to_categorical:

        encoder_categorical_data(df_test)

    

    #X = scaler_X.transform(df_test[cols])



    X, _, scaler_X, _, _ = format_dataset(df = df_test, columns = cols, scaler_X = scaler_X,scaler_y = None )

    print(X.shape)

    

    return df_test, X, data
df_test, X_submit,  data_test = load_test(data_train, scaler_X, scaler_y, cols=feature_importances, verbose = 0)
def make_submition(model, X_test, data_test, scaler_y):

    save_file_name = 'submission.csv'

    

    #X_test = X_test.astype('float32')

    print(X_test.shape)

    pred = model.predict(X_test)

    print(pred.shape)

    pred = scaler_y.inverse_transform(pred.reshape(-1, 1))

    

    df_submit = data_test[['Property Id']]

    df_submit['score'] = pred

    df_submit['score'] = df_submit['score'].apply(lambda x: int(x) if x < 100.0 else 100)

    df_submit['score'] = df_submit['score'].apply(lambda x: int(x) if x > 0 else 0)

    df_submit.to_csv(save_file_name, sep=',',index=False)

    print('submission sent')
make_submition(model, X_submit, data_test, scaler_y)