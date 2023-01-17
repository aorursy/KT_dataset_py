import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = "0"

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from scipy import stats
from pandas.api.types import CategoricalDtype

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

pd.set_option("display.max_columns", 100)
pd.set_option("mode.chained_assignment", None)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
print('Shape:', df_train.shape)
df_train.head(5)
target_column = 'SalePrice'
def find_columns_with_missing_value(df):
    missing_value_count = df.isnull().sum()
    missing_value_percentage = missing_value_count / df.isnull().count()
    missing_value_type = df.dtypes
    missing_value = pd.DataFrame({
        "Total": missing_value_count,
        "Percentage": missing_value_percentage,
        "Type": missing_value_type
    })
    return missing_value[missing_value['Total'] > 0].sort_values(by="Total",ascending=False)
find_columns_with_missing_value(df_train)
def find_numeric_columns(df):
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numeric_types)
    return numeric_df
def find_non_numeric_columns(df):
    non_numeric_types = ['object']
    non_numeric_df = df.select_dtypes(include=non_numeric_types)
    return non_numeric_df
numeric_df_train = find_numeric_columns(df_train)
numeric_df_train = numeric_df_train.drop([target_column, 'Id'], axis=1)

non_numeric_df_train = find_non_numeric_columns(df_train)

print("Numeric columns:", numeric_df_train.columns)
print("Non-numeric columns:", non_numeric_df_train.columns)
def series_has_null_value(series):
    return series.isnull().values.any()
def replace_nan_with_zero(df):
    for col in df.columns:
        if series_has_null_value(df[col]):
            df[col] = df[col].fillna(0)
            
    return df
numeric_df_train = replace_nan_with_zero(numeric_df_train)
numeric_df_train.head(5)
def one_hot_encode(df, categories_dict={}):
    missing_value_representation = 'None'
    
    # iteratively encode each column
    for col in df.columns:
        # replace missing value if any
        if series_has_null_value(df[col]):
            df[col] = df[col].fillna(missing_value_representation)
        
        # get numpy array from the series
        X = df[col].values
        # transform the array shape to satisfy sklearn OneHotEncoder format
        X = X.reshape(-1,1)
        
        # get unique values for its column as categories
        if col not in categories_dict:
            unique_value = df[col].unique()
            unique_value.sort()
            categories_dict[col] = unique_value
        
        # create one hot encoder and feed our data
        encoder = OneHotEncoder(categories=[categories_dict[col]], handle_unknown='ignore')
        encoder.fit(X)
        
        # encode the value
        encoded_data = encoder.transform(X).toarray()
        
        # get the columns name, it will produce column name with format "x0_value"
        encoded_columns_name = encoder.get_feature_names()
        # remove x0 and replace it with actual column name and remove column representing missing value
        encoded_columns_name = [name.replace("x0", col) 
                                for name in encoded_columns_name]
        
        # transform the encoded value to DataFrame format
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns_name)
        
        # omit missing value column
        filtered_columns = [name for name in encoded_columns_name 
                            if name[name.find("_")+1:] != missing_value_representation]
        encoded_df = encoded_df[filtered_columns]
        
        # combine it to our result DataFrame
        df = pd.concat([df, encoded_df], axis=1)
        # remove the original column
        df.drop([col], axis=1, inplace=True)
    
    return df, categories_dict
non_numeric_df_train, categories_dict = one_hot_encode(non_numeric_df_train)
non_numeric_df_train.head(5)
def get_input_and_target(numeric_df_train, non_numeric_df_train):
    input_train = pd.concat([numeric_df_train, non_numeric_df_train], axis=1)
    target_train = df_train[target_column]

    X = input_train.values
    Y = target_train.values
    
    return X, Y
X, Y = get_input_and_target(numeric_df_train, non_numeric_df_train)
def build_model(input_shape, lr):
    NN_classifier = Sequential()

    NN_classifier.add(Dense(128, kernel_initializer='normal', activation='relu', input_dim=input_shape))
    NN_classifier.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_classifier.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_classifier.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    NN_classifier.add(Dense(1, kernel_initializer='normal', activation='linear'))
    
    optimizer = Adam(lr=lr)
    NN_classifier.compile(loss='mean_absolute_error', optimizer=optimizer)
    
    NN_classifier.summary()
    
    return NN_classifier
def cross_validation_score(X, Y, lr, epochs, batch_size, validation_portion, train_history_callback, n_fold=5):
    np.random.seed(1234)
    rn.seed(1234)
    tf.set_random_seed(1234)
    
    kf = KFold(n_splits=n_fold)

    results = []
    histories = []
    for train, test in kf.split(X, Y):
        X_train, Y_train = X[train], Y[train]
        X_test, Y_test = X[test], Y[test]

        # Build classification model
        model = build_model(X.shape[1], lr)
        
        # use ModelCheckpoint to automatically save the best model's weight into hdf5 file
        model_path = "model.hdf5"
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only = True)
        callbacks_list = [checkpoint]
        
        # train model
        history = model.fit(
            X_train, 
            Y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_portion, 
            callbacks=callbacks_list)
        
        # load the best model's weight
        model.load_weights(model_path)
        
        result = model.evaluate(X_test, Y_test)
        results.append(result)
        histories.append(history)
        
    for index, history in enumerate(histories):
        train_history_callback(history, index)
        
    results = np.array(results)
    return results.mean()
def train_history_plot(history, index=0):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss - cross validation {0}'.format(index+1))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
def measure_model_performance(X, Y, lr=0.001, epochs=100, batch_size=32, validation_proportion=0.2):
    score = cross_validation_score(X, Y, 
                                   lr=lr,
                                   epochs=epochs, 
                                   batch_size=batch_size, 
                                   validation_portion=validation_proportion,
                                   train_history_callback=train_history_plot)

    print("Mean absolute error:", score)
measure_model_performance(X, Y)
def plot_correlation(corr):
    fig = plt.figure(figsize = (15,15))

    sb.heatmap(corr, square = True)
    plt.show()
numeric_df_train = find_numeric_columns(df_train)
numeric_df_train_corr = numeric_df_train.corr()

plot_correlation(numeric_df_train_corr)
def filter_numeric_columns_by_correlation(corr, threshold):
    indices = corr[target_column].map(lambda x: abs(x) >= threshold)
    columns = corr[target_column][indices].index
    columns = [x for x in columns if x != target_column]
    return columns
# these columns are excluded because these columns decreasing the model performance
excluded_columns = ['GarageYrBlt','2ndFlrSF','OpenPorchSF']

numeric_df_train_columns = filter_numeric_columns_by_correlation(numeric_df_train_corr, threshold=0.3)
numeric_df_train_columns = [col for col in numeric_df_train_columns if col not in excluded_columns]

numeric_df_train = numeric_df_train[numeric_df_train_columns]
numeric_df_train = replace_nan_with_zero(numeric_df_train)

print(numeric_df_train_columns)
X, Y = get_input_and_target(numeric_df_train, non_numeric_df_train)
measure_model_performance(X, Y)
numeric_df_train['BuildingAge'] = df_train['YrSold'] - df_train['YearRemodAdd']
numeric_df_train.drop('YearBuilt', axis=1, inplace=True)

print(numeric_df_train.columns)
X, Y = get_input_and_target(numeric_df_train, non_numeric_df_train)
measure_model_performance(X, Y)
def mean_normalization(df, normalization_dict={}):
    for col in df.columns:
        if col not in normalization_dict:
            normalization_dict[col] = {
                'min' : df[col].min(),
                'max' : df[col].max(),
                'mean' : df[col].mean()
            }
            
        min_value = normalization_dict[col]['min']
        max_value = normalization_dict[col]['max']
        mean_value = normalization_dict[col]['mean']
        df[col] = (df[col] - mean_value) / (max_value - min_value)
    
    return df, normalization_dict
numeric_df_train, normalization_dict = mean_normalization(numeric_df_train)
        
numeric_df_train.head(5)
X, Y = get_input_and_target(numeric_df_train, non_numeric_df_train)
measure_model_performance(X, Y)
def ANOVA_test(df, target_column, alpha_level=0.05):
    columns = []
    for col in df.columns:
        if col != target_column:
            values = df[col][df[col].isnull() == False].unique()
            groups = [df[target_column][df[col] == group] for group in values]

            df_between = len(values) - 1
            df_within = sum([len(members) for members in groups]) - len(values)
            f_crit = stats.f.ppf(q=1-alpha_level, dfn=df_between, dfd=df_within)

            f_stat, p_value = stats.f_oneway(*groups)
            if p_value <= alpha_level:
                columns.append(col)
            
    return columns
# These columns are excluded because contains a lot of missing values
excluded_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Utilities']

non_numeric_df_train = find_non_numeric_columns(df_train)

non_numeric_df_train_columns = non_numeric_df_train.columns
non_numeric_df_train_columns = [col for col in non_numeric_df_train_columns if col not in excluded_columns]

non_numeric_df_train = non_numeric_df_train[non_numeric_df_train_columns]
non_numeric_df_train = pd.concat([non_numeric_df_train, df_train[target_column]], axis=1)

non_numeric_df_train_columns = ANOVA_test(non_numeric_df_train, target_column, alpha_level=0.01)

print(non_numeric_df_train_columns)
non_numeric_df_train = non_numeric_df_train[non_numeric_df_train_columns]
non_numeric_df_train, categories_dict = one_hot_encode(non_numeric_df_train)

non_numeric_df_train.head(5)
X, Y = get_input_and_target(numeric_df_train, non_numeric_df_train)
measure_model_performance(X, Y)
def build_model(input_shape, lr):
    NN_classifier = Sequential()

    print("new build model function")
    NN_classifier.add(Dropout(0.2, seed=1234, input_shape=(input_shape,)))
    NN_classifier.add(Dense(128, kernel_initializer='normal', activation='relu'))
    NN_classifier.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_classifier.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_classifier.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    NN_classifier.add(Dense(1, kernel_initializer='normal', activation='linear'))
    
    optimizer = Adam(lr=lr)
    NN_classifier.compile(loss='mean_absolute_error', optimizer=optimizer)
    
    NN_classifier.summary()
    
    return NN_classifier
X, Y = get_input_and_target(numeric_df_train, non_numeric_df_train)
measure_model_performance(X, Y, epochs=500, lr=0.0002)
np.random.seed(1234)
rn.seed(1234)
tf.set_random_seed(1234)

X, Y = get_input_and_target(numeric_df_train, non_numeric_df_train)
model = build_model(X.shape[1], lr=0.0002)

# use ModelCheckpoint to automatically save the best model's weight into hdf5 file
model_path = "model.hdf5"
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only = True)
callbacks_list = [checkpoint]

# train model
history = model.fit(
    X, Y, 
    epochs=500, 
    batch_size=32, 
    validation_split=0.1, 
    callbacks=callbacks_list)

# load the best model's weight
model.load_weights(model_path)

train_history_plot(history)
df_test = pd.read_csv('../input/test.csv')
print(df_test.shape)
df_test.head(5)
df_test['BuildingAge'] = df_test['YrSold'] - df_test['YearRemodAdd']
numeric_columns = numeric_df_train.columns
non_numeric_columns = non_numeric_df_train_columns

numeric_df_test = df_test[numeric_columns]
non_numeric_df_test = df_test[non_numeric_columns]

print("Numeric columns:", numeric_columns)
print("Non-numeric columns:", non_numeric_columns)
find_columns_with_missing_value(numeric_df_test)
find_columns_with_missing_value(non_numeric_df_test)
numeric_df_test = replace_nan_with_zero(numeric_df_test)
numeric_df_test, _ = mean_normalization(numeric_df_test, normalization_dict)

numeric_df_train.head(5)
non_numeric_df_test, _ = one_hot_encode(non_numeric_df_test, categories_dict)

non_numeric_df_test.head(5)
X_test = pd.concat([numeric_df_test, non_numeric_df_test], axis=1).values

predicted = model.predict(X_test)
print(predicted.shape)
submission = pd.DataFrame({'Id':df_test.Id,'SalePrice':predicted[:,0]})
submission.to_csv('{}.csv'.format("submission"),index=False)
print('A submission file has been made')