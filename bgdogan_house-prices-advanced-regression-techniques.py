# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_original = train.copy()

target = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True)

test_index = test['Id']
missing_threshold = 0.05

def get_missing_ratios(data, ratio=missing_threshold):
    missing_ratios = pd.Series({col: train[col].isnull().sum() / train.shape[0] for col in train.columns}).sort_values(ascending=False)
    
    return missing_ratios[missing_ratios > ratio]
(set(get_missing_ratios(train, missing_threshold).index).union(
    set(get_missing_ratios(train, missing_threshold).index))).difference(set(get_missing_ratios(train, missing_threshold).index))
drop_cols = set(get_missing_ratios(train, missing_threshold).index)
drop_cols = drop_cols.union(set(['Id']))

train.drop(drop_cols, axis=1, inplace=True)
test.drop(drop_cols, axis=1, inplace=True)
def show_cat_counts(train, test, field):
    plt.figure(figsize=(6, 2))
    plt.subplot(1,2, 1)
    plt.bar(train[field].value_counts().index.values, train[field].value_counts())
    plt.title(field + ' train')
    plt.subplot(1, 2, 2)
    plt.bar(test[field].value_counts().index.values, test[field].value_counts())
    plt.title(field + ' test')
train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
#loops = train.shape[1] // 5
#for j in range(loops):
#    for col in train.columns[j*5:(j+1)*5]:
#        show_cat_counts(train, test, col)
col_cat = [col for (col, typ) in train.dtypes.iteritems() if typ == 'O']
col_num = [col for (col, typ) in train.dtypes.iteritems() if typ != 'O']
col_size = 4
rows = len(col_cat) // col_size
cols = col_size
import seaborn as sns
import matplotlib.pyplot as plt

#plt.figure(figsize=(20, 20))

#for i in range(len(col_cat)):
#    col = col_cat[i]
#    #row = i // 5 + 1
#    #col_index = i % 5 + 1
#    plt.subplot(rows, cols, i + 1)
#    sns.barplot(train[col].value_counts().index, train[col].value_counts())
    
#plt.show()
from sklearn.impute import SimpleImputer

def Impute(train, test, cols, strategy='mean'):
    simpleImputer = SimpleImputer(strategy=strategy)
    
    train_imp = train.copy()
    test_imp = test.copy()

    train_imp[cols] = pd.DataFrame(simpleImputer.fit_transform(train[cols]))
    test_imp[cols] = pd.DataFrame(simpleImputer.transform(test[cols]))
    
    train_imp.columns = train.columns
    test_imp.columns = test.columns
    
    return (train_imp, test_imp)

train_imp = train.copy()
test_imp = test.copy()

(train_imp, test_imp) = Impute(train_imp, test_imp, col_num, 'mean')
(train_imp, test_imp) = Impute(train_imp, test_imp, col_cat, 'most_frequent')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def LabelEncode(train, test, cols):
    le = LabelEncoder()
    
    train_enc = train.copy()
    test_enc = test.copy()

    for col in cols:
        train_enc[col] = le.fit_transform(train_enc[col])
        test_enc[col] = le.transform(test_enc[col])
        
    return (train_enc, test_enc)

def OneHotEncode(train, test, cols):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    train_enc = train.copy()
    test_enc = test.copy()

    train_ohe = pd.DataFrame(ohe.fit_transform(train_enc[cols]))
    test_ohe = pd.DataFrame(ohe.transform(test_enc[cols]))
    
    train_ohe.index = train_enc.index
    test_ohe.index = test_enc.index
    
    train_enc = pd.concat([train_enc, train_ohe], axis=1)
    test_enc = pd.concat([test_enc, test_ohe], axis=1)
    
    train_enc.drop(cols, axis=1, inplace=True)
    test_enc.drop(cols, axis=1, inplace=True)
        
    return (train_enc, test_enc)


(train_enc, test_enc) = LabelEncode(train_imp, test_imp, col_cat)
#(train_enc, test_enc) = OneHotEncode(train_imp, test_imp, col_cat)
train_full = pd.concat([train_enc, target], axis=1)
corr = train_full.corr()['SalePrice']
cols_to_keep = (corr[abs(corr) > 0.1].drop(['SalePrice'], axis=0)).index.values
#train_enc = train_enc[cols_to_keep]
#test_enc = test_enc[cols_to_keep]
from sklearn.model_selection import train_test_split

y = target
X = train_enc

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
from sklearn.metrics import mean_absolute_error

def TestRandomForest(X_train, X_valid, y_train, y_valid, n_estimators):
    my_model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    my_model.fit(X_train, y_train)
    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

def CheckRandomForestEstimators():
    from sklearn.ensemble import RandomForestRegressor
    
    for i in [100, 200, 230]:
        TestRandomForest(X_train, X_valid, y_train, y_valid, i)


def ApplyRandomForest():
    from sklearn.ensemble import RandomForestRegressor
    
    my_model = RandomForestRegressor(n_estimators=200, random_state=0)
    my_model.fit(X_train, y_train)
    
    return my_model

def ApplyXGB():
    from xgboost import XGBRegressor

    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
    my_model.fit(X_train, y_train, 
                 early_stopping_rounds=5, 
                 eval_set=[(X_valid, y_valid)], 
                 verbose=True)
    
    return my_model
def ApplyDeepLearning(epochs=4, batch_size=16):
    import keras
    from keras import layers
    from keras import models
        
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    _X = (X - mean) / std
        
    _X_train, _X_valid, _y_train, _y_valid = train_test_split(_X.values, y.values, train_size=0.8, test_size=0.2,
                                                                random_state=0)    
    
    my_model = models.Sequential()
    my_model.add(layers.Dense(128, activation='relu',
                           input_shape=(_X_train.shape[1],)))
    my_model.add(layers.Dropout(0.5))
    my_model.add(layers.Dense(128, activation='relu'))
    my_model.add(layers.Dropout(0.5))
    my_model.add(layers.Dense(1))
    
    my_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    history = my_model.fit(_X_train, _y_train, batch_size=batch_size, epochs=epochs, validation_data=(_X_valid, _y_valid), verbose=1)
    #my_model.fit(_X, y, batch_size=16, epochs=100, validation_split=0.2)
    
    return (my_model, _X_valid, mean, std, history)
#my_model = ApplyRandomForest()
my_model = ApplyXGB()
predictions = my_model.predict(X_valid)

#epochs=100
#batch_size=32
#(my_model, _X_valid, mean, std, history) = ApplyDeepLearning(epochs=epochs, batch_size=batch_size)
#predictions = my_model.predict(_X_valid)
plt.figure()
plt.plot(range(1, epochs+1), history.history['loss'], label='loss')
plt.plot(range(1, epochs+1), history.history['val_loss'], label='val_loss')
plt.legend()

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
# used for only Deep Learning
#test_enc = (test_enc - mean) / std

preds_test = my_model.predict(test_enc)

# Save test predictions to file
output = pd.DataFrame({'Id': test_index,
                       'SalePrice': preds_test.ravel()})
output.to_csv('submission.csv', index=False)