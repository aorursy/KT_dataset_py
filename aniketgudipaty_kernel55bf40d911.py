# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
def binary_label_encoder(y):
    '''
    converts multiclass labels to binary labels
    '''
    y["BinaryLabel"] = (y["Label"] != 'BENIGN').astype(int)

    return y


def split_dataset(df):
    '''
    Takes a dataframe df and splits it in 0.7:0.15:0.15 ratio for training, cross-validation and test sets.
    '''
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    for train_index, test_index in split.split(df, df['BinaryLabel']):
        train = df.loc[train_index]
        test = df.loc[test_index]

    for train_index, cv_index in split.split(train, train['BinaryLabel']):
        train = df.loc[train_index]
        cv = df.loc[cv_index]

    return train, cv, test
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv("../input/resized-data/resized_data.csv")
dataset = binary_label_encoder(dataset)
train, cv, test = split_dataset(dataset)


y = train[['BinaryLabel']]
X = train.drop(['Label', 'BinaryLabel'], axis=1, inplace=False)
cols = X.columns.tolist()
mm_scaler = MinMaxScaler()
X_new = X
X_new[cols] = mm_scaler.fit_transform(X[cols])
y = y['BinaryLabel'].values
features_needed = 15
print("Selection based on chi square distibution")

chi_selector = SelectKBest(chi2, k=features_needed)
X_kbest_features = chi_selector.fit_transform(X_new, y)
chi_support = chi_selector.get_support()
chi_feature = X_new.loc[:, chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')
print(chi_feature)
X_chi = X_new[chi_feature]
X_chi
from keras.models import Sequential , Model
from keras.layers import Dense, Input, Add, Activation, BatchNormalization 
from keras.metrics import Precision , Recall
from keras import activations
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
def res_block(X, n_units):
    X_shortcut = X
    X = BatchNormalization()(X)
    X = Dense(n_units, activation = activations.linear)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(n_units, activation = activations.linear)(X)
    X = BatchNormalization()(X)
    X = Add()([X_shortcut, X])
    X = BatchNormalization()(X)
    
    X = Activation('relu')(X)
    
    return X
def res_model(n_units, n_blocks):
    X_input = Input((15,))
    X = Dense(n_units, activation = 'relu')(X_input)
    #X = res_block(X_input, n_units)
    
    for i in range(n_blocks):
        X = res_block(X, n_units)
        
    X = BatchNormalization()(X)
    X = Dense(1, activation = 'sigmoid')(X)
    model = Model(inputs = X_input, outputs = X)
    
    return model
    
MyModel = res_model(20, 50)
MyModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy',Precision(name='precision') , Recall(name='recall')])
MyModel.fit(X_chi, y, epochs = 200, batch_size = 1024)
nn_model = Sequential()
nn_model.add(Dense(50, input_dim = 15, activation = 'relu'))
nn_model.add(Dense(100, activation = 'relu'))
nn_model.add(Dense(100, activation = 'relu'))
nn_model.add(Dense(100, activation = 'relu'))
nn_model.add(Dense(100, activation = 'relu'))
nn_model.add(Dense(100, activation = 'relu'))
nn_model.add(Dense(50, activation = 'relu'))
nn_model.add(Dense(1, activation = 'sigmoid'))
nn_model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy',Precision(name='precision') , Recall(name='recall')])
nn_model.fit(X_chi, y, epochs= 300, batch_size= 4096)
nn_model_a = Sequential()
nn_model_a.add(Dense(20, input_dim = 15, activation = 'relu'))

for i in range(10):
    nn_model_a.add(Dense(50, activation = 'relu'))
    
nn_model_a.add(Dense(1, activation = 'sigmoid'))
nn_model_a.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy',Precision(name='precision') , Recall(name='recall')])
nn_model_a.fit(X_chi, y, epochs= 300, batch_size= 4096)
