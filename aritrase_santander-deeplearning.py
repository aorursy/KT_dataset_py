# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train rows and columns : ", train_df.shape)
print("Test rows and columns : ", test_df.shape)
y = train_df['target'].copy()
y = np.log1p(y)
X = train_df.drop(labels=['target','ID'],axis=1)
X_test = test_df.drop(labels=['ID'],axis=1)
#X.head()
print(type(y))
print(X.shape)
print(X_test.shape)
# Function to find out % of missing values in each column
def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns
# Capturing the columns with more than 95% of missing values
mis_val_table_ren_columns = missing_values_table(X)
#print(mis_val_table_ren_columns)
nan_col = list(mis_val_table_ren_columns[mis_val_table_ren_columns['% of Total Values']> 95].index)
print(nan_col) # no missing values column
#Finding out the no variation columns
# 256 cols with no variation
for col in X.columns.values:
    if(len(X[col].unique()) == 1):
        nan_col.append(col)
print(len(nan_col))
# Drop these columns = missing values + no variance column
X.drop(nan_col,inplace = True ,axis=1)
X_test.drop(nan_col,inplace = True ,axis=1)
print(X.shape)
print(X_test.shape)
print(type(X))
print(type(X_test))
# Feature Scaling - StandardScaler
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)
"""
# Feature Scaling - normalize
from sklearn.preprocessing import normalize
X = normalize(X)
X_test = normalize(X_test)
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1024, input_dim= 4735, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512,kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(200,kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(150,kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(100,kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(50,kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation = 'linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=500, verbose=0)
#kfold = KFold(n_splits=5, random_state=seed)
#results = cross_val_score(estimator, X, y, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
estimator.fit(
        X, 
        y, 
        epochs=500,
        #validation_data=(X_val, y_val),
        #verbose=2
        #callbacks=callbacks,
        #shuffle=True
    )
pred_nn = np.expm1(estimator.predict(X_test))
pred_nn
# Making a submission file #
sub_df = pd.DataFrame({"ID":test_df["ID"].values})
sub_df["target"] = pred_nn
sub_df.to_csv("submission_dl_normalize.csv", index=False)
