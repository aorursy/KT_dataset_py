from datetime import datetime

import numpy as np

import numpy as numpy

import pandas as pd

import pylab

import calendar

from scipy import stats

import seaborn as sns

from sklearn import model_selection, preprocessing

from scipy.stats import kendalltau

import warnings

import matplotlib.pyplot as plt

import pandas

## Keras comes here

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout, BatchNormalization

from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import Adam

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

# Load train, Prop and sample

print('Loading train, prop and sample data')

train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

prop = pd.read_csv('../input/properties_2016.csv')

sample = pd.read_csv('../input/sample_submission.csv')

 

print('Fitting Label Encoder on properties')

for c in prop.columns:

    prop[c]=prop[c].fillna(-1)

    if prop[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(prop[c].values))

        prop[c] = lbl.transform(list(prop[c].values))

        

#Create df_train and x_train y_train from that

print('Creating training set:')

df_train = train.merge(prop, how='left', on='parcelid')



###########################################################

df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])

df_train["transactiondate_year"] = df_train["transactiondate"].dt.year

df_train["transactiondate_month"] = df_train["transactiondate"].dt.month

df_train['transactiondate_quarter'] = df_train['transactiondate'].dt.quarter

df_train["transactiondate"] = df_train["transactiondate"].dt.day



select_qtr4 = df_train["transactiondate_quarter"] == 4





###########################################



print('Fill  NA/NaN values using suitable method' )

#df_train.fillna(df_train.mean(),inplace = True)

df_train.fillna(-1.0)



#df_train =df_train[ df_train.logerror > -0.4005 ]

#df_train=df_train[ df_train.logerror < 0.412 ]



print('Create x_train and y_train from df_train' )

x_train_all = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)

y_train_all = df_train["logerror"]

y_train = y_train_all[~select_qtr4]

x_train = x_train_all[~select_qtr4]

x_valid = x_train_all[select_qtr4]

y_valid = y_train_all[select_qtr4]





#print("Bind x_train to float32:")

#x_train = x_train.values.astype(np.float32, copy=False)





y_mean = np.mean(y_train)

print(x_train.shape, y_train.shape)

train_columns = x_train.columns



for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)

# Create df_test and test set

print('Creating df_test  :')

sample['parcelid'] = sample['ParcelId']



print("Merge Sample with property data :")

df_test = sample.merge(prop, on='parcelid', how='left')





########################

df_test["transactiondate"] = pd.to_datetime(df_train["transactiondate"])

df_test["transactiondate_year"] = df_test["transactiondate"].dt.year

df_test["transactiondate_month"] = df_test["transactiondate"].dt.month

df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter

df_test["transactiondate"] = df_test["transactiondate"].dt.day     



#################################





x_test = df_test[train_columns]



print('Shape of x_test:', x_test.shape)

print("Preparing x_test:")

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)

  
from sklearn.preprocessing import Imputer

imputer= Imputer()

imputer.fit(x_train.iloc[:, :])

x_train = imputer.transform(x_train.iloc[:, :])

imputer.fit(x_valid.iloc[:, :])

x_valid = imputer.transform(x_valid.iloc[:, :])

imputer.fit(x_test.iloc[:, :])

x_test = imputer.transform(x_test.iloc[:, :])



#########################Standard Scalar##############



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

x_valid = sc.fit_transform(x_valid)

x_val = np.array(x_valid)

y_val = np.array(y_valid)

len_x=int(x_train.shape[1])

print("len_x is:",len_x)

#########################################################################

####################ANN Starts here#



nn = Sequential()

nn.add(Dense(units = 400 , kernel_initializer = 'normal', activation = 'relu', input_dim = len_x))

nn.add(Dropout(.36))

nn.add(Dense(units = 160 , kernel_initializer = 'normal', activation = 'relu'))

nn.add(BatchNormalization())

nn.add(Dropout(.6))

nn.add(Dense(units = 64 , kernel_initializer = 'normal', activation = 'relu'))

nn.add(BatchNormalization())

nn.add(Dropout(.48))

nn.add(Dense(units = 28, kernel_initializer = 'normal', activation = 'relu'))

nn.add(BatchNormalization())

nn.add(Dropout(.48))

nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=1e-3))
nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = 100, verbose=2, 

       validation_data=(x_val,y_val))
print("x_test.shape:",x_test.shape)

y_pred_ann = nn.predict(x_test)



#######################################################################################



print( "\nPreparing results for write :" )
y_pred = y_pred_ann.flatten()



output = pd.DataFrame({'ParcelId': prop['parcelid'].astype(np.int32),

        '201610': y_pred, '201611': y_pred, '201612': y_pred,

        '201710': y_pred, '201711': y_pred, '201712': y_pred})

# set col 'ParceID' to first col

cols = output.columns.tolist()

cols = cols[-1:] + cols[:-1]

output = output[cols]



print( "\nWriting results to disk:" )

output.to_csv('Only_ANN_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)



print( "\nFinished!" )
