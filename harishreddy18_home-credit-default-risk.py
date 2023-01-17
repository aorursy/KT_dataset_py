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
#reading the data

train_df=pd.read_csv('../input/application_train.csv')

test_df=pd.read_csv('../input/application_test.csv')
print(train_df.shape)

train_df.head()

train_len=len(train_df)
print(test_df.shape)

test_df.head()
train_df.describe()
test_df.describe()
train_df.isnull().sum()
train_df=pd.concat([train_df,test_df])

train_df.head()
train_df.shape
#label encoding features with less than 3 categories

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in train_df.columns:

    if train_df[col].dtype=='object' and int(len(train_df[col].value_counts()))<3:

        print(col)

        le.fit(train_df[col].astype(str))

        train_df[col]=le.transform(train_df[col].astype(str))
train_df.head()
#onehotencoding features with more than 2 categories

train_df=pd.get_dummies(train_df)

train_df.head()
#retrieving back train and test df from concatenated df

test_df=train_df.iloc[train_len:,:]

test_df.head()

train_df=train_df.iloc[0:train_len,:]

train_df.head()
train_df.shape
test_df.shape
burba_df=pd.read_csv(r'../input/bureau_balance.csv')

bur_df=pd.read_csv(r'../input/bureau.csv')
bur_df.head()
bur_g_skidc=bur_df.drop('SK_ID_BUREAU',axis=1).groupby('SK_ID_CURR',as_index=False)
bur_g_skidc=bur_g_skidc.agg(['count','mean','min','max'])
bur_g_skidc.columns
bur_g_skidc=bur_g_skidc.reset_index()

bur_g_skidc.head()
bur_g_skidc.columns
# List of column names

columns = ['SK_ID_CURR']

# Iterate through the variables names

for var in bur_g_skidc.columns.levels[0]:

    # Skip the id name

    if var != 'SK_ID_CURR':

        

        # Iterate through the stat names

        for stat in bur_g_skidc.columns.levels[1][:-1]:

            # Make a new column name for the variable and stat

            columns.append('bureau_%s_%s' % (var, stat))
bur_g_skidc.columns=columns
bur_g_skidc.head()
train_df=train_df.merge(bur_g_skidc,how='left',on='SK_ID_CURR')

train_df.head()
train_df.shape
bur_c_df=bur_df.select_dtypes(include=['object'])

bur_c_df['SK_ID_CURR']=bur_df['SK_ID_CURR']
bur_c_df=pd.get_dummies(bur_c_df)

bur_c_df.head()
bur_c_df=bur_c_df.groupby('SK_ID_CURR',as_index=False)

bur_c_df=bur_c_df.agg(['count','mean'])
bur_c_df=bur_c_df.reset_index()

bur_c_df.head()
# List of column names

columns = ['SK_ID_CURR']

# Iterate through the variables names

for var in bur_c_df.columns.levels[0]:

    # Skip the id name

    if var != 'SK_ID_CURR':

        

        # Iterate through the stat names

        for stat in bur_c_df.columns.levels[1][:-1]:

            # Make a new column name for the variable and stat

            columns.append('bureau_%s_%s' % (var, stat))
bur_c_df.columns=columns

bur_c_df.head()
train_df=train_df.merge(bur_c_df,how='left',on='SK_ID_CURR')

train_df.head()
#dropping unwanted columns and creating features and lables

X_train=train_df.drop(['SK_ID_CURR','TARGET'],axis=1)

Y_train=train_df['TARGET']

#X_test=test_df.drop(['SK_ID_CURR','TARGET'],axis=1)

print(X_train.shape)
#visualising class imbalance in TARGET column

import matplotlib.pyplot as plt

plt.hist(Y_train)
#standard scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

#X_test=sc.transform(X_test)
#imputing missing values

from fancyimpute import IterativeImputer

ii=IterativeImputer(n_iter=1,random_state=0)

X_t=ii.fit_transform(X_train)

#X_test=ii.transform(X_test)

#converting label series to array

Y_t=Y_train.iloc[:].values
from keras.layers import Dense,BatchNormalization,Activation,Input

from keras.optimizers import Adam

from keras.models import Model

import keras.backend as K
#trying out various models using stratified kfold cross validation

n=X_train.shape[1]

nh1=500

nh2=200

nh3=50

X=Input(shape=(n,))

Y=Dense(nh1)(X)

Y=BatchNormalization()(Y)

Y=Activation('relu')(Y)

Y=Dense(nh2)(Y)

Y=BatchNormalization()(Y)

Y=Activation('relu')(Y)

Y=Dense(nh3)(Y)

Y=BatchNormalization()(Y)

Y=Activation('relu')(Y)

Y=Dense(1)(Y)

Y=Activation('sigmoid')(Y)

model=Model(X,Y)
opt = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,decay=0.01)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3,random_state=1)

troc=[]

vroc=[]

for train_index, test_index in skf.split(X_t,Y_t):

    X_train, X_val = X_t[train_index], X_t[test_index]

    Y_train, Y_val = Y_t[train_index], Y_t[test_index]

    model.fit(X_train,Y_train, epochs=50, batch_size=64,validation_data=(X_val,Y_val))

    from sklearn.metrics import roc_auc_score

    Y_tp=model.predict(X_train)

    rosc=roc_auc_score(Y_train,Y_tp)

    troc.append(rosc)

    print('train auc: ',rosc)

    Y_vp=model.predict(X_val)

    rosc=roc_auc_score(Y_val,Y_vp)

    vroc.append(rosc)

    print('test auc :',rosc)
from statistics import mean

print('training auc: ',mean(troc))

print('validation auc: ',mean(vroc))
test_df=test_df.merge(bur_g_skidc,how='left',on='SK_ID_CURR')

test_df=test_df.merge(bur_c_df,how='left',on='SK_ID_CURR')

X_test=test_df.drop(['SK_ID_CURR','TARGET'],axis=1)

X_test=sc.transform(X_test)

X_test=ii.transform(X_test)
Y_pred=model.predict(X_test)

p_df=pd.DataFrame(Y_pred)

p_df.rename(columns={0:'TARGET'},inplace=True)

#p_df['target']=p_df['target'].astype(float)

output=pd.DataFrame({'SK_ID_CURR':test_df['SK_ID_CURR'],'TARGET':p_df['TARGET']})

#print(output)

output.to_csv('out.csv',index=False)
print(output)