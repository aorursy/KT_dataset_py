# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Input data

train_id = pd.read_csv('/kaggle/input/train_identity.csv')

test_id = pd.read_csv('/kaggle/input/test_identity.csv')

test_tx = pd.read_csv('/kaggle/input/test_transaction.csv')

train_tx = pd.read_csv('/kaggle/input/train_transaction.csv')

# sample = pd.read_csv('/kaggle/input/sample_submission.csv')
# train_id.head()
# train_tx.loc[train_tx.isFraud==1].head()
# train_tx.loc[train_tx.isFraud==0].head()
device = ['Windows', 'iOS Device', 'MacOS']

train_id.loc[~train_id.DeviceInfo.isin(device), 'DeviceInfo'] = 'other'

test_id.loc[~test_id.DeviceInfo.isin(device), 'DeviceInfo'] = 'other'
#https://stackoverflow.com/questions/43311555/how-to-drop-column-according-to-nan-percentage-for-dataframe

train_tx = train_tx.loc[:, train_tx.isnull().sum() < 0.85*train_tx.shape[0]]
train_id = train_id.loc[:, train_id.isnull().sum() < 0.85*train_id.shape[0]]
train_tx.shape, train_id.shape
train_df = pd.merge(left=train_id,right=train_tx, how = 'right', left_on='TransactionID', right_on='TransactionID')

del train_id

del train_tx
train_df.shape
train_df['isFraud'].value_counts()
df_n = train_df[train_df['isFraud'] ==0].sample(30000,random_state = 2)

df_y = train_df[train_df['isFraud'] == 1]

train_df = pd.concat([df_n,df_y])

train_df['isFraud'].value_counts()

del df_n

del df_y
train_df.shape
# train_id.dtypes
train_df['DeviceType'].value_counts()
# device = ['Windows', 'iOS Device', 'MacOS']

# train_df.loc[~train_df.DeviceInfo.isin(device), 'DeviceInfo'] = 'other'
# train_df['DeviceInfo'].value_counts()
# train_tx.dtypes
# train_df['P_emaildomain'].value_counts()
# train_df['R_emaildomain'].value_counts()
categorical = ['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2', 'DeviceType', 'DeviceInfo',

               'P_emaildomain','R_emaildomain','M1','M2','M3','M4','M5','M6','M7','M8','M9']

numerical = list(set(list(train_df.columns))-set(['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2', 'DeviceType', 'DeviceInfo',

                                                  'P_emaildomain','R_emaildomain','M1','M2','M3','M4','M5','M6','M7','M8','M9']))
target = ['isFraud']
X = train_df[numerical].copy()



X.fillna(0,inplace = True)



Y = train_df[categorical].copy()



Y.fillna('NA',inplace = True)
train_df = pd.concat([X,Y],axis = 1)

del X

del Y
train_df = train_df.loc[:,~train_df.columns.duplicated()]
corr_matrix = train_df[numerical].corr()
var_list = corr_matrix['isFraud'].sort_values()[:85]

del corr_matrix
# var_list
num_var_list = list(var_list.index)

cat_var_list_prev = ['card4','card6','M4','ProductCD','DeviceType', 'DeviceInfo', 'P_emaildomain','R_emaildomain']
import tensorflow as tf

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing
##Splitting data into train, test & validation

from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(train_df, test_size=0.15, random_state=0)

X_val, X_test = train_test_split(X_val, test_size=0.5, random_state=0)

del train_df
X_train.shape, X_val.shape, X_test.shape
# Deep features

# Defining tensor type for numerical attributes

numerical_features = [tf.feature_column.numeric_column(key = k) for k in num_var_list]



# Defining tensor type for categorical attributes

from tensorflow.contrib import layers



cat_features1 = [tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key = k, hash_bucket_size=100), dimension=4) for k in cat_var_list_prev]



features = num_var_list + cat_var_list_prev

DEEP_FEATURES= numerical_features + cat_features1

LABEL_NAME = target
# Cross columns

_HASH_BUCKET_SIZE=1000

crossed_columns = [

     tf.feature_column.crossed_column(

         ["DeviceType","DeviceInfo"], hash_bucket_size=_HASH_BUCKET_SIZE),

     tf.feature_column.crossed_column(

         ["P_emaildomain", "R_emaildomain"],

         hash_bucket_size=_HASH_BUCKET_SIZE)

]
# Wide features

from tensorflow.contrib import layers



cat_var_wide = list(set(cat_var_list_prev) - set(['DeviceType', 'DeviceInfo','P_emaildomain','R_emaildomain']))

cat_features2 = [tf.feature_column.categorical_column_with_hash_bucket(key = k, hash_bucket_size=100) for k in cat_var_wide]



WIDE_FEATURES = numerical_features + cat_features2  + crossed_columns
# initializer function for train data 

def train_input_fn(df, batch_size = 256):

    #1. Convert dataframe into correct (features,label) format for Estimator API

    dataset = tf.data.Dataset.from_tensor_slices(tensors = (dict(df[features]), df[LABEL_NAME]))

    

    # Note:

    # If we returned now, the Dataset would iterate over the data once  

    # in a fixed order, and only produce a single element at a time.

    

    #2. Shuffle, repeat, and batch the examples.

    dataset = dataset.shuffle(buffer_size = 1000).repeat(count = None).batch(batch_size = batch_size)

   

    return dataset



# initializer function for validate data 

def eval_input_fn(df, batch_size = 256):

    #1. Convert dataframe into correct (features,label) format for Estimator API

    dataset = tf.data.Dataset.from_tensor_slices(tensors = (dict(df[features]), df[LABEL_NAME]))

    

    #2.Batch the examples.

    dataset = dataset.batch(batch_size = batch_size)

   

    return dataset



# initializer function for test data 

def predict_input_fn(df, batch_size = 256):

    #1. Convert dataframe into correct (features) format for Estimator API

    dataset = tf.data.Dataset.from_tensor_slices(tensors = dict(df[features])) # no label



    #2.Batch the examples.

    dataset = dataset.batch(batch_size = batch_size)

   

    return dataset
# DNN Classifier object

print("No. of Layers 3")

print("No. of Units [256,128, 64]")

layer1 = 128

layer2 = 64

layer3 = 32

#layer4 = 64

model = tf.estimator.DNNLinearCombinedClassifier(

    linear_feature_columns=WIDE_FEATURES,

    linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01),

    dnn_feature_columns=DEEP_FEATURES,

    dnn_hidden_units=[layer1, layer2],

    dnn_activation_fn=tf.nn.relu, 

#     dnn_dropout=0.7,

    dnn_optimizer=tf.train.ProximalAdagradOptimizer(

      learning_rate=0.008),

#     l1_regularization_strength=0.001,

#     l2_regularization_strength=0.001),

    n_classes=2,

    config = tf.estimator.RunConfig(tf_random_seed = 1), # for reproducibility

    batch_norm=True   #loss_reduction=tf.losses.Reduction.SUM

#    ,model_dir = 'mar/'

)
# Executing the DNN model on train data

tf.reset_default_graph()

import logging

tf.logging.set_verbosity(tf.logging.INFO) # so loss is printed during training

model.train(input_fn = lambda: train_input_fn(df = X_train), steps = 3500)
print(model.evaluate(input_fn = lambda: eval_input_fn(df = X_train)))
eval = model.evaluate(input_fn = lambda: eval_input_fn(df = X_val))

print(eval)
print(model.evaluate(input_fn = lambda: eval_input_fn(df = X_test)))
##### Function to calculate various evaluation metrics

def acc_matrix(model, data, features, LABEL_NAME):



	test = model.predict(input_fn = lambda: predict_input_fn(df = data[features]))



	l=[]

	for i in test:

	   l.append(int(i["classes"][0])) 



	model_predict=np.array([i for i in l]) 



	model_actual=np.array(data[LABEL_NAME]) 



	print(confusion_matrix(model_actual,model_predict) )





# Accuracy matrix for train data 

print("For Train")

acc_matrix(model, X_train, features, LABEL_NAME)



# Accuracy matrix for validate data 

print("For Validate")

acc_matrix(model, X_val, features, LABEL_NAME)



# Accuracy matrix for test data 

print("For Test")

acc_matrix(model, X_test, features, LABEL_NAME)
# Predict dataset

test_df = pd.merge(left=test_id,right=test_tx,how = 'right',left_on='TransactionID', right_on='TransactionID')



# test_df.loc[~test_df.DeviceInfo.isin(device), 'DeviceInfo'] = 'other'



X = test_df[num_var_list].copy()



X.fillna(0,inplace = True)



Y = test_df[cat_var_list_prev].copy()



Y.fillna('NA',inplace = True)



test_df = pd.concat([X,Y],axis = 1)

test_df = test_df.loc[:,~test_df.columns.duplicated()]

del X

del Y
test_df.shape
# Prediction metrics

results = model.predict(input_fn = lambda: predict_input_fn(df = test_df))
l1=[]

for i in results:

    l1.append((i["probabilities"][1]))

#     prob=np.array([i for i in l1])



# l2=[]

# for probability_of_1 in l1:

#   if probability_of_1 > 0.5:

#     l2.append(int(1))

#   else:

#     l2.append(int(0))
# l2
transaction_id = pd.merge(left=test_id,how = 'right',right=test_tx, left_on='TransactionID', right_on='TransactionID')['TransactionID']

del test_id

del test_tx

del test_df
len(transaction_id)
len(l1)
v1 = pd.DataFrame({'TransactionID':list(transaction_id),'isFraud':l1})

v1.to_csv('submission2.csv',index = False)
v1.head()