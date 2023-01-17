# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
from sklearn.preprocessing import MinMaxScaler,normalize

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

import imblearn as iml

from sklearn.decomposition import PCA
tf.__version__
test_ids = test_data['ID_code']
Target = train_data['target']
pca_df = normalize(train_data.drop(columns=['ID_code','target']),axis=1)

pca_test_df = normalize(test_data.drop(columns=['ID_code']),axis=1)



def _get_number_components(model, threshold):

    component_variance = model.explained_variance_ratio_

    explained_variance = 0.0

    component = 0

    for var in component_variance:

        explained_variance += var

        component += 1

        if (explained_variance >= threshold):

            break

    return component



### Get the optimal number of components

pca = PCA()

train_pca = pca.fit_transform(pca_df)

test_pca = pca.fit_transform(pca_test_df)

component = _get_number_components(pca, threshold=0.9)

component
# Implement PCA 

obj_pca = PCA(n_components=component)

X_pca = obj_pca.fit_transform(pca_df)

X_t_pca = obj_pca.fit_transform(pca_test_df)
# add the decomposed features in the train dataset

def _add_decomposition(df, decomp, ncomp, flag):

    for i in range(1, ncomp+1):

        df[flag+"_"+str(i)] = decomp[:,i-1]
_add_decomposition(train_data, X_pca, 90, 'pca')

_add_decomposition(test_data, X_t_pca, 90, 'pca')
del X_pca

del X_t_pca
idx = features = train_data.columns.values[2:202]

for df in [train_data, test_data]:

    df['sum'] = df[idx].sum(axis=1)

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurt(axis=1)

    df['med'] = df[idx].median(axis=1)
train_columns = train_data.drop(columns=['ID_code','target']).columns
features = [c for c in train_data.columns if c not in ['ID_code','target']]
"""

from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')

train_sampled_x,train_sampled_y = smote.fit_sample(train_data.drop(columns=['ID_code','target']),train_data['target'])

"""
"""

train_data = pd.DataFrame(train_sampled_x)

train_data.columns = train_columns

y = pd.Series(train_sampled_y)

"""
pipeline = Pipeline([('minmaxScaler', MinMaxScaler())])

train_data = pipeline.fit_transform(train_data.drop(columns=['ID_code','target']))

test_data = pipeline.transform(test_data.drop(columns=['ID_code']))
xtrain,xtest,ytrain,ytest = train_test_split(train_data,Target,test_size=0.20,random_state=2020)
del train_data
xtrain = pd.DataFrame(data=xtrain,columns=train_columns)

xtest = pd.DataFrame(data=xtest,columns=train_columns)
xtrain.shape
batch_size=128
import gc

gc.collect()


dataset_train = tf.data.Dataset.from_tensor_slices((dict(xtrain),ytrain))

dataset_train = dataset_train.shuffle(1000).repeat(10).batch(batch_size)

def train_inputfc():

    feature,label = dataset_train.make_one_shot_iterator().get_next()

    return feature,label
feature_colms = [ tf.feature_column.numeric_column(col) for col in train_columns]


dataset_val = tf.data.Dataset.from_tensor_slices((dict(xtest),ytest))

dataset_val = dataset_val.batch(batch_size)

def eval_inputfc():

    feature,label = dataset_val.make_one_shot_iterator().get_next()

    return feature,label
gc.collect()
model = tf.estimator.DNNClassifier(hidden_units=[1024,512,256]

                                   ,feature_columns=feature_colms,

                                   optimizer=lambda:tf.train.RMSPropOptimizer(learning_rate=tf.train.exponential_decay(

                                       learning_rate=0.083,

                                       global_step= tf.train.get_global_step(),

                                       decay_steps=1000,

                                       decay_rate=0.005)),

                                   dropout=0.5)
#model.train(input_fn=train_inputfc)
gc.collect()
#eval=model.evaluate(input_fn=eval_inputfc)
eval
test_data = pd.DataFrame(data=test_data,columns=train_columns)
dataset_test = tf.data.Dataset.from_tensor_slices(dict(test_data))

dataset_test = dataset_test.batch(batch_size)

def predict_fn():

    feature = dataset_test.make_one_shot_iterator().get_next()

    return feature
#prediction = model.predict(input_fn=predict_fn)
"""

def input_func(train_data):

    return tf.estimator.inputs.pandas_input_fn(

        x=train_data.drop(columns=['ID_code','target']),

        y=train_data['target'],

        batch_size=128,

        num_epochs=20,

        shuffle=True,

        queue_capacity=1000

    )

"""
#featcol = [tf.feature_column.numeric_column(feat) for feat in features]
"""

def predict_func(test_data):

    return tf.estimator.inputs.pandas_input_fn(

        x=test_data.drop(columns=['ID_code']),

        y=None,

        shuffle=False

    )

"""
"""

model = tf.estimator.DNNClassifier(hidden_units=[1024,664,1024,256],feature_columns=featcol,

                                  optimizer=lambda: tf.train.AdamOptimizer(

                                      learning_rate=tf.train.exponential_decay(

                                          learning_rate=0.001,

                                          global_step=tf.train.get_global_step(),

                                          decay_steps=10000,

                                          decay_rate=0.096)),

                                   loss_reduction=tf.losses.Reduction.SUM

                                  )

"""
#model.train(input_fn=input_func(train_data))
#prediction=model.predict(predict_func(test_data),yield_single_examples=True)
"""

predictionArray = np.zeros(len(test_data))

for x in range(len(test_data)):

    temp=next(prediction)

    predictionArray[x]=np.max(temp['probabilities'])

"""
#len(predictionArray)
"""

submissionTF = pd.DataFrame({'ID_code':test_ids,'target':predictionArray})

submissionTF.to_csv('submissionTF.csv',index=False)

submissionTF.head()

"""