# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.



boston = pd.read_csv('../input/train.csv')

boston_test = pd.read_csv('../input/train.csv')
def is_num(x):

    x = boston[x]

    try:

        if x.dtype == np.int64 or x.dtype == np.float64:

            return True

        else:

            return False

    except AttributeError:

        return False

    

def is_complete(x):

    if x == 'SalePrice': return False

    return pd.isnull(boston[x]).sum() + pd.isnull(boston_test[x]).sum() == 0



categorical = list(filter(lambda x: not is_num(x) and is_complete(x), list(boston)))

continuous = list(filter(lambda x: is_num(x) and is_complete(x), list(boston)))

#continuous = list(filter(lambda x: not x == 'SalePrice', continuous))

#columns = list(boston)
import tensorflow as tf



#from sklearn import datasets, metrics, preprocessing

cat_layers = []

real_layers = []

deep = []

wide = []



for x in categorical:

    cat_layers.append(tf.contrib.layers.sparse_column_with_keys(x, keys=set(boston[x]), combiner='sqrtn'))    

for x in continuous:

    real_layers.append(tf.contrib.layers.real_valued_column(x, dimension=1, dtype=tf.float32))

for x in cat_layers:

    deep.append(tf.contrib.layers.embedding_column(x,dimension=8))

    wide.append(x)

for x in real_layers:

    deep.append(x)

 

    

    
estimator = tf.contrib.learn.DNNLinearCombinedRegressor(

    model_dir = "model",

    # wide settings

    linear_feature_columns=wide,

    linear_optimizer=tf.train.FtrlOptimizer(

                                        learning_rate=0.5,

                                        l1_regularization_strength=0.001,

                                        l2_regularization_strength=0.001),

    # deep settings

    dnn_feature_columns=deep,

    dnn_hidden_units=[256, 128, 64],

    dnn_optimizer=tf.train.ProximalAdagradOptimizer(

                                        learning_rate=0.05,

                                        l1_regularization_strength=0.001,

                                        l2_regularization_strength=0.001)

)
def input_fn(df, train=True):



    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])

                        for k in continuous}



    categorical_cols = {k: tf.SparseTensor(

        indices=[[i, 0] for i in range(df[k].size)],

        values=df[k].values,

        dense_shape=[df[k].size, 1])

                        for k in categorical}



   

    feature_cols = continuous_cols.copy()

    feature_cols.update(categorical_cols)



    label = None

    if train:

        label = tf.constant(df['SalePrice'].values)



    return feature_cols, label



def train_input_fn():

    return input_fn(boston)



def eval_input_fn():

    return input_fn(boston_test, train=False)
estimator.fit(input_fn=train_input_fn, steps=10000)

print(' ')
results = estimator.predict(input_fn=eval_input_fn)



out = list(zip(list(boston_test['Id']),list(results)))

cols = ['Id', 'SalePrice']

df_out = pd.DataFrame(out, columns=cols)



df_out.to_csv(path_or_buf='pred.csv', index=False)