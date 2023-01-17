# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
tf.logging.set_verbosity(tf.logging.ERROR)
import os
print(os.listdir("../input/cylinderds3"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/cylinderds3/CylinderDS2.csv")

df.head()
df["r3"] = df['radius']*df['radius']*df['radius']
df["h3"] = df['Height']*df['Height']*df['Height']
df["rh2"] = df['radius']*df['Height']*df['Height']
df["r2h"] = df['radius']*df['radius']*df['Height']


df_train, df_valid = train_test_split(df, test_size=0.2)
df_train, df_test = train_test_split(df_train, test_size=0.25)

#df_train = df[0:400]
#df_valid = df[400:450]
#df_test = df[450:500]

tf_feature_columns = [tf.feature_column.numeric_column('radius'),
                      tf.feature_column.numeric_column('Height'),
                      tf.feature_column.numeric_column('r3'),
                      tf.feature_column.numeric_column('h3'),
                      tf.feature_column.numeric_column('rh2'),
                      tf.feature_column.numeric_column('r2h')
                     ]

print(tf_feature_columns)
tf_input_train=tf.estimator.inputs.pandas_input_fn(x = df_train[['radius','Height','r3','h3','r2h','rh2']],
                                                   y = df_train['volume'],
                                                    #batch_size = 20,
                                                    num_epochs = 3000,
                                                    shuffle = True#,
                                                    #queue_capacity = 100
                                                  )
model = tf.estimator.LinearRegressor(feature_columns = tf_feature_columns)
#model = tf.estimator.DNNRegressor(hidden_units = [10,2,1],
#                                  feature_columns = tf_feature_columns)
model.train(input_fn =tf_input_train)
tf_input_valid=tf.estimator.inputs.pandas_input_fn(x = df_valid,
                                                   y = df_valid['volume'],
                                                   shuffle = False)
tf_input_test=tf.estimator.inputs.pandas_input_fn(x = df_test,
                                                   y = df_test['volume'],
                                                   shuffle = False)

predictions = model.predict(input_fn =tf_input_test)
t=[]
for items in predictions:
    t.append(items['predictions'])

Py=np.asarray(t) 
df_test['Predict']=Py
df_test.head()
metrics = model.evaluate(input_fn = tf_input_train)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))

metrics = model.evaluate(input_fn = tf_input_valid)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))

metrics = model.evaluate(input_fn = tf_input_test)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))