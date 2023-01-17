import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)
CSV_COLUMNS = ['X', 'Y']
df_train = pd.read_csv('../input/dataTraining.txt', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv('../input/dataTraining.txt', header = None, names = CSV_COLUMNS)
df_test = pd.read_csv('../input/dataPrediction.txt', header = None, names = CSV_COLUMNS)
df_train.head()
df_test.head()
FEATURES = CSV_COLUMNS[0:len(CSV_COLUMNS)-1]
LABEL = CSV_COLUMNS[1]
tf_feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
tf_input_train=tf.estimator.inputs.pandas_input_fn(x = df_train,
                                                   y = df_train[LABEL],
                                                   shuffle = True,
                                                  num_epochs=1000)
tf_input_valid=tf.estimator.inputs.pandas_input_fn(x = df_valid,
                                                   y = df_valid[LABEL],
                                                   shuffle = False)
tf_input_test=tf.estimator.inputs.pandas_input_fn(x = df_test,
                                                   y = None,    
                                                   shuffle = False)

model = tf.estimator.LinearRegressor(feature_columns = tf_feature_columns)
model.train(input_fn =tf_input_train)
metrics = model.evaluate(input_fn = tf_input_valid)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
model2 = tf.estimator.DNNRegressor(hidden_units = [20,16, 8,3],
                                  feature_columns = tf_feature_columns)
model2.train(input_fn = tf_input_train)
metrics2 = model2.evaluate(input_fn = tf_input_valid)

print('RMSE DNN on dataset = {}'.format(np.sqrt(metrics2['average_loss'])))
print('RMSE Linear on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
predictions = model.predict(input_fn =tf_input_test)
t=[]
for items in predictions:
    t.append(items['predictions'])

Py=np.asarray(t )
Py=np.squeeze(Py)
print(Py)

df=df_train.as_matrix() 
aX=df[:,0:1]
aY=df[:,1:2]
###################################
predictions = model.predict(input_fn =tf.estimator.inputs.pandas_input_fn(x = df_train,
                                                   y = None,    
                                                   shuffle = False))
t=[]
for items in predictions:
    t.append(items['predictions'])
Py=np.asarray(t )
Py=np.squeeze(Py)
#######################################

predictions2 = model2.predict(input_fn =tf.estimator.inputs.pandas_input_fn(x = df_train,
                                                   y = None,    
                                                   shuffle = False))
t2=[]
for items in predictions2:
    t2.append(items['predictions'])
Py2=np.asarray(t2 )
Py2=np.squeeze(Py2)

#######################################
plt.subplot(121)
plt.scatter(aX,aY) 
plt.plot(aX, Py,color='r')
plt.title('Linear Regressor\n (RMSE={}) '.format(np.round(np.sqrt(metrics['average_loss']),4)))

plt.subplot(122)
plt.scatter(aX,aY) 
plt.plot(aX, Py2,color='r')
plt.title('DNN Regressor\n (RMSE={}) '.format(np.round(np.sqrt(metrics2['average_loss']),4)))
plt.show()


