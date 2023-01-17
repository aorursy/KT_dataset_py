#Author: ram1510(guhan).This notebook is licensed under Apache 2.0 open source license
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
#import necessary libraries and modules
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Create a dataframe 
raw_data=pd.read_csv('/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv')
raw_data.shape
#check for consistency of the data
raw_data.describe()
#check for data redunduncy and correlation
mod_data=raw_data.loc[:,['ambient','coolant','u_d','u_q','motor_speed','i_d','i_q','stator_tooth']]
mod_data.corr()#Pearson's correlation
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(mod_data.corr(),annot=True)
#a collection of samples from data
final_data=raw_data.loc[:10000,['ambient','coolant','motor_speed','i_d','u_d','u_q','i_q','stator_tooth']]
#getting first 5 samples
final_data.head(n=5)
#spliting for the training and testing phase
Y=final_data.pop('stator_tooth')
X=final_data
X_train,X_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# define a neural network function
def Neural_Network():
     model=keras.Sequential([layers.Dense(64,input_shape=[len(X_train.columns)],activation='relu'),
                            layers.Dense(64,activation='relu'),
                            layers.Dense(1)])
     optimizer=tf.keras.optimizers.RMSprop(0.001)
     model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse',tf.keras.metrics.RootMeanSquaredError()])
     return model
#instanciate the function
model=Neural_Network()
#model training with batch size of 100 
epoch=100
history=model.fit(X_train,Y_train,epochs=epoch,batch_size=100,validation_split=0.2,verbose=0,callbacks=[tfdocs.modeling.EpochDots()])
#prediction on the model using the test data
test_pred=model.predict(X_test).flatten()
#predicted data
pd.DataFrame({'Actual':y_test,'Predicted':test_pred})
#Histogram plot of error
error=test_pred-y_test
plt.hist(error,bins=20)
plt.title('Histogram of error curve')
plt.xlabel('Error')
plt.ylabel('No.of.samples')
# metrics to evaluate the effectiveness of the model
mae=metrics.mean_absolute_error(y_test,test_pred)#mean absolute error
mse=metrics.mean_squared_error(y_test,test_pred)#mean squared error
r2_score=metrics.r2_score(y_test,test_pred)#R_squared value
print('The mean absolute error is :' f'{mae}',
     'The Root mean square value is :'f' {np.sqrt(mse)}',
      'The R_squared value is :'f'{r2_score}',sep='\n')