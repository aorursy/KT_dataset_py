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
import pandas as pd

import numpy as np

import os



raw_data=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

raw_data
data=raw_data.copy()

pd.set_option('display.max_columns', None)

data

pd.set_option('display.max_columns', None)

data.describe(include='all')



nas=data.isnull().sum()

nas

data2=data.drop(['id','Unnamed: 32'],axis=1)

data2

new_data=data.drop(['id','Unnamed: 32'],axis=1)
corr=new_data.corr()

corr.style.background_gradient(cmap='coolwarm')
#radium_mean is highly correlated with perimeter_mean,area mean,area_worst,radius_worst,perimeter_worst

#I am keeing radus_mean only

#texture_mean is correlated with texture_worst

#Compactness_mean, concavity_mean and concave points_mean are correlated with each other.Therefore I only choose concavity_mean

#radius_se coreelated with perimeter_se and area_se I am keeping radius_se
drop_list = ['perimeter_mean','area_mean','compactness_mean','concave points_mean','radius_se','perimeter_se',

             'radius_worst','area_worst','perimeter_worst','compactness_worst',

             'texture_worst']

new_data1 = new_data.drop(drop_list,axis = 1 )        



new_data1.loc[(new_data.diagnosis == 'B'),'diagnosis']=0

new_data1.loc[(new_data.diagnosis == 'M'),'diagnosis']=1

new_data1

x=new_data1.drop(['diagnosis'],axis=1)

y=new_data1['diagnosis']

np.shape(y)

y.value_counts()

indices_to_remove = []

zero_counter=0

for i in range(new_data1.shape[0]):

    if new_data1.diagnosis[i] ==0:

        zero_counter += 1

        if zero_counter > 212:

            indices_to_remove.append(i)



np.shape(indices_to_remove)
x1=np.array(x)

y1=np.array(y)

x2= np.delete(x1,indices_to_remove, axis=0)

y2= np.delete(y1,indices_to_remove, axis=0)



from sklearn import preprocessing

x_scaled=preprocessing.scale(x2)
shuffled_indices=np.arange(x_scaled.shape[0])

np.random.shuffle(shuffled_indices)
x_shuffled=x_scaled[shuffled_indices]

y_shuffled=y2[shuffled_indices]

x_shuffled



samples_count = x_shuffled.shape[0]



train_samples_count = int(0.8 * samples_count)

validation_samples_count = int(0.1 * samples_count)





test_samples_count = samples_count - train_samples_count - validation_samples_count





x_train = x_shuffled[:train_samples_count]

y_train = y_shuffled[:train_samples_count].astype(np.float32)







x_validation = x_shuffled[train_samples_count:train_samples_count+validation_samples_count]

y_validation = y_shuffled[train_samples_count:train_samples_count+validation_samples_count].astype(np.float32)







x_test = x_shuffled[train_samples_count+validation_samples_count:]

y_test = y_shuffled[train_samples_count+validation_samples_count:].astype(np.float32)
type(y_train)

import tensorflow as tf



input_size = 19

output_size = 2



hidden_layer_size = 50

    



model = tf.keras.Sequential([

   

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), 

    

    tf.keras.layers.Dense(output_size, activation='softmax') 

])





model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])







batch_size = 100





max_epochs = 100





early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)





model.fit(x_train,

          y_train, 

          batch_size=batch_size,

          epochs=max_epochs, 

          callbacks=[early_stopping], 

          validation_data=(x_validation, y_validation),

          verbose = 2 

          )  
test_loss, test_accuracy = model.evaluate(x_test, y_test)