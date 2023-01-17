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
# Seed value

# Apparently you may use different seed values at each stage

seed_value= 1



# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

import os

os.environ['PYTHONHASHSEED']=str(seed_value)



# 2. Set the `python` built-in pseudo-random generator at a fixed value

import random

random.seed(seed_value)



# 3. Set the `numpy` pseudo-random generator at a fixed value

import numpy as np

np.random.seed(seed_value)



# 4. Set the `tensorflow` pseudo-random generator at a fixed value

import tensorflow as tf

tf.random.set_seed(seed_value)

# for later versions: 

# tf.compat.v1.set_random_seed(seed_value)



# 5. Configure a new global `tensorflow` session

from keras import backend as K

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

# K.set_session(sess)

# for later versions:

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)
from tqdm import tqdm

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt



#Get keras packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import BatchNormalization

from keras.layers import Concatenate

from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import EarlyStopping



import pandas as pd

test = pd.read_csv("../input/righthand-recommendation-engine-draft-v0/Test data.csv")

train = pd.read_csv("../input/righthand-recommendation-engine-draft-v0/Training data.csv")
train.head()
def convert_cols(df,cols=[]):

    if cols != [] and isinstance(df, pd.DataFrame):

        for col in cols:

            df.loc[:,col]= df.loc[:,col].astype('category')

        df = pd.get_dummies(df,columns=cols,drop_first=False)

    elif cols == [] and isinstance(df, pd.Series):

        df.astype('category')

        df = pd.get_dummies(df,columns=cols,drop_first=False)

    elif cols == [] and isinstance(df, pd.DataFrame):

        print("Please specify columns of DataFrame to be converted to categorical type.")

        return

    else:

        print("Please ensure data is a DataFrame or Series.")

        return

    return df;



x_train = convert_cols(train.drop(['User_id','Behaviour_id','Recommendation'], axis=1),['Behaviour_type','Behaviour_subtopic','Redirect_source','User_redirected'])

y_train = convert_cols(train['Recommendation'])

x_test = convert_cols(test.drop(['User_id','Behaviour_id','Recommendation'], axis=1),['Behaviour_type','Behaviour_subtopic','Redirect_source','User_redirected'])

y_test = convert_cols(test['Recommendation'])

print(x_train.memory_usage())

print('\nTotal memory usage: ',x_train.memory_usage(index=True).sum())
y_train.head()
x_train_array = x_train.to_numpy()

print(x_train_array.shape)

y_train_array = y_train.to_numpy()

print(y_train_array.shape)
#Turn keras model into sklearn estimator

def create_model(optimizer='adam', activation='relu', nl=1, nn=256): # Function that creates our Keras model

    model = Sequential()

    model.add(Dense(16, input_shape=(10,), activation=activation))

    # Add as many hidden layers as specified in nl

    for i in range(nl):

        # Layers have nn neurons

        model.add(Dense(nn, activation=activation))

    model.add(BatchNormalization())

    model.add(Dense(9, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=["accuracy"])

    return model

from keras.wrappers.scikit_learn import KerasClassifier # Import sklearn wrapper from keras

model = KerasClassifier(build_fn=create_model, epochs=6, batch_size=16) # Create a model as a sklearn estimator



#Random search on Keras models

# Define a series of parameters

params = dict(optimizer=['sgd','adam'], epochs=[3], batch_size=[5, 10, 20], activation=['relu','tanh'], nl=[1, 2, 9], nn=[128,256,1000])

# Create a random search cv object and fit it to the data

random_search = RandomizedSearchCV(model, param_distributions=params, cv=3)

random_search_results = random_search.fit(x_train, y_train)

# Print results

print(f"Best: {random_search_results.best_score_} using {random_search_results.best_params_}")
model = create_model(optimizer='sgd', activation='tanh', nl=9, nn=128)

monitor_val_acc = EarlyStopping(monitor='val_accuracy', patience=200)

history = model.fit(x_train_array,y_train_array, epochs =1000, batch_size=10,validation_split=0.2, callbacks=[monitor_val_acc])

model.evaluate(x_train.iloc[10].values.reshape(-1,10),y_train.iloc[10].values.reshape(-1,9))
model = create_model(optimizer='adam', activation='tanh', nl=1, nn=256)

monitor_val_acc = EarlyStopping(monitor='val_accuracy', patience=200)

history = model.fit(x_train_array,y_train_array, epochs =1000, batch_size=10,validation_split=0.2, callbacks=[monitor_val_acc])

model.evaluate(x_train.iloc[10].values.reshape(-1,10),y_train.iloc[10].values.reshape(-1,9))
model.evaluate(x_test,y_test)
def plot_loss(loss,val_loss):

  plt.figure()

  plt.plot(loss)

  plt.plot(val_loss)

  plt.title('Model loss')

  plt.ylabel('Loss')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Test'], loc='upper right')

  plt.show()

    

def plot_accuracy(accuracy,val_accuracy):

  plt.figure()

  plt.plot(accuracy)

  plt.plot(val_accuracy)

  plt.title('Model accuracy')

  plt.ylabel('Accuracy')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Test'], loc='upper right')

  plt.show()

    

plot_loss(history.history['loss'], history.history['val_loss']) # Plot train vs test loss during training

plot_accuracy(history.history['accuracy'], history.history['val_accuracy']) # Plot train vs test accuracy during training
y_test.head()
results = pd.DataFrame(model.predict(x_test), columns=['attachment quickly','attachment slowly','credentials quickly','credentials slowly','link quickly','link slowly','none','spear phishing','whaling'])

results['PREDICTION'] = results.idxmax(axis=1)

results
test.columns
#Comparison of Actual VS Prediction



comparison = pd.DataFrame(test['Recommendation'])

comparison['Prediction'] = results['PREDICTION']

comparison.rename(columns={"Recommendation": "Actual"}, inplace=True)

comparison
results.columns[0]
model.metrics_names