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
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.corr()['Outcome']
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
from sklearn.preprocessing import StandardScaler
se = StandardScaler()
X = se.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)

import tensorflow
from tensorflow import keras 
from keras import Sequential
from keras.layers import Dense,Dropout
import kerastuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Dense(32,activation='relu',input_dim = 8))
    model.add(Dense(1,activation='sigmoid'))
    optimizer = hp.Choice('optimizer',values = ['adam','sgd','rmsprop','adadelta'])
    model.compile(optimizer= optimizer,loss = 'binary_crossentropy',metrics = ['accuracy'])
    return (model)
tuner = kt.RandomSearch(build_model, objective = 'val_accuracy', max_trials = 10, directory = 'mydir', project_name = 'keras_tuer1')
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss', patience = 30)
help(EarlyStopping)
tuner.search(X_train,y_train,epochs = 200,validation_data = (X_test,y_test), callbacks = [early_stop])
tuner.get_best_hyperparameters()[0].values
model = tuner.get_best_models(num_models=1)[0]
model.evaluate(X_test,y_test)
def build_model(hp):
    model = Sequential()
    counter = 0
    for i in range(hp.Int('num_layers', min_value = 1, max_value = 10)):
        if counter == 0:
            model.add(
          Dense(hp.Int('units'+str(i),min_value = 8, max_value = 128, step = 2), 
                activation = hp.Choice('activation'+str(i), values = ['relu','tanh','sigmoid']),
                input_dim = 8))
            model.add(Dropout(hp.Choice('dropout'+str(i), values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))    
      
    else:
        model.add(Dense(
           hp.Int('units'+str(i),min_value = 8, max_value = 128, step = 2),
           activation = hp.Choice('activation'+str(i),values = ['relu','tanh','sigmoid'])
       ))
        model.add(Dropout(hp.Choice('dropout'+str(i), values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))    

    counter += 1
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = hp.Choice('optimizer', values = ['adam', 'rmsprop', 'sgd']) , loss = 'binary_crossentropy',
    metrics = ['accuracy'])
    return(model)
tuner = kt.RandomSearch(build_model, objective = 'val_accuracy',max_trials = 5,directory = 'mydir', project_name = 'keras_tue1' )
tuner.search_space_summary()
tuner.search(X_train,y_train, epochs = 200 ,batch_size = 16,  validation_data = (X_test,y_test))
tuner.get_best_hyperparameters()[0].values
tuner.results_summary()
best_model = tuner.get_best_models(num_models =1)[0]
best_model.evaluate(X_test,y_test, batch_size = 16)
from keras.callbacks  import History, EarlyStopping
early_stop = EarlyStopping(monitor = 'val_accuracy', patience  =  20)
history = History()

best_model.fit(X_train,y_train,epochs = 200, batch_size = 20,verbose = 1, validation_data = (X_test,y_test), callbacks = [history, early_stop])
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model"s training and validation loss across epochs')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['Train','validation'], loc ='right')
plt.show()
best_model.evaluate(X_test,y_test)