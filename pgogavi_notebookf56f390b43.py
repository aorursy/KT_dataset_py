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
path = '/kaggle/input/titanic/'

train = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')
train
test
train.info()
train = train.fillna({'Age':train['Age'].mean()})

X_df = train.drop(columns=['PassengerId','Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'])

y_df = train['Survived']
X_df = X_df.replace('male', 0)

X_df = X_df.replace('female', 1)
from sklearn.preprocessing import *

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_df.values, y_df.values, test_size=0.25, shuffle=True, random_state=0)
print(X_train)
X_train.shape
from keras.layers import Input

from keras.layers.core import *

from keras.optimizers import Adam, Adagrad, RMSprop, SGD

from keras.models import Model

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

import keras.backend as K

import optuna
def objective(trial):

    K.clear_session()

    

    activation = trial.suggest_categorical('activation',['relu','tanh','linear'])

    optimizer = trial.suggest_categorical('optimizer',['adam','rmsprop','adagrad', 'sgd'])



    

    num_hidden_layer = trial.suggest_int('num_hidden_layer',1,5,1)

    num_hidden_unit = trial.suggest_int('num_hidden_unit',10,100,10)

    



    

    learning_rate = trial.suggest_loguniform('learning_rate', 0.00001,0.1)

    if optimizer == 'adam':

      optimizer = Adam(learning_rate=learning_rate)

    elif optimizer == 'adagrad':

      optimizer = Adagrad(learning_rate=learning_rate)

    elif optimizer =='rmsprop':

      optimizer = RMSprop(learning_rate=learning_rate)

    elif optimizer =='sgd':

      optimizer = SGD(learning_rate=learning_rate)

    

    

    model = create_model(activation, num_hidden_layer, num_hidden_unit)

    model_list.append(model)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc', 'mape'],)

    

    es = EarlyStopping(monitor='val_acc', patience=50)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, epochs=200, batch_size=20, callbacks=[es])

    

   

    history_list.append(history)

    

    val_acc = np.array(history.history['val_acc'])

    

    return 1-val_acc[-1]





def create_model(activation, num_hidden_layer, num_hidden_unit):

    inputs = Input(shape=(X_train.shape[1],))

    model = inputs

    for i in range(1,num_hidden_layer):

        model = Dense(num_hidden_unit, activation=activation,)(model)

        

        

    model = Dense(1, activation='sigmoid')(model)

    model = Model(inputs, model)

    



    

    return model
%%time

model_list=[]

history_list=[]

study_name = 'titanic_study'

study = optuna.create_study(study_name=study_name,storage='sqlite:///../titanic_study.db', load_if_exists=True)

study.optimize(objective, n_trials=50, )
print(study.best_params)

print('')

print(study.best_value)

print('')

print(study.best_trial)
print(study.best_trial._number)

print(study.best_trial.params['activation'])

model_list[study.best_trial._number-1].summary()
test_df_index = test['PassengerId']

test_df = test.fillna({'Age':test['Age'].mean(),'Fare':test['Fare'].mean()})

test_df = test_df.drop(columns=[ 'PassengerId', 'Name','Ticket', 'Cabin', 'Embarked'])

X_test = test_df.replace('male', 0)

X_test = X_test.replace('female', 1)
model_list[study.best_trial._number-1].compile(optimizer=study.best_trial.params['optimizer'], loss='binary_crossentropy', metrics=['acc', 'mape'],)  

es = EarlyStopping(monitor='val_acc', patience=100)

history = model_list[study.best_trial._number-1].fit(X_train, y_train, validation_data=(X_val, y_val), verbose=1, epochs=400, batch_size=20, callbacks=[es])

predicted = model_list[study.best_trial._number-1].predict(X_test.values)

predicted_survived = np.round(predicted).astype(int)
predicted = model_list[study.best_trial._number-1].predict(X_test.values)

predicted_survived = np.round(predicted).astype(int)

predicted_survived
df = pd.concat([test_df_index,pd.DataFrame(predicted_survived, columns=['Survived'])], axis=1)
df
df.to_csv('gender_submission.csv', index=False)