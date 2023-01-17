import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras import backend as K

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from keras.layers import Dense,Input,Dropout

from keras.models import Model

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from keras.models import load_model

import matplotlib.pyplot as plt

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.drop("molecule_name", axis=1, inplace=True)

test.drop("molecule_name", axis=1, inplace=True)
test_id = test['id']

train.drop("id", axis=1, inplace=True)

test.drop("id", axis=1, inplace=True)
train_type = pd.get_dummies(train['type'])

test_type = pd.get_dummies(test['type'])
train['type'] = train['type'].astype("category").cat.codes

test['type'] = test['type'].astype("category").cat.codes
#train_new = pd.concat([train, train_type], axis=1)

#train_new.drop("type", axis=1, inplace=True)

#test_new = pd.concat([test, test_type], axis=1)

#test_new.drop("type", axis=1, inplace=True)
train.head()
y = train["scalar_coupling_constant"]

train.drop("scalar_coupling_constant", axis=1, inplace=True)

X = train
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
K.clear_session()

def RegressionModel(in_):

    model = Dense(256,kernel_initializer='normal',activation="relu")(in_)

    model = Dense(128,kernel_initializer='normal',activation="relu")(model)

    model = Dense(64,kernel_initializer='normal',activation="relu")(model)

    model = Dense(32,kernel_initializer='normal',activation="relu")(model)

    

    model = Dense(1,kernel_initializer='normal',activation="linear")(model)

    

    return model
Input_Sample = Input(shape=(x_train.shape[1],))

Output_ = RegressionModel(Input_Sample)

EnhanceRegression = Model(inputs=Input_Sample, outputs=Output_)
EnhanceRegression.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse','mae'])

EnhanceRegression.summary()
ES = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=200, verbose=1, mode='auto', baseline=None,

                              restore_best_weights=False)

MC = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='auto', verbose=1, save_best_only=True)
num_epochs =300

num_batch_size = 1080

ModelHistory = EnhanceRegression.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, 

                                    validation_data=(x_test, y_test),

                                     callbacks = [ES,MC],

                                    verbose=1)
#Loss Curves

plt.figure(figsize=[20,9])

plt.plot(ModelHistory.history['loss'], 'r')

plt.plot(ModelHistory.history['val_loss'], 'b')

plt.legend(['Training Loss','Validation Loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss Curves')
#Accuracy Curves

plt.figure(figsize=[20,9])

plt.plot(ModelHistory.history['mean_absolute_error'], 'r')

plt.plot(ModelHistory.history['val_mean_absolute_error'], 'b')

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Accuracy Curves')
#saved_model = load_model('best_model.h5')
y_pred_test = EnhanceRegression.predict(test)
prediction = y_pred_test.flatten()

prediction 
my_submission = pd.DataFrame({'id':test_id ,'scalar_coupling_constant': prediction })

my_submission.to_csv('SubmissionVictorX.csv', index=False)