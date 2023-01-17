import pandas as pd

import numpy as np

da=pd.read_csv('../input/wwwkagglecompuneetsinghfile/FileName.csv')

da
#da=da.drop(['SibSp','Parch'],axis=1)

#da

da
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



da[['Age','Fare']]=scaler.fit_transform(da[['Age','Fare']].values)

da


x=da['Survived']



y=da.iloc[:,1:8]
y
import keras

from keras.models import Sequential # its for create ANN algorithm

from keras.layers import Dense

from keras.layers import LeakyReLU,PReLU,ELU # its for relu activations function 

model = Sequential()

#model.add(tf.keras.Input(shape=(7,)))

model.add(Dense(440, input_dim=7, activation='relu'))

model.add(Dense(440,activation='relu'))

model.add(Dense(1, activation='sigmoid'))

opt=keras.optimizers.Adam(learning_rate=0.01)



model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

history=model.fit(y, x,epochs=20, batch_size=5, verbose=1) 



from kerastuner.tuners import RandomSearch

def build_model(hp):

    model = keras.Sequential()

    model.add(Dense(units=hp.Int('units',

                                        min_value=16,

                                        max_value=512,

                                        step=8),

                           activation=hp.Choice(

        'dense_activation',

        values=['relu', 'tanh', 'sigmoid'],

        default='relu'

    )))

    model.add(Dense(units=hp.Int('units',

                                        min_value=8,

                                        max_value=512,

                                        step=8),

                           activation=hp.Choice(

        'dense_activation',

        values=['relu', 'tanh', 'sigmoid'],

        default='relu'

    )))

    



    model.add(Dense(1, activation='sigmoid'))

    model.compile(

        optimizer=keras.optimizers.Adam(

            hp.Choice('learning_rate',

                      values=[1e-2, 1e-3, 1e-4])),

        loss='binary_crossentropy',

        metrics=['accuracy'])

    return model
tuner = RandomSearch(

    build_model,

    objective='val_accuracy',

    max_trials=5,

    executions_per_trial=3,

    directory='my_dggluir',

    project_name='helloworld')
tuner.search_space_summary()

tuner.search(y, x,validation_split=0.01,epochs=20, batch_size=5, verbose=1)
tuner.get_best_hyperparameters()[0].values

                                                    

m=tuner.get_best_models()[0]

tuner.results_summary()

history=m.fit(y, x,epochs=20,validation_split=0.1, batch_size=5, verbose=1) 

m.summary()