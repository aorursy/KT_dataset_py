import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
df = pd.read_csv(r"../input/heart.csv")
df.head()
# Some distribution plots



f, axes = plt.subplots(2, 2, figsize=(10, 10)) #sharex=True)

sns.distplot(df["age"][df["target"]==1], color="red", label="heart diease", ax=axes[0, 0])

sns.distplot(df["age"][df["target"]==0] , color="skyblue", label="No heart diease", ax=axes[0, 0])



sns.distplot(df["chol"][df["target"]==1], color="red", label="heart diease", ax=axes[0, 1])

sns.distplot(df["chol"][df["target"]==0] , color="skyblue", label="No heart diease", ax=axes[0, 1])



sns.distplot(df["thalach"][df["target"]==1], color="red", label="heart diease", ax=axes[1, 0])

sns.distplot(df["thalach"][df["target"]==0] , color="skyblue", label="No heart diease", ax=axes[1, 0])



sns.distplot(df["trestbps"][df["target"]==1], color="red", label="heart diease", ax=axes[1, 1])

sns.distplot(df["trestbps"][df["target"]==0] , color="skyblue", label="No heart diease", ax=axes[1, 1])

plt.legend()



plt.show()

sns.heatmap(df.corr(),annot=True,cmap='RdYlGn') 

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show



# Well, there is nothing too striking in the correlation heatmap
from keras.optimizers import SGD 



from keras.utils import np_utils

from keras.models import Sequential # The common deep learning network

from keras.layers.core import Dense, Activation

from keras.activations import relu, elu, selu, sigmoid, exponential, tanh
X = df.drop("target",axis=1)

y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
X_train.shape
y_train.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled  = scaler.fit_transform(X_test)
NB_CLASSES = 2 # Number of classes to predict
y_train_new = np_utils.to_categorical(y_train,NB_CLASSES)

y_test_new = np_utils.to_categorical(y_test,NB_CLASSES)
import talos as ts
from keras.activations import relu, elu, selu, sigmoid, exponential, tanh



p = {

    'activation': [relu, elu, selu, sigmoid, exponential, tanh],

    'batch_size': [64,128,256],

    'First_Neron' : [64,128,256],

    'Second_Neron' : [64,128,256],

    'Third_Neron' : [64,128,256],

    'epochs': [50,100,150,200]

}
def get_model_talos(X_train,y_train,X_test,y_test,params):

    model = Sequential() # load the sequential model

    model.add(Dense(NB_CLASSES, input_shape=(13,), activation=params['activation'])) # add a dense layer

    model.add(Dense(params['First_Neron'], input_shape=(13,), activation=params['activation']))

    model.add(Dense(params['Second_Neron'], input_shape=(13,), activation=params['activation']))

    model.add(Dense(params['Third_Neron'], input_shape=(13,), activation=params['activation']))

    model.add(Dense(NB_CLASSES, input_shape=(13,))) # add a dense layer

    model.add(Activation('softmax'))

    

    model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    out = model.fit(X_train, y_train,

                    epochs=params['epochs'],

                    batch_size=params['batch_size'],

                    validation_data=(X_test,y_test),

                    verbose=0)

            

    return out, model
t = ts.Scan(x=X_train_scaled, y=y_train_new, x_val=X_test_scaled, y_val=y_test_new, params=p, model=get_model_talos)
from talos import Reporting

r = Reporting(t)
r.high()
r.rounds2high()
frame = t.data
frame[frame["val_acc"] >= '0.89']
ts.Deploy(t, 'heart_diease_predmodel');
heart_diease_predmodel = ts.Restore('heart_diease_predmodel.zip')