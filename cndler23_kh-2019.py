# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_boston

boston = load_boston()

print(boston.data.shape)

print(str(boston.data[0]))

print(str(boston.target[0]))
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



class SGDLinearRegression(BaseEstimator, RegressorMixin):



    def __init__(self, max_iter=1000, eta=0.01):

        self.max_iter = max_iter

        self.eta = eta



    def fit(self, X, y):



        if len(X.shape) == 1: X = X.reshape((-1, 1))



        X = np.hstack((np.ones((len(X), 1)), X))



        self.beta_ = np.linalg.inv(X.T @ X) @ X.T @ y



        return self



    def predict(self, X):



        if len(X.shape) == 1: X = X.reshape((-1, 1))



        X = np.hstack((np.ones((len(X), 1)), X))



        return X @ self.beta_



    def mse(self, X, y):

        preds = self.predict(X)

        return mean_squared_error(y, preds)

    

    #utilizando mean_squared_error en vez de r2_score
from sklearn.datasets import load_boston, load_diabetes

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



for loader in (load_boston, load_diabetes):

    X, y = loader(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



    sgd_lr = Pipeline([('stds', StandardScaler()), ('sgd_lr', SGDLinearRegression())])

    sgd = Pipeline([('stds', StandardScaler()), ('sgd', SGDRegressor(penalty=None, shuffle=False, learning_rate='constant'))])



    sgd_lr.fit(X_train, y_train)

    sgd.fit(X_train, y_train)



    sgd_lr_score = sgd_lr.score(X_test, y_test)

    sgd_score = sgd.score(X_test, y_test)

    print(loader.__name__ + ':\n\tR2 SGDLR: ' + str(sgd_lr_score) + '\n\tR2 SGDR: ' + str(sgd_score))

    np.testing.assert_almost_equal(sgd_lr_score, sgd_score, decimal=1, err_msg='Los scores difieren demasiado')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.1)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)







from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor

estimator = Pipeline([('standardizer', StandardScaler()), ('predictor', SGDRegressor())])







estimator.fit(X_train, y_train)

print(estimator.score(X_test, y_test))





from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

estimator = Pipeline([('standardizer', StandardScaler()), ('predictor', Ridge(alpha=0.5))])

estimator.fit(X_train, y_train)

print(estimator.score(X_test, y_test))
#second estimator for the regularized model changing the alpha values



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor



estimator_2 = Pipeline([('standardizer', StandardScaler()), ('predictor', 

                                                           MLPRegressor(

                hidden_layer_sizes=(), activation='relu', solver='adam', alpha=0.20,batch_size='auto',

                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=2000, shuffle=True,

                random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,

                nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,

                epsilon=1e-08

                                                           )

                                                          )])





estimator_2.fit(X_train, y_train)

print(estimator.score(X_test, y_test))



# Se ve que el modelo ha mejorado utilizando MLPRegressor





#best params para el perceptron multicapa

from sklearn.model_selection import GridSearchCV

estimator_3 = Pipeline([('standardizer', StandardScaler()), ('predictor', 

                                                           MLPRegressor(

                hidden_layer_sizes=(100,100), activation='tanh', solver='adam', alpha=0.25,batch_size='auto',

                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=2000, shuffle=True,

                random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,

                nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,

                epsilon=1e-08

                                                           )

                                                          )])



search_2 = GridSearchCV (estimator_3, {'predictor__alpha': [0.0, 0.01, 0.18, 0.25, 0.40, 0.45, 0.5, 0.75, 0.83, 1.0]})

search_2.fit(X_train, y_train)

print(search_2.best_params_)

print(search_2.score(X_test, y_test))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.1)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D

from keras.utils import to_categorical

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam



K.set_image_data_format('channels_first')



input_shape = (1, 8, 8)

X_train_keras = X_train.reshape(X_train.shape[0], *input_shape)

X_test_keras = X_test.reshape(X_test.shape[0], *input_shape)

X_train_keras = X_train_keras.astype('float32')

X_test_keras = X_test_keras.astype('float32')

X_train_keras /= 16

X_test_keras /= 16

num_classes = 10

y_train_keras = to_categorical(y_train, num_classes)

y_test_keras = to_categorical(y_test, num_classes)



model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(1, 8, 8)))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

model.fit(X_train_keras, y_train_keras, batch_size=32, epochs=1000, verbose=0, validation_split=0.1)

score = model.evaluate(X_test_keras, y_test_keras, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
import tensorflow as tf

import tensorflow_probability as prob
sq_footage = tf.contrib.layers.real_valued_column("sq_footage")

feature_columns = [sq_footage]

 



#define input function 

def input_fn (feature_data, label_data=None):

    return {"sq_footage": feature_data}, label_data





#instantiate linear regression model 

estimator = tf.contrib.learn.LinearRegressor(

    feature_columns=feature_columns,

    optimizer=tf.train.FtrlOptimizer(learning_rate=100)

)



#train



estimator.fit(

    input_fn=lambda:input_fn(tf.constant([1000,2000]),

                             tf.constant([100000,20000])),

    

    steps=100)
#predict 

estimator.predict(input_fn=lambda:input_fn(tf.constant([3000])))