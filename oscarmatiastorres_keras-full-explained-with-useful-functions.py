#Import libraries

import pandas as pd

pd.options.display.max_columns = 500

import numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.metrics as metrics

from sklearn.metrics import mean_squared_error

import warnings

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston
boston_dataset = load_boston()

X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)



Y=boston_dataset['target']

XY = pd.concat([X,pd.Series(Y, name = 'target' )], axis=1)
XY.head(2)
print(u'- The rows numbers is: {}'.format(XY.shape[0]))

print(u'- The number of columns is: {}'.format(XY.shape[1]))
#Usefull functions that you coud use for a lot of similar regresions works 

    

def relaciones_vs_target_reg(X, Y, return_type='axes'):

    '''

    Function that represents scatter plots of the variables

    in X as a function of the variable Y

    '''

    fig_tot = (len(X.columns))

    fig_por_fila = 4.

    tamanio_fig = 4.

    num_filas = int( np.ceil(fig_tot/fig_por_fila) )    

    plt.figure( figsize=( fig_por_fila*tamanio_fig+5, num_filas*tamanio_fig+5 ) )

    c = 0 

    for i, col in enumerate(X.columns):

        plt.subplot(num_filas, fig_por_fila, i+1)

        sns.regplot(x=X[col], y=Y)

        plt.title( '%s vs %s' % (col, 'target') )

        plt.ylabel('Target')

        plt.xlabel(col)

    plt.show()

    

def represento_historico(historico):

    hist = pd.DataFrame(historico.history)

    hist['epoch'] = historico.epoch



    plt.figure(figsize=(15,7))

    plt.xlabel('Epoch')

    plt.ylabel('Mean absolute error [MAE]')

    plt.plot(hist['epoch'], hist['mae'],

           label='Training error')

    plt.plot(hist['epoch'], hist['val_mae'],

           label = 'Validation Error')

    plt.title('MAE error in training and in test')

    plt.ylim([0,5])

    plt.legend()



    plt.figure(figsize=(15,7))

    plt.xlabel('Epoch')

    plt.ylabel('Root mean square error [MSE]')

    plt.plot(hist['epoch'], hist['mse'],

           label='Training error')

    plt.plot(hist['epoch'], hist['val_mse'],

           label = 'Validation error')

    plt.title('MSE error in training and in test')

    plt.ylim([0,20])

    plt.legend()

    plt.show()



def hist_pos_neg_feat(x, y, density=0, nbins=11, targets=(0,1)):

    '''

    Represent the variables in x divided into two distributions

    depending on its value of y is 1 or 0

    '''

    fig_tot = len(x.columns)

    fig_tot_fila = 4.; fig_tamanio = 4.

    num_filas = int( np.ceil(fig_tot/fig_tot_fila) )

    plt.figure( figsize=( fig_tot_fila*fig_tamanio+2, num_filas*fig_tamanio+2 ) )

    target_neg, target_pos = targets

    for i, feat in enumerate(x.columns):

        plt.subplot(num_filas, fig_tot_fila, i+1);

        plt.title('%s' % feat)

        idx_pos = y == target_pos

        idx_neg= y == target_neg

        represento_doble_hist(x[feat][idx_pos].values, x[feat][idx_neg].values, nbins, 

                   density = density, title=('%s' % feat))
plt.figure(figsize=(15,7))

ax = sns.boxplot(data=X)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title(u'Box representation of the independent variables X')

plt.ylabel('Values of the variable Y')

_ = plt.xlabel('Variables names')


#plt.figure(figsize=(18,20))

#n = 0

#for i, column in enumerate(X.columns):

#   n+=1

#    plt.subplot(5, 5, n)

#    sns.distplot(X[column], bins=30)

#    plt.title('Distribución var {}'.format(column))

#plt.show()

relaciones_vs_target_reg(X, Y)
matriz_correlaciones = XY.corr(method='pearson')

n_ticks = len(XY.columns)

plt.figure( figsize=(9, 9) )

plt.xticks(range(n_ticks), XY.columns, rotation='vertical')

plt.yticks(range(n_ticks), XY.columns)

plt.colorbar(plt.imshow(matriz_correlaciones, interpolation='nearest', 

                            vmin=-1., vmax=1., 

                            cmap=plt.get_cmap('Blues')))

_ = plt.title('Correlation Matrix Pearson method')
correlaciones_target = matriz_correlaciones.values[ -1, : -1]

indices_inversos =  abs(correlaciones_target[ : ]).argsort()[ : : -1]

diccionario = {}

for nombre, correlacion in zip( X.columns[indices_inversos], list(correlaciones_target[indices_inversos] ) ):

    diccionario[nombre] = correlacion

pd.DataFrame.from_dict(diccionario, orient='index', columns=['Correlación con la target'])
obj_escalar = StandardScaler()

X_estandarizado = obj_escalar.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_estandarizado, Y, test_size=0.2, random_state=0)
def constructor_modelo():

    # model definition

    modelo = keras.Sequential([

    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),

    layers.Dense(64, activation='relu'),

    layers.Dense(1)])

    

    # def optimizer

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    

    # compile model

    modelo.compile(loss='mse',

                optimizer=optimizer,

                metrics=['mae', 'mse'])

    return modelo
model = constructor_modelo()
model.fit(X_train, Y_train)
X_train.shape[1]
model.summary()
example = X_train[:10]

ex_pred = model.predict(example)

ex_pred
# I show one point for each completed epochs

class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0: print('=D')

        print('.', end='')



EPOCHS = 1000

historico = model.fit(X_train, Y_train, 

                    epochs=EPOCHS,

                    validation_split = 0.2, 

                    verbose=0,

                    callbacks=[PrintDot()])
hist = pd.DataFrame(historico.history)

hist['epoch'] = historico.epoch

hist.tail()
represento_historico(historico)
model = constructor_modelo()



early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=10)



history = model.fit(X_train, Y_train, 

                    epochs=EPOCHS,

                    validation_split = 0.2, 

                    verbose=0, 

                    callbacks=[early_stop, PrintDot()])



represento_historico(history)
Y_train_pred = model.predict(X_train)

plt.title('Real values  vs predictions in train')

plt.xlabel('Real Values')

plt.ylabel('Predictions')

_ = plt.plot(Y_train, Y_train_pred, '.', Y_train, Y_train, '-')
Y_test_pred = model.predict(X_test)

plt.title('Real Values vs predictions in test')

plt.xlabel('Real Values')

plt.ylabel('Predictions')

_ = plt.plot(Y_test, Y_test_pred, '.', Y_test, Y_test, '-')
plt.xlabel('Errores en train y en test')

plt.ylabel('freqs')

plt.hist(Y_train - Y_train_pred.flatten(), bins=21, label='Train')

plt.hist(Y_test - Y_test_pred.flatten(), bins=21, label='Test')

_ = plt.legend()
error_mse_train = round(mean_squared_error(Y_train, Y_train_pred),2)

error_mse_test = round(mean_squared_error(Y_test, Y_test_pred),2)

print('El error cuadrático medio en train es: {}'.format(error_mse_train))

print('El error cuadrático medio en test es: {}'.format(error_mse_test))
# Pleas coment and upvote if you think that notbook was usefull. Sorry about my english but is not my native language