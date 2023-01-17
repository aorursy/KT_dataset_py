import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import keras, os, pickle, ast

# import implicit

import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from IPython.display import SVG

from keras.optimizers import Adam

from keras.layers import Dense,Dropout, Activation

from keras.layers.normalization import BatchNormalization

from keras.callbacks import CSVLogger

from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

from keras.callbacks import ReduceLROnPlateau, History

from keras.regularizers import l1,l2

import seaborn as sns

sns.set()

rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
df = pd.read_excel("../input/EngagementTest3.xlsx")



#remove first column from importing excel

df = df.drop('Unnamed: 0', axis = 1)
#shuffle dataframe for randomness in splitting

df = df.sample(frac = 1)
df.head()
#changing values to numpy array

X = df.iloc[:,:5].to_numpy()

Y = df.iloc[:,5:].to_numpy()



print(X.shape)

print(Y.shape)
#normalizing the data, will help with different ranges of values for different features

min_max_scaler = preprocessing.MinMaxScaler()

values_scaled = min_max_scaler.fit_transform(X)

scaled_df = pd.DataFrame(values_scaled)



scaled_df.head()
#apply one hot encoding to Y values

ohe = OneHotEncoder()

Y = ohe.fit_transform(Y).toarray()
Y
#Split the dataset into x and y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
#Building the model, to train this we will build a simple 3 layer NN



def create_model():

    n_cols = X_train.shape[1]

    

    #batch normalization layers need to be added before relu activation

    model = keras.Sequential()

    model.add(Dense(50, activity_regularizer = l2(0.00001), input_shape = (n_cols,)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(200))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(200))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

              

    model.add(Dense(200))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(200))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(5, activation = 'softmax'))

    

    return model
#training the model

#test if model has any serious loss/accuracy problems

model = create_model()

model.compile(optimizer = Adam(lr = 0.00001), loss = 'categorical_crossentropy',

                  metrics = ['categorical_accuracy'])

#adding callbacks

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 50, min_lr = 1e-7, verbose = 1)



history = History()



history = model.fit(X_train, Y_train,

                    validation_split = 0.4,

                    epochs = 1000,

                    batch_size = 64,

                    verbose = 2,

                    callbacks = [reduce_lr, history])
result_dict = {}

subset_data_list = [0.2, 0.4, 0.6, 0.8]

os.mkdir('/kaggle/logs')



for subset_data in subset_data_list:

    #training the model on a proportion of data and save the loss and accuracy

    

    model = create_model()

    model.compile(optimizer = Adam(lr = 0.00001), loss = 'categorical_crossentropy',

                  metrics = ['categorical_accuracy'])

    

    #adding callbacks

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5,

                                  patience=20, min_lr=1e-7, verbose = 1)

    history = History()

    logger = CSVLogger(os.path.join(rootPath, 'logs', 'log_results-{}.csv'.format(subset_data)))

    

    #training the model with subset of data

    history = model.fit(X_train, Y_train,

                        validation_split = 1 - subset_data,

                        epochs = 1000,

                        verbose = 2,

                        callbacks = [reduce_lr, history, logger])

    

    result_dict[subset_data] = {

        'training_loss' : history.history['loss'],

        'validation_loss': history.history['val_loss'],

        'training_accuracy': history.history['categorical_accuracy'],

        'validation_accuracy': history.history['val_categorical_accuracy']

    }
def plotGraph(result_dict, plot_var):

    #this plots a graph of training loss vs epochs

    for subset_data in result_dict.keys():

        ax = sns.lineplot(x = np.arange(1000), 

                          y = result_dict[subset_data][plot_var], label =subset_data)

        ax.set(xlabel = 'training epoch', ylabel = plot_var)

        

    if plot_var == "training_loss":

        #rescale for easier comparison

        ax.set_ylim(0.5, 2)



        

    ax.set_title('{} with epoch'.format(plot_var))

    ax.legend()

    

    return ax
plotGraph(result_dict, 'training_loss')
plotGraph(result_dict,'validation_loss')
plotGraph(result_dict,'training_accuracy')
plotGraph(result_dict,'validation_accuracy')
predictions = model.predict(X_test[:10])
print(X_test[:10])
print(predictions)