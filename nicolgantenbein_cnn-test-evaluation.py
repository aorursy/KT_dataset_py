#include libraries

%matplotlib inline

import numpy as np

from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout

from keras.constraints import maxnorm

%load_ext autoreload

%autoreload 2
#set target folder to create plots

import os

TARGET_DIR = "/kaggle/working/figures"

if not os.path.isdir(TARGET_DIR):

    os.mkdir(TARGET_DIR)
def standardize_scale(*args):

    """ Standardize the feature matrix x by multipling by 10E9 -> values will be >0 and around 1.

    """

    #set scale manualy to 10e9

    scale = np.power(10,9)

    

    for x in args:

        yield x*scale
print("Start_Program")
"""Choose which version of the data set should be used

- reduce: set this option equal true, if you want to use just the upper left quarter of the

          dataset containing the full 64x64 matrice

          

          if set to false the full matrice will be used to train the model

- quarter: set this value to true, if you wish to use the dataset containing just the upper

           left quarter of the matrice in higher resolution (64x64 for the quarter)

"""

reduce_ = False

quarter = True

#abort if both options are equal true

assert not (quarter and reduce_)

#set the file path

#if working on kaggle

x_file_path = "../data-input/20191205/cloak_mat_cs64{}_2.csv".format("" if quarter else "full")

#if working on local computer (not recommend, unless you can use GPU, oterwise it will be really slow)

#x_file_path = "../data/cloak_mat_cs64{}_2.csv".format("" if quarter else "full")



#setting file name extensions for plots and pickled lists

annotation = "_quarter_64" if quarter else ""

reduce_str = "_reduced" if reduce_ else ""
# Load data (each row corresponds to one sample)

x_train = np.loadtxt('../input/quarter/x_train{}.csv'.format(annotation + reduce_str), dtype=np.float64, delimiter=',')

x_test  = np.loadtxt('../input/quarter/x_test{}.csv'.format( annotation + reduce_str), dtype=np.float64, delimiter=',')



# Reshape x to recover its 2D content

side_length = 32 if reduce_ else 64

x_train = x_train.reshape(x_train.shape[0], side_length, side_length, 1)

x_test = x_test.reshape(x_test.shape[0], side_length, side_length, 1)

print(x_train.shape)

print(x_test.shape)



# Load labels:

y_train = np.loadtxt('../input/quarter/y_train.csv', dtype=np.float64, delimiter=',')

y_test = np.loadtxt('../input/quarter/y_test.csv', dtype=np.float64, delimiter=',')



# Transform the labels y so that min(y) == 0 and max(y) == 1. Importantly, y_train and y_test must be considered jointly.

y_train, y_test = standardize_scale(y_train, y_test)
from tensorflow.keras import datasets, layers, models

from tensorflow import keras

import numpy as np

from keras.layers import LeakyReLU



#activation_fct = "relu" #"leakyrelu

nb_epochs = 50



#fixed values from first exploratory search:

n_filters = 15

k_sizes = 5

nb_layers = 4

dense_neurons = 1



#fixed argument found in second grid search

#dim 0

activation_list = ["sigmoid"]

#dim 1

filter_mult_list = [3]

#dim 2

activation_last_list = ["relu"]

#dim 3

dim3_list = [1]



#define 4 4D Vectors to store test and train error (mae and mse)

rmse_test = np.ones((len(activation_list),len(filter_mult_list),len(activation_last_list),len(dim3_list)))

rmse_train = np.ones((len(activation_list),len(filter_mult_list),len(activation_last_list),len(dim3_list)))



rmae_test = np.ones((len(activation_list),len(filter_mult_list),len(activation_last_list),len(dim3_list)))

rmae_train = np.ones((len(activation_list),len(filter_mult_list),len(activation_last_list),len(dim3_list)))



## keep structure of old program, but as all list have a single element no iteration will be done



#iterate over dim 0

for dim0, activation_fct in enumerate(activation_list):

    print("entering_loop_of_dim_0_with_activation_equal_{}".format(activation_fct))

    

    for dim1, filter_mult in enumerate(filter_mult_list):

        print("entering_loop_of_dim_1_with_filter_mult_equal_{}".format(filter_mult))

        

        for dim2,activation_last in enumerate(activation_last_list):

            print("entering_loop_of_dim_2_with_activation_last_equal_{}".format(activation_last))

            

            for dim3, b in enumerate(dim3_list):

                print("entering_loop_of_dim_3_with_dim3_equal_{}".format(b))

                

                # defining the neural network

                print("build_model")             

                model = models.Sequential()

                

                #add first convolutional layer

                """if reduce_:

                    model.add(layers.Conv2D(n_filters, kernel_size =(k_sizes, k_sizes), strides=(1, 1), activation=activation_fct, input_shape=(32, 32, 1), padding = "same"))

                else:"""

                if(activation_fct == "leakyrelu.1"):

                    model.add(layers.Conv2D(n_filters, kernel_size =(k_sizes, k_sizes), strides=(1, 1), input_shape=(64, 64, 1), padding = "same"))

                    keras.layers.LeakyReLU(alpha=0.1)

                elif(activation_fct == "leakyrelu.2"):

                    model.add(layers.Conv2D(n_filters, kernel_size =(k_sizes, k_sizes), strides=(1, 1), input_shape=(64, 64, 1), padding = "same"))

                    keras.layers.LeakyReLU(alpha=0.2)

                elif(activation_fct == "leakyrelu.4"):

                    model.add(layers.Conv2D(n_filters, kernel_size =(k_sizes, k_sizes), strides=(1, 1), input_shape=(64, 64, 1), padding = "same"))

                    keras.layers.LeakyReLU(alpha=0.4)

                else:

                    model.add(layers.Conv2D(n_filters, kernel_size =(k_sizes, k_sizes), strides=(1, 1), activation=activation_fct, input_shape=(64, 64, 1), padding = "same"))

                

                layer = 1

                while layer < nb_layers:

                    # reduce image size by maxpooling

                    model.add(layers.MaxPooling2D(pool_size =(2, 2), strides =(2, 2)))

                    # add another convolutional layer

                    if(activation_fct == "leakyrelu.1"):

                        model.add(layers.Conv2D(int(n_filters*filter_mult), kernel_size =(k_sizes, k_sizes), strides=(1, 1), padding = "same"))            

                        keras.layers.LeakyReLU(alpha=0.1)

                    elif(activation_fct == "leakyrelu.2"):

                        model.add(layers.Conv2D(int(n_filters*filter_mult), kernel_size =(k_sizes, k_sizes), strides=(1, 1), padding = "same"))            

                        keras.layers.LeakyReLU(alpha=0.2)

                    elif(activation_fct == "leakyrelu.4"):

                        model.add(layers.Conv2D(int(n_filters*filter_mult), kernel_size =(k_sizes, k_sizes), strides=(1, 1), padding = "same"))            

                        keras.layers.LeakyReLU(alpha=0.4)

                    else:

                        model.add(layers.Conv2D(int(n_filters*filter_mult), kernel_size =(k_sizes, k_sizes), strides=(1, 1), activation=activation_fct, padding = "same"))

                    layer += 1;

                    

                #add a dense neural network at the end

                model.add(layers.Flatten()) 

                #add a layer with a single neuron

                if(activation_last == "leakyrelu.1"):

                    model.add(layers.Dense(1))

                    keras.layers.LeakyReLU(alpha=0.1)

                elif(activation_last == "leakyrelu.2"):

                    model.add(layers.Dense(1))

                    keras.layers.LeakyReLU(alpha=0.2)

                elif(activation_last == "leakyrelu.4"):

                    model.add(layers.Dense(1))

                    keras.layers.LeakyReLU(alpha=0.4)

                else:

                    model.add(layers.Dense(1, activation=activation_last))

                

                print("copile_model:")

                #run model and save results in vector

                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mae']) 

        

                #print(model.summary())

            

                print("sucessfully_compiled_fit:")

                history = model.fit(x_train, y_train, epochs = nb_epochs)

                

                #rmse_test[dim0][dim1][dim2][dim3] = np.sqrt(history.history['val_mean_squared_error'][-1])

                #rmse_train[dim0][dim1][dim2][dim3] = np.sqrt(history.history['mean_squared_error'][-1])

                

                #rmae_test[dim0][dim1][dim2][dim3] = np.sqrt(history.history['val_mae'][-1])

                #rmae_train[dim0][dim1][dim2][dim3] = np.sqrt(history.history['mae'][-1])





"""evaluate the model on the test set

This values have never been used to train or validate the model and give thus and

unbiased result

"""

mse_loss, mse_metric, mae_metric = model.evaluate(x_test, y_test)

print(np.isclose(mse_loss, mse_metric))

print("RMSE on test set:", np.sqrt(mse_metric))

print("MAE on test set: ", mae_metric)

print("end_of_code")
#generate a zip of the produced figures

import shutil

shutil.make_archive("/kaggle/working/output_figures_RMSE", 'zip', TARGET_DIR)