import numpy as np

import pandas as pd



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



# keras import

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping



# hyperparameter optimization

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier



# data augmentation

from keras.preprocessing.image import ImageDataGenerator



# visualisation

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

#set figure size

plt.rcParams['figure.figsize'] = 12, 6

sns.set_style('white')



# others

from random import randrange

from time import time
n_epochs = 30 # 30 

n_epochs_cv = 10 # 10  # reduce number of epochs for cross validation for performance reason



n_cv = 3

validation_ratio = 0.10

# load dataset and check dimension

data_set = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print(data_set.shape)
data_set.sample(3)
# segregate training data set in pixel features and label

y = data_set['label']

X = data_set.drop(labels = ['label'], axis=1) 

# free memory

del data_set



# check distribution of the handwritten digits

sns.countplot(y, color='skyblue');
# show multiple images chosen randomly 

fig, axs = plt.subplots(6, 10, figsize=(10, 6)) # 6 rows of 10 images



for ax in axs.flat:

    i = randrange(X.shape[0])

    ax.imshow(X.loc[i].values.reshape(28, 28), cmap='gray_r')

    ax.set_axis_off()
# Normalize pixel value to range 0 to 1

X = X / 255.0



# extract train and validation set

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = validation_ratio)
# define model

mlp = Sequential()

mlp.add(Dense(128, activation='relu', input_shape=(784,)))

mlp.add(Dense(64, activation='sigmoid'))  

mlp.add(Dense(10, activation='softmax'))



mlp.compile(

  optimizer='adam',

  loss='categorical_crossentropy',

  metrics=['accuracy'],

)



mlp.summary()
# Train the model



#define callbacks

early_stop = EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience=5, restore_best_weights=True)



history = mlp.fit(

    X_train,

    to_categorical(y_train),

    epochs = n_epochs,  

    validation_data = (X_val, to_categorical(y_val)),

    batch_size = 32,

    callbacks = [early_stop]

)
# compare accuracy accuracy on training and validation data

df_history = pd.DataFrame(history.history)

sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
start=time()



# define a function to create model, required for KerasClassifier

# the function takes drop_out rate as argument so we can optimize it  

def create_mlp_model(dropout_rate=0):

    # create model

    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(784,))) 

    # add a dropout layer if rate is not null

    if dropout_rate != 0:

        model.add(Dropout(rate=dropout_rate))        

    model.add(Dense(64, activation='sigmoid')) 

    # add a dropout layer if rate is not null    

    if dropout_rate != 0:

        model.add(Dropout(rate=dropout_rate))           

    model.add(Dense(10, activation='softmax'))

    

    # Compile model

    model.compile( 

        optimizer='adam',

        loss='categorical_crossentropy',

        metrics=['accuracy'],

        )    

    return model



# define function to display the results of the grid search

def display_cv_results(search_results):

    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))

    means = search_results.cv_results_['mean_test_score']

    stds = search_results.cv_results_['std_test_score']

    params = search_results.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):

        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))    

    

# create model

model = KerasClassifier(build_fn=create_mlp_model, verbose=1)

# define parameters and values for grid search 

param_grid = {

    'batch_size': [16, 32, 64],

    'epochs': [n_epochs_cv],

    'dropout_rate': [0.0, 0.10, 0.20, 0.30],

}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)

grid_result = grid.fit(X, to_categorical(y))  # fit the full dataset as we are using cross validation 



# print out results

print('time for grid search = {:.0f} sec'.format(time()-start))

display_cv_results(grid_result)
# reload best model

mlp = grid_result.best_estimator_ 



# retrain best model on the full training set 

history = mlp.fit(

    X_train,

    to_categorical(y_train),

    validation_data = (X_val, to_categorical(y_val)),

    epochs = n_epochs,

    callbacks = [early_stop]    

)
# get prediction on validation dataset 

y_pred = mlp.predict(X_val)

print('Accuracy on validation data = {:.4f}'.format(accuracy_score(y_val, y_pred)))



# plot accuracy on training and validation data

df_history = pd.DataFrame(history.history)

sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
# load test data and make prediction

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

y_test = mlp.predict(test)
# convert prediction to df

submission = pd.DataFrame(data=y_test)



# set label as the 0-9 class with highest value 

submission['Label'] = submission.idxmax(axis=1)

submission['ImageId'] = np.asarray([i+1 for i in range(submission.shape[0])])



submission.to_csv('submission-mlp_dropout.csv', 

                  columns=['ImageId','Label'],

                  header=True,

                  index=False)

# Reshape the images

img_size = 28

X_cnn = X.values.reshape(-1, img_size, img_size, 1)

# check 

print(X_cnn.shape)



X_train, X_val, y_train, y_val = train_test_split(X_cnn, y, test_size = validation_ratio)
# function to create the model for Keras wrapper to scikit learn

# we will optimize the type of pooling layer (max or average) and the activation function of the 2nd and 3rd convolution layers 

def create_cnn_model(pool_type='max', conv_activation='sigmoid', dropout_rate=0.10):

    # create model

    model = Sequential()

    

    # first layer: convolution

    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))) 

        

    # second series of layers: convolution, pooling, and dropout

    model.add(Conv2D(32, kernel_size=(5, 5), activation=conv_activation))  

    if pool_type == 'max':

        model.add(MaxPooling2D(pool_size=(2, 2)))

    if pool_type == 'average':

        model.add(AveragePooling2D(pool_size=(2, 2)))

    if dropout_rate != 0:

        model.add(Dropout(rate=dropout_rate))     

    

    # third series of layers: convolution, pooling, and dropout    

    model.add(Conv2D(64, kernel_size=(3, 3), activation=conv_activation))   # 32   

    if pool_type == 'max':

        model.add(MaxPooling2D(pool_size=(2, 2)))

    if pool_type == 'average':

        model.add(AveragePooling2D(pool_size=(2, 2)))

    if dropout_rate != 0:

        model.add(Dropout(rate=dropout_rate))     

      

    # fourth series

    model.add(Flatten())         

    model.add(Dense(64, activation='sigmoid')) # 64

    # add a dropout layer if rate is not null    

    if dropout_rate != 0:

        model.add(Dropout(rate=dropout_rate)) 

        

    model.add(Dense(10, activation='softmax'))

    

    # Compile model

    model.compile( 

        optimizer='adam',

        loss='categorical_crossentropy',

        metrics=['accuracy'],

        )    

    return model



cnn = create_cnn_model()



cnn.compile(

  optimizer='adam',

  loss='categorical_crossentropy',  

  metrics=['accuracy'],

)



cnn.summary()
# Train the default CNN model

history = cnn.fit(

    X_train,

    to_categorical(y_train),

    epochs=n_epochs,  

    validation_data=(X_val, to_categorical(y_val)), 

    batch_size=32,

    callbacks = [early_stop]

)
# optimize model 

start = time()



# create model

model = KerasClassifier(build_fn=create_cnn_model, verbose=1)

# define parameters and values for grid search 

param_grid = {

    'pool_type': ['max', 'average'],

    'conv_activation': ['sigmoid', 'tanh'],    

    'epochs': [n_epochs_cv],

}



grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)

grid_result = grid.fit(X_train, to_categorical(y_train))



# summarize results

print('time for grid search = {:.0f} sec'.format(time()-start))

display_cv_results(grid_result)
# optimize parameters of the fit method 

cnn_model = create_cnn_model(pool_type = grid_result.best_params_['pool_type'],

                             conv_activation = grid_result.best_params_['conv_activation'])



# With data augmentation 

datagen = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False,  # randomly flip images

        fill_mode='constant', cval = 0.0)



datagen.fit(X_train)



history = cnn_model.fit_generator(datagen.flow(X_train,to_categorical(y_train), batch_size=32),

                                  epochs = n_epochs, 

                                  validation_data = (X_val,to_categorical(y_val)),

                                  verbose = 1, 

                                  steps_per_epoch = X_train.shape[0] / 32,

                                  callbacks = [early_stop])



# plot accuracy on training and validation data

df_history = pd.DataFrame(history.history)

sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
# optimize parameters of the fit method 

cnn_model = create_cnn_model(pool_type = grid_result.best_params_['pool_type'],

                             conv_activation = grid_result.best_params_['conv_activation'], 

                            dropout_rate=0.0)



#define early stop on the accuracy as this is the metric we want to improve

early_stop = EarlyStopping(monitor = 'accuracy', mode = 'max', patience=5, restore_best_weights=True)

history = cnn_model.fit_generator(datagen.flow(X_train,to_categorical(y_train), batch_size=32),

                                  epochs = n_epochs, 

                                  validation_data = (X_val,to_categorical(y_val)),

                                  verbose = 1, 

                                  steps_per_epoch = X_train.shape[0] / 32,

                                  callbacks = [early_stop])



# plot accuracy on training and validation data

df_history = pd.DataFrame(history.history)

sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
# save weights

cnn_model.save_weights('mnist_cnn.h5')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



X_test = test.values.reshape(-1, img_size, img_size, 1)

y_test = cnn_model.predict(X_test)



# convert to df

submission = pd.DataFrame(data=y_test)



# set label as the 0-9 class with highest value 

submission['Label'] = submission.idxmax(axis=1)

submission['ImageId'] = np.asarray([i+1 for i in range(submission.shape[0])])



submission.to_csv('submission-cnn.csv', 

                  columns=['ImageId','Label'],

                  header=True,

                  index=False)