%matplotlib inline

import pandas as pd
import numpy as np

from time import time
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

# My seed

seed = 42
df_train = pd.read_csv('../input/train.csv')
X_train = df_train.drop(['label'], axis=1)
y_train = df_train['label']

# Free memory space

del df_train

print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
X_train = X_train / 255
X_train = X_train.values.reshape(-1,28,28,1) # (height = 28px, width = 28px , canal = 1)
print('Shape of X_train:', X_train.shape)
# One Hot Categories

y_train = to_categorical(y_train, num_classes = 10)
y_train
def baseline_model(layers = 1, 
                   filter_l1 = 32, 
                   filter_l2 = 64, 
                   filter_l3 = 128,
                   kernel_size_l1 = 3,
                   kernel_size_l2 = 3,
                   kernel_size_l3 = 3,
                   pool_size_l1 = 2,
                   pool_size_l2 = 2,
                   pool_size_l3 = 2,
                   activation_l1 = 'relu',
                   activation_l2 = 'relu',
                   activation_l3 = 'relu',
                   optimizer = 'Adamax',
                   dense_1 = 256,
                   dense_2 = 256,
                   dropout_1 = 0.25, 
                   dropout_2 = 0.25, 
                   dropout_3 = 0.4):
    
    # Create baseline
    
    model = Sequential()

    # First group
    #--------------------------------------------------
    model.add(Conv2D(filter_l1, (kernel_size_l1, kernel_size_l1), 
                     padding='same', activation=activation_l1,
                    input_shape = (28, 28, 1)))
    
    if (layers >= 2):
        for i in range(layers-1):
            model.add(Conv2D(filter_l1, (kernel_size_l1, kernel_size_l1), 
                             padding='same', activation=activation_l1))
        
    model.add(MaxPool2D(pool_size=(pool_size_l1, pool_size_l1)))
    
    # Second group
    #--------------------------------------------------
    model.add(Conv2D(filter_l2, (kernel_size_l2, kernel_size_l2), 
                     padding='same', activation=activation_l2))
    
    if (layers >= 2):
        for i in range(layers-1):
            model.add(Conv2D(filter_l2, (kernel_size_l2, kernel_size_l2), 
                             padding='same', activation=activation_l2))
        
    model.add(MaxPool2D(pool_size=(pool_size_l2, pool_size_l2)))
    
    # Third group
    #--------------------------------------------------
    model.add(Conv2D(filter_l3, (kernel_size_l3, kernel_size_l3), 
                     padding='same', activation=activation_l3))
    
    if (layers >= 2):
        for i in range(layers-1):
            model.add(Conv2D(filter_l3, (kernel_size_l3, kernel_size_l3),
                             padding='same', activation=activation_l3))
        
    model.add(MaxPool2D(pool_size=(pool_size_l3, pool_size_l3)))
    
    #--------------------------------------------------
    model.add(Dropout(dropout_1))
    model.add(Flatten())
              
    model.add(Dense(output_dim=dense_1, activation='relu'))
    model.add(Dropout(dropout_2))
              
    model.add(Dense(output_dim=dense_2, activation='relu'))
    model.add(Dropout(dropout_3))
    
    #--------------------------------------------------
    model.add(Dense(10, activation = "softmax"))
    
    # Compile the baseline model including the optimizer and evaluating 
    # the performance of the baseline by accuracy
    
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    return model
# If after the third epoch we didn't have an improvement of validation accuracy, 
# the learning rate will be decreased by 50% (factor).

lr_reduction = ReduceLROnPlateau(monitor='val_acc',
                                 patience=3, 
                                 verbose=0, 
                                 factor=0.5, 
                                 min_lr=0.00001)
# The idea is to alter the training data with small transformations to reproduce the variations 
# occuring when someone is writing a digit. It's a way to minimize the overfitting of the model.

generator = ImageDataGenerator(featurewise_center = False,
                               samplewise_center = False, 
                               featurewise_std_normalization = False,
                               samplewise_std_normalization = False,
                               zca_whitening = False,
                               rotation_range = 10, # Rotate image in 10 degrees
                               zoom_range = 0.10, # Zoom image (10% zoom) 
                               width_shift_range = 0.10, # Shift image horizontally (10% of width)
                               height_shift_range = 0.10, # Shift image vertically (10% of height)
                               horizontal_flip = False,
                               vertical_flip = False)
# I tried to use 5 epochs, but sometimes I didn't get to the finish line. 

epochs = 5
batch_size = 90

# Dictionary of parameters that gave the best results

best = {}
X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X_train, y_train, test_size = 0.1)
# -----------------------------------------------------
# Test #1 (Layers and Filters)
# -----------------------------------------------------

max_score = 0.0

# Define the grid search parameters

# First group to be evaluated

layers = [1, 2, 3]
filter_l1 = [32, 64]
filter_l2 = [32, 64]
filter_l3 = [32, 64, 128]

# Default parameters from baseline

'''
kernel_size_l1 = [5]
kernel_size_l2 = [3]
kernel_size_l3 = [3]
activation_l1 = ['relu']
activation_l2 = ['relu']
activation_l3 = ['relu']
pool_size_l1 = [2]
pool_size_l2 = [2]
pool_size_l3 = [2]
optimizer = ['Adamax']
dense_1 = [256]
dense_2 = [256]
dropout_1 = [0.25]
dropout_2 = [0.25]
dropout_3 = [0.4]
'''

# Data Augmentation
generator.fit(X_train_aux)

start_time = time()
print('Starting Test #1 (Epochs: {0})...'.format(epochs))
for layer in layers:
    for f_l1 in filter_l1:
        for f_l2 in filter_l2:
            for f_l3 in filter_l3:
                if ((f_l3 < f_l1) or (f_l3 < f_l2) or (f_l2 < f_l1)): continue
                digits = baseline_model(layers=layer,
                                        filter_l1=f_l1,
                                        filter_l2=f_l2,
                                        filter_l3=f_l3)
                                
                history = digits.fit_generator(generator.flow(X_train_aux,
                                                              y_train_aux, 
                                                              batch_size = batch_size),
                                               epochs = epochs, 
                                               steps_per_epoch = X_train_aux.shape[0] // batch_size, 
                                               validation_data = (X_test_aux, y_test_aux),
                                               callbacks = [lr_reduction],
                                               verbose=0)
                
                print("Layer: {0}, Filter L1: {1:>3d}, Filter L2: {2:>3d}, Filter L3: {3:>3d} :: (Max. Val. Acc. = {4:.5f})".format(layer, f_l1, f_l2, f_l3,
                                                                                                                                    max(history.history['val_acc'])))
                
                if max(history.history['val_acc']) > max_score:
                    best['layers'] = layer
                    best['filter_l1'] = f_l1
                    best['filter_l2'] = f_l2
                    best['filter_l3'] = f_l3
                    max_score = max(history.history['val_acc'])
                    
end_time = time()
duration = end_time - start_time
hours, rest = divmod(duration, 3600)
minutes, seconds = divmod(rest, 60)
print('Test #1 - duration: {0} h {1} min {2:.2f} sec'.format(int(hours), int(minutes), seconds))
print('-----------------------------------------------------------------------')
print('Best score: {0:.5f}'.format(max_score))
print('Best parameters: layers = {0}, filter_l1 = {1}, filter_l2 = {2}, filter_l3 = {3}'.format(best['layers'],best['filter_l1'], best['filter_l2'], best['filter_l3']))
print('-----------------------------------------------------------------------')
X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X_train, y_train, test_size = 0.1)
# -----------------------------------------------------
# Test #2 (Kernel Size & best parameters from Test #1)
# -----------------------------------------------------

max_score = 0.0

# Define the grid search parameters

# Second group to be evaluated

kernel_size_l1 = [3, 4, 5]
kernel_size_l2 = [3, 4, 5]
kernel_size_l3 = [3, 4, 5]

# Default parameters from baseline

'''
activation_l1 = ['relu']
activation_l2 = ['relu']
activation_l3 = ['relu']
pool_size_l1 = [2]
pool_size_l2 = [2]
pool_size_l3 = [2]
optimizer = ['Adamax']
dense_1 = [256]
dense_2 = [256]
dropout_1 = [0.25]
dropout_2 = [0.25]
dropout_3 = [0.4]
'''

# Data Augmentation
generator.fit(X_train_aux)

start_time = time()
print('Starting Test #2 (Epochs: {0})...'.format(epochs))
for k_l1 in kernel_size_l1:
    for k_l2 in kernel_size_l2:
        for k_l3 in kernel_size_l3:
            digits = baseline_model(layers=best['layers'],
                                    filter_l1=best['filter_l1'],
                                    filter_l2=best['filter_l2'],
                                    filter_l3=best['filter_l3'],
                                    kernel_size_l1=k_l1,
                                    kernel_size_l2=k_l2,
                                    kernel_size_l3=k_l3)
            history = digits.fit_generator(generator.flow(X_train_aux,
                                                          y_train_aux, 
                                                          batch_size = batch_size),
                                           epochs = epochs, 
                                           steps_per_epoch = X_train_aux.shape[0] // batch_size, 
                                           validation_data = (X_test_aux, y_test_aux),
                                           callbacks = [lr_reduction],
                                           verbose=0)
                        
            print("Kernel L1: {0}, Kernel L2: {1}, Kernel L3: {2} :: (Max. Val. Acc = {3:.5f})".format(k_l1, k_l2, k_l3, max(history.history['val_acc'])))
                       
            if max(history.history['val_acc']) > max_score:
                best['kernel_size_l1'] = k_l1
                best['kernel_size_l2'] = k_l2
                best['kernel_size_l3'] = k_l3
                max_score = max(history.history['val_acc'])
                            
end_time = time()
duration = end_time - start_time
hours, rest = divmod(duration, 3600)
minutes, seconds = divmod(rest, 60)
print('Test #2 - duration: {0} h {1} min {2:.2f} sec'.format(int(hours), int(minutes), seconds))
print('-----------------------------------------------------------------------')
print('Best score: {0:.5f}'.format(max_score))
print('Best parameters: kernel_size_l1 = {0}, kernel_size_l2 = {1}, kernel_size_l3 = {2}'.format(best['kernel_size_l1'], best['kernel_size_l2'], best['kernel_size_l3']))
print('-----------------------------------------------------------------------')
X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X_train, y_train, test_size = 0.1)
# -----------------------------------------------------
# Test #3 (Activation functions & best parameters from Tests #1 and #2)
# -----------------------------------------------------

max_score = 0.0

# Define the grid search parameters

# Third group to be evaluated

activation_l1 = ['relu', 'sigmoid']
activation_l2 = ['relu', 'sigmoid']
activation_l3 = ['relu', 'sigmoid']

# Default parameters from baseline

'''
pool_size_l1 = [2]
pool_size_l2 = [2]
pool_size_l3 = [2]
optimizer = ['Adamax']
dense_1 = [256]
dense_2 = [256]
dropout_1 = [0.25]
dropout_2 = [0.25]
dropout_3 = [0.4]
'''

# Data Augmentation
generator.fit(X_train_aux)

start_time = time()
print('Starting Test #3 (Epochs: {0})...'.format(epochs))
for act_l1 in activation_l1:
    for act_l2 in activation_l2:
        for act_l3 in activation_l3:
            digits = baseline_model(layers=best['layers'],
                                    filter_l1=best['filter_l1'],
                                    filter_l2=best['filter_l2'],
                                    filter_l3=best['filter_l3'],
                                    kernel_size_l1=best['kernel_size_l1'],
                                    kernel_size_l2=best['kernel_size_l2'],
                                    kernel_size_l3=best['kernel_size_l3'],                                   
                                    activation_l1=act_l1,
                                    activation_l2=act_l2,
                                    activation_l3=act_l3)
            history = digits.fit_generator(generator.flow(X_train_aux,
                                                          y_train_aux, 
                                                          batch_size = batch_size),
                                           epochs = epochs,
                                           steps_per_epoch = X_train_aux.shape[0] // batch_size, 
                                           validation_data = (X_test_aux, y_test_aux),
                                           callbacks = [lr_reduction],
                                           verbose=0)
            
            print("Act. L1: {0}, Act. L2: {1}, Act. L3: {2} :: (Max. Val. Acc = {3:.5f})".format(act_l1, act_l2, act_l3, max(history.history['val_acc'])))
            
            if max(history.history['val_acc']) > max_score:
                best['activation_l1'] = act_l1
                best['activation_l2'] = act_l2
                best['activation_l3'] = act_l3
                max_score = max(history.history['val_acc'])
                            
end_time = time()
duration = end_time - start_time
hours, rest = divmod(duration, 3600)
minutes, seconds = divmod(rest, 60)
print('Test #3 - duration: {0} h {1} min {2:.2f} sec'.format(int(hours), int(minutes), seconds))
print('-----------------------------------------------------------------------')
print('Best score: {0:.5f}'.format(max_score))
print('Best parameters: activation_l1 = {0}, activation_l2 = {1}, activation_l3 = {2}'.format(best['activation_l1'], best['activation_l2'], best['activation_l3']))
print('-----------------------------------------------------------------------')
X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X_train, y_train, test_size = 0.1)
# -----------------------------------------------------
# Test #4 (Pool Size and Dense & best parameters from Tests #1, #2 and #3)
# -----------------------------------------------------

max_score = 0.0

# Define the grid search parameters

# Forth group to be evaluated

pool_size_l1 = [2, 3]
pool_size_l2 = [2, 3]
pool_size_l3 = [2, 3]
dense_1 = [128, 256]
dense_2 = [128, 256]

# Default parameters from baseline

'''
optimizer = ['Adamax']
dropout_1 = [0.25]
dropout_2 = [0.25]
dropout_3 = [0.4]
'''

# Data Augmentation
generator.fit(X_train_aux)

start_time = time()
print('Starting Test #4 (Epochs: {0})...'.format(epochs))
for p_l1 in pool_size_l1:
    for p_l2 in pool_size_l2:
        for p_l3 in pool_size_l3:
            for d1 in dense_1:
                for d2 in dense_2:
                    digits = baseline_model(layers=best['layers'],
                                            filter_l1=best['filter_l1'],
                                            filter_l2=best['filter_l2'],
                                            filter_l3=best['filter_l3'],
                                            kernel_size_l1=best['kernel_size_l1'],
                                            kernel_size_l2=best['kernel_size_l2'],
                                            kernel_size_l3=best['kernel_size_l3'],
                                            activation_l1=best['activation_l1'],
                                            activation_l2=best['activation_l2'],
                                            activation_l3=best['activation_l3'],
                                            pool_size_l1=p_l1,
                                            pool_size_l2=p_l2,
                                            pool_size_l3=p_l3,
                                            dense_1=d1,
                                            dense_2=d2)

                    history = digits.fit_generator(generator.flow(X_train_aux,
                                                                  y_train_aux, 
                                                                  batch_size = batch_size),
                                                   epochs = epochs, 
                                                   steps_per_epoch = X_train_aux.shape[0] // batch_size, 
                                                   validation_data = (X_test_aux, y_test_aux),
                                                   callbacks = [lr_reduction],
                                                   verbose=0)  

                    print("Pool Size L1: {0}, Pool Size L2: {1}, Pool Size L3: {2}, Dense 1: {3}, Dense 2: {4} :: (Max. Val. Acc = {5:.5f})".format(p_l1, p_l2, p_l3, d1, d2, max(history.history['val_acc'])))

                    if max(history.history['val_acc']) > max_score:
                        best['pool_size_l1'] = p_l1
                        best['pool_size_l2'] = p_l2
                        best['pool_size_l3'] = p_l3
                        best['dense_1'] = d1
                        best['dense_2'] = d2
                        max_score = max(history.history['val_acc'])
                
end_time = time()
duration = end_time - start_time
hours, rest = divmod(duration, 3600)
minutes, seconds = divmod(rest, 60)
print('Test #4 - duration: {0} h {1} min {2:.2f} sec'.format(int(hours), int(minutes), seconds))
print('-----------------------------------------------------------------------')
print('Best score: {0:.5f}'.format(max_score))
print('Best parameters: pool_size_l1 = {0}, pool_size_l2 = {1}, pool_size_l3 = {2}, dense_1 = {3}, dense_2 = {4}'.format(best['pool_size_l1'], best['pool_size_l2'], best['pool_size_l3'], best['dense_1'], best['dense_2']))
print('-----------------------------------------------------------------------')
X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X_train, y_train, test_size = 0.1)
# -----------------------------------------------------
# Test #5 (Dropout & best parameters from Tests #1, #2, #3 and #4)
# -----------------------------------------------------

max_score = 0.0

# Define the grid search parameters

# Fifth group to be evaluated

dropout_1 = [0.25, 0.4, 0.5]
dropout_2 = [0.25, 0.4, 0.5]
dropout_3 = [0.25, 0.4, 0.5]

# Default parameters from baseline

'''
optimizer = ['Adamax']
'''

# Data Augmentation
generator.fit(X_train_aux)

start_time = time()
print('Starting Test #5 (Epochs: {0})...'.format(epochs))
for drop_1 in dropout_1:
    for drop_2 in dropout_2:
        for drop_3 in dropout_3:
            digits = baseline_model(layers=best['layers'],
                                    filter_l1=best['filter_l1'],
                                    filter_l2=best['filter_l2'],
                                    filter_l3=best['filter_l3'],
                                    kernel_size_l1=best['kernel_size_l1'],
                                    kernel_size_l2=best['kernel_size_l2'],
                                    kernel_size_l3=best['kernel_size_l3'],
                                    activation_l1=best['activation_l1'],
                                    activation_l2=best['activation_l2'],
                                    activation_l3=best['activation_l3'],
                                    pool_size_l1=best['pool_size_l1'],
                                    pool_size_l2=best['pool_size_l2'],
                                    pool_size_l3=best['pool_size_l3'],
                                    dense_1=best['dense_1'],
                                    dense_2=best['dense_2'],
                                    dropout_1=drop_1,
                                    dropout_2=drop_2,
                                    dropout_3=drop_3)
            
            history = digits.fit_generator(generator.flow(X_train_aux,
                                                          y_train_aux, 
                                                          batch_size = batch_size),
                                           epochs = epochs, 
                                           steps_per_epoch = X_train_aux.shape[0] // batch_size, 
                                           validation_data = (X_test_aux, y_test_aux),
                                           callbacks = [lr_reduction],
                                           verbose=0)  
                    
            print("Drop 1: {0}, Drop 2: {1}, Drop 3: {2} :: (Max. Val. Acc = {3:.5f})".format(drop_1, drop_2, drop_3, max(history.history['val_acc'])))
            
            if max(history.history['val_acc']) > max_score:
                best['dropout_1'] = drop_1
                best['dropout_2'] = drop_2
                best['dropout_3'] = drop_3
                max_score = max(history.history['val_acc'])
                        
end_time = time()
duration = end_time - start_time
hours, rest = divmod(duration, 3600)
minutes, seconds = divmod(rest, 60)
print('Test #5 - duration: {0} h {1} min {2:.2f} sec'.format(int(hours), int(minutes), seconds))
print('-----------------------------------------------------------------------')
print('Best score: {0:.5f}'.format(max_score))
print('Best parameters: dropout_1 = {0}, dropout_2 = {1}, dropout_3 = {2}'.format(best['dropout_1'], best['dropout_2'], best['dropout_3']))
print('-----------------------------------------------------------------------')
X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X_train, y_train, test_size = 0.1)
# -----------------------------------------------------
# Test #6 (Optimizers & best parameters from Tests #1, #2, #3, #4 and #5)
# -----------------------------------------------------

max_score = 0.0

# Define the grid search parameters

# Sixth group to be evaluated

optimizer = ['Adamax', 'Adam', 'RMSProp', 'sgd', 'Adagrad', 'Adadelta', 'Nadam']

# Data Augmentation
generator.fit(X_train_aux)

start_time = time()
print('Starting Test #6 (Epochs: {0})...'.format(epochs))
for opt in optimizer:
    digits = baseline_model(layers=best['layers'],
                            filter_l1=best['filter_l1'],
                            filter_l2=best['filter_l2'],
                            filter_l3=best['filter_l3'],
                            kernel_size_l1=best['kernel_size_l1'],
                            kernel_size_l2=best['kernel_size_l2'],
                            kernel_size_l3=best['kernel_size_l3'],
                            activation_l1=best['activation_l1'],
                            activation_l2=best['activation_l2'],
                            activation_l3=best['activation_l3'],
                            pool_size_l1=best['pool_size_l1'],
                            pool_size_l2=best['pool_size_l2'],
                            pool_size_l3=best['pool_size_l3'],
                            dense_1=best['dense_1'],
                            dense_2=best['dense_2'],
                            dropout_1=best['dropout_1'],
                            dropout_2=best['dropout_2'],
                            dropout_3=best['dropout_3'],
                            optimizer=opt)
    history = digits.fit_generator(generator.flow(X_train_aux,
                                                      y_train_aux, 
                                                      batch_size = batch_size),
                                       epochs = epochs, 
                                       steps_per_epoch = X_train_aux.shape[0] // batch_size, 
                                       validation_data = (X_test_aux, y_test_aux),
                                       callbacks = [lr_reduction],
                                       verbose=0)
    print("Optimizer: {0} :: (Max. Val. Acc = {1:.5f})".format(opt, max(history.history['val_acc'])))
        
    if max(history.history['val_acc']) > max_score:
        best['optimizer'] = opt
        max_score = max(history.history['val_acc'])
                        
end_time = time()
duration = end_time - start_time
hours, rest = divmod(duration, 3600)
minutes, seconds = divmod(rest, 60)
print('Test #6 - duration: {0} h {1} min {2:.2f} sec'.format(int(hours), int(minutes), seconds))
print('-----------------------------------------------------------------------')
print('Best score: {0:.5f}'.format(max_score))
print('Best parameters: optimizer = {0}'.format(best['optimizer']))
print('-----------------------------------------------------------------------')
X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X_train, y_train, test_size = 0.1)
# -----------------------------------------------------
# Test #7 (Image Generator - Data Augmentation
# -----------------------------------------------------

max_score = 0.0

# Seventh group to be evaluated

augmentation = [5, 10, 15, 20]

digits = baseline_model(layers=best['layers'],
                        filter_l1=best['filter_l1'],
                        filter_l2=best['filter_l2'],
                        filter_l3=best['filter_l3'],
                        kernel_size_l1=best['kernel_size_l1'],
                        kernel_size_l2=best['kernel_size_l2'],
                        kernel_size_l3=best['kernel_size_l3'],
                        activation_l1=best['activation_l1'],
                        activation_l2=best['activation_l2'],
                        activation_l3=best['activation_l3'],
                        pool_size_l1=best['pool_size_l1'],
                        pool_size_l2=best['pool_size_l2'],
                        pool_size_l3=best['pool_size_l3'],
                        dense_1=best['dense_1'],
                        dense_2=best['dense_2'],
                        dropout_1=best['dropout_1'],
                        dropout_2=best['dropout_2'],
                        dropout_3=best['dropout_3'],
                        optimizer=best['optimizer'])

start_time = time()
print('Starting Test #7 (Epochs: {0})...'.format(epochs))
for aug in augmentation:
    
    # Data Augmentation
    generator = ImageDataGenerator(featurewise_center = False,
                               samplewise_center = False, 
                               featurewise_std_normalization = False,
                               samplewise_std_normalization = False,
                               zca_whitening = False,
                               rotation_range = aug, # Rotate image in aug degrees
                               zoom_range = aug/100, # Zoom image (aug% zoom) 
                               width_shift_range = aug/100, # Shift image horizontally (aug% of width)
                               height_shift_range = aug/100, # Shift image vertically (aug% of height)
                               horizontal_flip = False,
                               vertical_flip = False)
    generator.fit(X_train_aux)
    
    history = digits.fit_generator(generator.flow(X_train_aux,
                                                  y_train_aux, 
                                                  batch_size = batch_size),
                                   epochs = epochs, 
                                   steps_per_epoch = X_train_aux.shape[0] // batch_size, 
                                   validation_data = (X_test_aux, y_test_aux),
                                   callbacks = [lr_reduction],
                                   verbose=0)
    print("Augmentation: {0} :: (Max. Val. Acc = {1:.5f})".format(aug, max(history.history['val_acc'])))
        
    if max(history.history['val_acc']) > max_score:
        best['augmentation'] = aug
        max_score = max(history.history['val_acc'])
                        
end_time = time()
duration = end_time - start_time
hours, rest = divmod(duration, 3600)
minutes, seconds = divmod(rest, 60)
print('Test #7 - duration: {0} h {1} min {2:.2f} sec'.format(int(hours), int(minutes), seconds))
print('-----------------------------------------------------------------------')
print('Best score: {0:.5f}'.format(max_score))
print('Best parameters: augmentation = {0}'.format(best['augmentation']))
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('Best parameters of this CNN:', best)
print('-----------------------------------------------------------------------')