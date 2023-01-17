# Author: Pablo Leo 
# data handling
import pandas as pd
import numpy as np
import math

# model selection
from sklearn.model_selection import train_test_split

# ML classifiers
from sklearn.ensemble import RandomForestClassifier

# plot
import matplotlib.pyplot as plt

# system
import os

# deep learning
import keras
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
data_dir = '../input'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'), dtype = np.uint8)
x_train = df_train.drop('label', axis = 1).values # attributes
y_train = df_train['label'].values # labels
# get the labels and the counts
vals, counts = np.unique(y_train, return_counts = True)
plt.bar(vals, counts)
plt.show()
print('Label | Counts')
print('--------------')
for val, count in zip(vals, counts):
    print('    {}: {}'.format(val, count)) 
def plot_numbers(x, y, n_cols = 6, one_hot = False):
    """
    Function that plots a given set of images 'x' of 28x28 pixels
    """
    
    # number of rows and cols
    if one_hot:
        uniques = np.unique(np.argmax(y, axis = 1))
    else:
        uniques = np.unique(y)
        
    n_rows = len(uniques) # amount of different numbers (n_rows)

    # get random indexes
    indexes = np.random.randint(len(x), size = n_rows * n_cols)

    plt.figure(figsize=(6, 10))
    for i in range(n_rows):
        if one_hot:
            indexes = np.random.choice(np.where(np.argmax(y, axis = 1) == i)[0], n_cols)
        else:
            indexes = np.random.choice(np.where(y == i)[0], n_cols)
        
        for j in range(n_cols):
            # get the current case
            case = indexes[j]

            # generate a new subplot
            ax = plt.subplot(n_rows, n_cols, (i * n_cols) + j + 1)

            # get the image and label
            image = x[case].reshape(28,28)
                
            label = y[case]

            # plot the image and the label
            plt.imshow(image, cmap = 'gray')
            plt.axis('off')

    plt.show()
plot_numbers(x_train, y_train, n_cols = 7)
# normalize data
x_train_norm = x_train.reshape(len(x_train), 28, 28)/255

# convert into one_hot encoding
y_train_oneh = keras.utils.to_categorical(y_train, num_classes=10)
# check that the data was correctly normalized and the labels are in the correct encoding
plot_numbers(x_train_norm, y_train_oneh, n_cols = 7, one_hot = True)
x_train, x_val, y_train, y_val = train_test_split(x_train_norm, y_train_oneh, test_size = 0.2, shuffle = True)
def create_model_own():
    
    x_in = Input(shape = (28, 28, 1))
    
    x = Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform')(x_in)
    x = BatchNormalization()(x)
    x = Activation(activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x) # Reduce to -> 13x13
    x = Dropout(0.2)(x)
    
    x = Conv2D(filters = 64, kernel_size = (3, 3), kernel_initializer = 'he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation(activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x) # Reduce to -> 5x5
    x = Dropout(0.2)(x)
    
    x = Conv2D(filters = 128, kernel_size = (3, 3), kernel_initializer = 'he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation(activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x) # Reduce to -> 3x3(truncation)
    x = Dropout(0.2)(x)
    
    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation(activation = 'relu')(x)
    x = Dropout(0.4)(x)
    
    x = Flatten(name = 'last_Flat')(x)
    x = Dense(10, kernel_initializer = 'he_uniform')(x)
    x_out = Activation(activation = 'softmax')(x)
    
    model = Model(inputs = x_in, outputs = x_out)
    
    model.summary()
    
    return model
# get the model
model = create_model_own()

# compile variables
lr = 1e-3 # learning rate
op = Adam(lr) # optimizer
loss_func = 'categorical_crossentropy'

# compile the model
model.compile(op, loss_func, metrics = ['accuracy'])
checkpoint = ModelCheckpoint('best_model.h5',
                             monitor = 'val_loss',
                             verbose = 1,
                             save_best_only = True,
                             mode = 'min',
                             period = 1)
model = model.fit(np.expand_dims(x_train, axis = -1), y_train, 
                  batch_size = 200, 
                  epochs = 50,
                  validation_data = (np.expand_dims(x_val, axis = -1), y_val),
                  callbacks = [checkpoint])
# load best model
model = keras.models.load_model('best_model.h5')

# load test data
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
x_test = df_test.values.reshape(len(df_test), 28, 28, 1)/255
y_pred = model.predict(x_test, batch_size = 50, verbose = 1)
idx = 0
plt.imshow(x_test[idx].reshape(28, 28), cmap = 'gray')
plt.show()
print(np.argmax(y_pred[idx]))
y_pred_arg = np.argmax(y_pred, axis = 1)
y_id = np.arange(1, len(y_pred) + 1)
df_test = pd.DataFrame(data = np.stack((y_id, y_pred_arg), axis = 1), columns = ['ImageId', 'Label'])
df_test.to_csv('submission.csv', index = False)