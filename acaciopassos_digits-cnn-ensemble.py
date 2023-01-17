%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from keras.models import model_from_json

import warnings
warnings.filterwarnings("ignore")

# My seed

seed = 42
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
df_train.info()
df_test.info()
X_train = df_train.drop(['label'], axis=1)
y_train = df_train['label']
X_test = df_test

# Free memory space

del df_train
del df_test

print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test :', X_test.shape)

counter = Counter(y_train)
counter
X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.values.reshape(-1,28,28,1) # (height = 28px, width = 28px , canal = 1)
X_test = X_test.values.reshape(-1,28,28,1)

print('Shape of X_train:', X_train.shape)
print('Shape of X_test :', X_test.shape)
# One Hot Categories

y_train = to_categorical(y_train, num_classes = 10)
y_train.shape
def baseline_model():
    
    # Create baseline
    
    baseline = Sequential()

    #------------------------------------------------------------
    
    # Parameters tunned by GridSearchCV    
    # 32 filters for the three firsts conv2D layers
    
    baseline.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation ='relu', 
                     input_shape = (28, 28, 1)))
    baseline.add(BatchNormalization())
    baseline.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation ='relu'))
    baseline.add(BatchNormalization())
    baseline.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation ='relu'))
    baseline.add(BatchNormalization())
    
    # This layer simply acts as a downsampling filter. 
    # It looks at the 2 neighboring pixels and picks the maximal value, reducing computational cost, 
    # and to some extent also reduce overfitting.
    
    # IMPORTANT: Combining convolutional and pooling layers, CNN are able to combine local features and 
    # learn more global features of the image.
    
    baseline.add(MaxPool2D(pool_size=(2,2)))
    
    # Dropout is a regularization method, where a proportion of nodes (25%) in the layer are randomly ignored 
    # for each training sample. This dropout forces the network to learn features in a distributed way 
    # and improves generalization and reduces the overfitting.
    
    baseline.add(Dropout(0.25))
    #------------------------------------------------------------
    
    # 64 filters for the three last conv2D layers
    
    baseline.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
    baseline.add(BatchNormalization())
    baseline.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
    baseline.add(BatchNormalization())
    baseline.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
    baseline.add(BatchNormalization())
    
    baseline.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    baseline.add(Dropout(0.25))
    #------------------------------------------------------------

    # The Flatten layer is use to convert the final feature maps into a one single 1D vector. 
    # IMPORTANT: It combines all the found local features of the previous convolutional layers.
    
    baseline.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation ='sigmoid'))
    baseline.add(BatchNormalization())
    
    baseline.add(Flatten())
    baseline.add(Dense(128, activation = "relu"))
    baseline.add(Dropout(0.4))
    
    # The net outputs distribution of probability of each class --> In our case, 10 output classes
    
    baseline.add(Dense(10, activation = "softmax"))
    
    # The optimizer will iteratively improve parameters in order to minimize the loss.
    # Compile the baseline including the optimizer and evaluating the performance of the baseline by accuracy
    
    baseline.compile(optimizer = 'Adamax' , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    return baseline
# If after the third epoch we didn't have an improvement of accuracy, the learning rate will be 
# reduced by 50% (factor).

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

generator.fit(X_train)
nets = 10
digits = [0] * nets
history = [0] * nets

epochs = 40
batch_size = 90
print('Creating {0} CNNs...'.format(nets))
for model in range(nets):
    digits[model] = baseline_model()
    
    # Splitting train and test datasets
    
    X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X_train, y_train, test_size = 0.1)
    
    history[model] = digits[model].fit_generator(generator.flow(X_train_aux,
                                                              y_train_aux, 
                                                              batch_size = batch_size),
                                                 epochs = epochs, 
                                                 steps_per_epoch = X_train_aux.shape[0] // batch_size, 
                                                 validation_data = (X_test_aux, y_test_aux), 
                                                 callbacks=[lr_reduction],
                                                 verbose=0)
    
    print("CNN {0:>2d}: Epochs = {1:d}, Max. Train accuracy = {2:.5f}, Max. Validation accuracy = {3:.5f}".format(
        model + 1, # Number of the CNN model
        epochs, # Total of epochs
        max(history[model].history['acc']), # Maximum Accuracy from Training
        max(history[model].history['val_acc']))) # Maximum Accuracy from Test (validation)
label_predicted = np.zeros( (X_test.shape[0], 10) )
label_predicted
for model in range(nets):
    label_predicted = label_predicted + digits[model].predict(X_test)
# Get the index with the maximum probability

label_predicted = np.argmax(label_predicted, axis = 1)
label_predicted = pd.Series(label_predicted, name = "Label")
solution = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), label_predicted], axis = 1)
solution.to_csv("solution_cnn_opt.csv", index=False)
solution.head(10)
for model in range(nets):
    model_saved = digits[model].to_json()
    name = 'model_' + str(model) + '.json'
    with open(name, 'w') as json_file:
        json_file.write(model_saved)
    name = 'model_' + str(model) + '.h5'
    digits[model].save_weights(name)
