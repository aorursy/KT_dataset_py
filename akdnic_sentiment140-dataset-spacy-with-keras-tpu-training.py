# Install the required libraries



import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import time

from datetime import timedelta

import pickle



print("Tensorflow version " + tf.__version__)
# list all the loaded datasets



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

AUTO = tf.data.experimental.AUTOTUNE



# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Load tweet embeddings created from spaCy NLP model 'en_vectors_web_lg'

# and assign it to X, the training variable



pickle_in = open("../input/sentiment140-dataset-16m-tweets-spacy-embeddings/Spacy_training_full.pickle","rb")    

df = pickle.load(pickle_in)

print(df.shape)

X = pd.DataFrame(data={'tweet_embeddings':[target for target in df]})

X['tweet_embeddings'].head()
X['tweet_embeddings'].tail()
# Load target training labels and assign it to y variable



pickle_in = open("../input/sentiment140-dataset-16m-tweets-spacy-embeddings/Spacy_target_training_full.pickle","rb")    

df1 = pickle.load(pickle_in)

print(df1.shape)

y = pd.DataFrame(data={'labels':[target for target in df1]})

y['labels'].head()
y['labels'].tail()
X['labels'] = y['labels']

X.head()
X.tail()
# Give three-four proper rounds of shuffles



X = X.sample(frac=1)

X = X.sample(frac=1)

X = X.sample(frac=1)

X.tail()
print(len(X), X.shape)

y = np.array([label for label in X['labels']])

print(y.shape)

X = np.array([embeding for embeding in X['tweet_embeddings']])

print(X.shape)
# Another variation of splitting the dataset into train and test subsets, takes up less memory than sklearn_train_test_split function

def shuffle(matrix, target, test_proportion):

    ratio = int(matrix.shape[0]/test_proportion)        # should be int

    X_train = matrix[ratio:,:]

    X_test =  matrix[:ratio,:]

    y_train = target[ratio:,:]

    y_test =  target[:ratio,:]

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = shuffle(X, y, 10)     # splits the dataset in the ratio of 1/10
## If using Conv1D layer, reshape the data from 2-dim to 3-dim to feed as an input

## If not using Conv1D, might just comment out this section and the Conv1D layers during model building

X_train = np.reshape(X_train, X_train.shape + (1,))

X_test = np.reshape(X_test, X_test.shape + (1,))



print(X_train.shape, X_test.shape)

print(y_train.shape, y_test.shape)
### Build Keras Deep Learning model



# Import the layers that you want to use to build model and visualize key metrics



from keras.models import Sequential

from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import optimizers

from sklearn.model_selection import KFold           # To be used for cross-validation

from matplotlib import pyplot

from keras.callbacks import Callback
# K-fold Cross Validation model building



num_folds = 10                  # Select the number of folds for cross-validation

acc_per_fold = []

loss_per_fold = []



# Merge inputs and targets

inputs = np.concatenate((X_train, X_test), axis=0)

targets = np.concatenate((y_train, y_test), axis=0)



# Define the K-fold Cross Validator

kfold = KFold(n_splits=num_folds, shuffle=True)



fold_no = 1


## Try different combinations of layers and their respective hyperparameters

## Comment out the layers which are not desired. Also, keep an eye on batch_input_shape parameter for Conv1D    

# instantiating the model in the strategy scope creates the model on the TPU



def create_model(a, b):

    with strategy.scope():

        model = tf.keras.Sequential([

             tf.keras.layers.Dropout(0.2, batch_input_shape=(None,300,1)),

             tf.keras.layers.Conv1D(128,8, activation='relu', use_bias=True, bias_initializer='zeros', batch_input_shape=(None,300,1)), 

             tf.keras.layers.Conv1D(64,4, activation='relu', use_bias=True, bias_initializer='zeros', batch_input_shape=(None,300,1)),

             tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.0001),

             tf.keras.layers.Flatten(),

             tf.keras.layers.Dropout(0.2),

             tf.keras.layers.Dense(128,activation='relu'),

             tf.keras.layers.Dense(64,activation='relu'),

             tf.keras.layers.Dense(2,activation='sigmoid'),        

         ])



        # Setup the optimizer

        sgd = tf.keras.optimizers.SGD(lr=0.05, momentum=0.9, nesterov=True)



        # Use EarlyStopping and ReduceLROnPlateau callbacks if needed

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        relr = ReduceLROnPlateau(monitor='val_loss', factor=0.125, patience=15, min_lr=0.0000001, mode='min', verbose=1, min_delta=1E-6)



        # Save best model per fold to be used later on for comparison, cross validation and predictions

        ch = ModelCheckpoint(f'bestmodel_{fold_no}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',period=2)



        callbacks_list = [es,ch,relr]



        model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])



        return model, callbacks_list

        

# Using a batch size of 2000 * 8

my_batch_size = 2000 * strategy.num_replicas_in_sync



fold_no = 1



# Define empty variables to obtain compiled model and callbacks from the create_model(a, b) function

a = None

b = []



# Start the iteration of training and fitting for each Fold, Best_model and Final_model versions of the trained models

# will be saved in the output directory



for train, test in kfold.split(inputs, targets):

    

    start_time = time.monotonic()

    

    print(f'Creating a model for Fold No. {fold_no}')

    model, callbacks_list = create_model(a, b)

    model.summary()

    

    print('------------------------------------------------------------------------')

    print(f'Training for fold {fold_no} ...')



    # Fit data to model

    history = model.fit(inputs[train],targets[train],epochs=2000, verbose=2, batch_size=my_batch_size, validation_split=0.1, callbacks=callbacks_list)

    

    # Generate generalization metrics

    scores = model.evaluate(inputs[test], targets[test], verbose=1)

    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

    acc_per_fold.append(scores[1] * 100)

    loss_per_fold.append(scores[0])

    print(f'Saving final model per fol... {fold_no}')

    model.save(f'Fold_{fold_no}.hdf5')

    

    #Plotting charts

    print('\Accuracy Chart\n')

    pyplot.plot(history.history['accuracy'], label='training_accuracy')

    pyplot.plot(history.history['val_accuracy'], label='validation_accuracy')

    pyplot.show() 

    

    print('\Loss Chart\n')

    pyplot.plot(history.history['loss'], label='training_loss')    

    pyplot.plot(history.history['val_loss'], label='validation_loss')

    pyplot.show() 

        

    end_time = time.monotonic()

    print(f'Time taken to train on {fold_no}: {timedelta(seconds=end_time - start_time)}')

        

    # Increase fold number

    fold_no+=1

    





# == Provide average scores ==

print('------------------------------------------------------------------------')

print('Score per fold')

print('Saving final model finally !!!.....')

model.save("Final.hdf5")



for i in range(0, len(acc_per_fold)):

    print('------------------------------------------------------------------------')

    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')



print('------------------------------------------------------------------------')

print('Average scores for all folds:')

print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')

print(f'> Loss: {np.mean(loss_per_fold)}')

print('------------------------------------------------------------------------')



    



# Evaluate the model using loss and accuracy metrics

model.evaluate(x=X_test,y=y_test)
# Make predictions on X_test using latest updated trained model

predictions = model.predict_classes(X_test)

print(predictions)
print(y_test.argmax(axis=1))
# Import metrics from sklearn

from sklearn import metrics



# Print a confusion matrix

print(metrics.confusion_matrix(y_test.argmax(axis=1),predictions))
# Print a classification report

print(metrics.classification_report(y_test.argmax(axis=1),predictions))
# Print the overall accuracy

print(metrics.accuracy_score(y_test.argmax(axis=1),predictions))