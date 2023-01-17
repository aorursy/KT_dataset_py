# import dependencies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.utils import plot_model

from IPython.display import Image

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



SEED = 999

np.random.seed(SEED)

sns.set(style='white', context='notebook', palette='pastel')
X_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')
Y_train = X_train['label']

X_train = X_train.drop(labels=['label'], axis=1) 



print(X_train.isnull().any().describe())



# normalize

X_train = X_train / 255.0

X_test = X_test / 255.0



# resize

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
# basic frequency plots

g = sns.countplot(Y_train)

Y_train.value_counts()
# preview an image

print('Label:', Y_train[0])

g = plt.imshow(X_train[0][:,:,0])
# one-hot encoding

Y_train = to_categorical(Y_train, num_classes=10)



# Split the data into training and validation sets

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,

                                                  test_size=0.1, random_state=SEED)
def create_models(num_nets):

    models = [0] * num_nets



    for i in range(num_nets):

        models[i] = Sequential()



        models[i].add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu',

                     input_shape=(28,28,1)))

        models[i].add(BatchNormalization())

        models[i].add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

        models[i].add(BatchNormalization())

        models[i].add(MaxPool2D(pool_size=(2,2)))

        models[i].add(Dropout(0.25))



        models[i].add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))

        models[i].add(BatchNormalization())

        models[i].add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))

        models[i].add(BatchNormalization())

        models[i].add(MaxPool2D(pool_size=(2,2)))

        models[i].add(Dropout(0.25))



        models[i].add(Flatten())

        models[i].add(Dense(256, activation='relu'))

        models[i].add(Dropout(0.4))

        models[i].add(Dense(10, activation='softmax'))

    

        models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print('Compiled model {}.'.format(i+1))

    

    return num_nets, models
num_nets, models = create_models(num_nets=7)
plot_model(models[0], to_file='model.png', show_shapes=True, show_layer_names=True)

Image('model.png')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, 

                                            factor=0.5, min_lr=0.00001)
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.10, 

                             width_shift_range=0.10, height_shift_range=0.10)
def train_models(models, epochs):

    num_nets = len(models)

    history = [0] * num_nets

    

    for i in range(num_nets):

        X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)

        history[i] = models[i].fit_generator(datagen.flow(X_train2, Y_train2, batch_size=64),

                                            epochs=epochs, steps_per_epoch=X_train2.shape[0]//64,  

                                            validation_data=(X_val2,Y_val2),

                                            callbacks=[learning_rate_reduction], verbose=0)

        

        print('CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}'.format(

            i+1, epochs, max(history[i].history['accuracy']), max(history[i].history['val_accuracy'])))

    

    return models, history
models, history = train_models(models=models, epochs=18)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment='center',

                 color='white' if cm[i, j] > thresh else 'black')



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





# Predict the values from the validation dataset 

Y_pred = np.zeros((X_val.shape[0], 10)) 



for i in range(num_nets):

    Y_pred = Y_pred + models[i].predict(X_val)

    

Y_pred_classes = np.argmax(Y_pred, axis=1)



# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val, axis=1) 



# Compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)



# Plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes=range(10)) 
# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)

    plt.subplots_adjust(hspace=2)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
results = np.zeros((X_test.shape[0], 10)) 



for i in range(num_nets):

    results = results + models[i].predict(X_test)

    

results = np.argmax(results, axis=1)

results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1, X_test.shape[0]+1), name='ImageId'), results], axis=1)

submission.to_csv('ensemble_submission.csv', index=False)