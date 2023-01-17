import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import numpy as np

from keras.utils import plot_model

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from keras.optimizers import RMSprop

from sklearn.metrics import confusion_matrix

import itertools

from keras.layers import LSTM, Dense,CuDNNLSTM, Dropout

from keras.models import Sequential

import seaborn as sns



batch_size = 128

nb_epoch = 10



img_rows, img_cols = 28, 28

nb_classes = 10



nb_lstm_outputs = 30

nb_time_steps = img_rows

dim_input_vector = img_cols

#Traing_Data. we will use both the data set as traing data set

#maleDigits And FemialDigits

train1 = pd.read_csv("../input/maleDigits.csv")

test = pd.read_csv("../input/femaleDigits.csv")

#mergeing maleDigits And FemialDigits 

train = pd.concat((train1,test), ignore_index=True)



#creaing an numpy Array of X_train And y_train

X_train = np.array(train.drop('label', axis=1))

print(X_train.shape)



#creating Dummy Variables for (0-9 digits)

y_train = np.array(pd.get_dummies(train['label']))

#scaling ling The values 

X_train = X_train.astype('float32') / 255.

print(X_train.shape)
#Reshaping The Value 28 by 28

X_train = X_train.reshape(30830,28,28)
#Split the train and the validation set for the fitting X_train, X_val, Y_train, Y_val

random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, 

                                                  random_state=random_seed)
#Defineing The Input Shape

input_shape = (28, 28)
#Buliding RNN Model

model = Sequential()

#If You Don't have gpu support than you can use LSTM() 

model.add(CuDNNLSTM(nb_lstm_outputs, input_shape=input_shape,return_sequences=True))

model.add(CuDNNLSTM(128,return_sequences=True))

model.add(CuDNNLSTM(256))

model.add(Dense(nb_classes, activation='softmax'))



#We Have Use RMSprop As optimizer 

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# Train

history = model.fit(X_train, Y_train, epochs=30, 

                    batch_size=batch_size,

                    validation_data = (X_val,Y_val),

                    shuffle=True, verbose=1,

                    callbacks=[learning_rate_reduction])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

   

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

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

        

        plt.tight_layout()

        plt.ylabel('True label')

        plt.xlabel('Predicted label')





Y_pred = model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_val,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(10))



plt.show()

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

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)

most_important_errors = sorted_dela_errors[-6:]

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)



plt.show()