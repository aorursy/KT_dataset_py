!pip install hyperas
# Basic compuational libaries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D, Flatten, GlobalMaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.models import Sequential

from keras.optimizers import RMSprop, Adam, SGD, Nadam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from keras import regularizers



# Import hyperopt for tunning hyper params

from hyperopt import hp, tpe, fmin

from hyperopt import space_eval



sns.set(style='white', context='notebook', palette='deep')

# Set the random seed

random_seed = 2
def data():

    # Load the data

    train = pd.read_csv("../input/digit-recognizer/train.csv")

    test = pd.read_csv("../input/digit-recognizer/test.csv")

    Y_train = train["label"]

    

    # Drop 'label' column

    X_train = train.drop(labels = ["label"],axis = 1) 

    

    # Normalize the data

    X_train = X_train / 255.0

    test = test / 255.0

    

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

    X_train = X_train.values.reshape(-1,28,28,1)

    test = test.values.reshape(-1,28,28,1)

    

    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

    Y_train = to_categorical(Y_train, num_classes = 10)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)

    

    return X_train, X_val, Y_train, Y_val, test



X_train, X_val, Y_train, Y_val, X_test = data()
g = sns.countplot(np.argmax(Y_train, axis=1))
for i in range(0, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i][:,:,0], cmap=plt.get_cmap('gray'))

    plt.title(np.argmax(Y_train[i]));

    plt.axis('off')

plt.tight_layout()    
epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 64
# With data augmentation to prevent overfitting (accuracy 0.99286)



datagen = ImageDataGenerator(        

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        )



# only apply data augmentation with train data

train_gen = datagen.flow(X_train,Y_train, batch_size=batch_size)



datagen = ImageDataGenerator()

valid_gen = datagen.flow(X_train,Y_train, batch_size=batch_size)

test_gen = datagen.flow(X_test, batch_size=batch_size)
# Set the CNN model 

def train_model(train_generator, valid_generator, params):    

    model = Sequential()



    model.add(Conv2D(filters = params['conv1'], kernel_size = params['kernel_size_1'], padding = 'Same', 

                     activation ='relu', input_shape = (28,28,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = params['conv2'], kernel_size = params['kernel_size_2'], padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size = params['pooling_size_1']))

    model.add(Dropout(params['dropout1']))



    model.add(BatchNormalization())

    model.add(Conv2D(filters = params['conv3'], kernel_size = params['kernel_size_3'], padding = 'Same', 

                     activation ='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = params['conv4'], kernel_size = params['kernel_size_4'], padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size = params['pooling_size_1'], strides=(2,2)))

    model.add(Dropout(params['dropout2']))



    model.add(Flatten())

    model.add(BatchNormalization())

    model.add(Dense(params['dense1'], activation = "relu"))

    model.add(Dropout(params['dropout3']))

    model.add(Dense(10, activation = "softmax"))

    

    if params['opt'] == 'rmsprop':

        opt = RMSprop()

    elif params['opt'] == 'sgd':

        opt = SGD()

    elif params['opt'] == 'nadam':

        opt = Nadam()

    else:

        opt = Adam()

    

    model.compile(loss=params['loss'], optimizer=opt, metrics=['acc'])

        

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, mode='auto', cooldown=2, min_lr=1e-7)

    early = EarlyStopping(monitor='val_loss', patience=3)

    

    callbacks_list = [reduce_lr, early]    

    

    history = model.fit_generator(train_generator,

                                  validation_data=valid_generator,

                                  steps_per_epoch=len(train_generator),

                                  validation_steps=len(valid_generator),

                                  callbacks=callbacks_list, epochs = epochs,

                                  verbose=2)

    

    score, acc = model.evaluate_generator(valid_generator, steps=len(valid_generator), verbose=0)

    

    return acc, model, history
#This is the space of hyperparameters that we will search

space = {

    'opt':hp.choice('opt', ['adam', 'sgd', 'rmsprop']),

    

    'conv1':hp.choice('conv1', [16, 32, 64, 128]),

    'conv2':hp.choice('conv2', [16, 32, 64, 128]),

    'kernel_size_1': hp.choice('kernel_size_1', [3, 5]),

    'kernel_size_2': hp.choice('kernel_size_2', [3, 5]),

    'dropout1': hp.choice('dropout1', [0, 0.25, 0.5]),

    'pooling_size_1': hp.choice('pooling_size_1', [2, 3]),

    

    'conv3':hp.choice('conv3', [32, 64, 128, 256, 512]),

    'conv4':hp.choice('conv4', [32, 64, 128, 256, 512]),

    'kernel_size_3': hp.choice('kernel_size_3', [3, 5]),

    'kernel_size_4': hp.choice('kernel_size_4', [3, 5]),

    'dropout2':hp.choice('dropout2', [0, 0.25, 0.5]),

    'pooling_size_2': hp.choice('pooling_size_2', [2, 3]),

    

    'dense1':hp.choice('dense1', [128, 256, 512, 1024]),

    'dropout3':hp.choice('dropout3', [0, 0.25, 0.5]),

    

    'loss': hp.choice('loss', ['categorical_crossentropy', 'kullback_leibler_divergence']),

}
def optimize(params):

    acc, model, history = train_model(train_gen, valid_gen, params)

    

    return -acc
best = fmin(fn = optimize, space = space, 

            algo = tpe.suggest, max_evals = 2)
best_params = space_eval(space, best)

print('best hyper params:\n', best_params)
acc, model, history = train_model(train_gen, valid_gen, best_params)

print("validation accuracy: {}".format(acc))
optimizers = ['rmsprop', 'sgd', 'adam']

hists = []

params = best_params

for optimizer in optimizers:

    params['opt'] = optimizer

    print("Train with optimizer: {}".format(optimizer))

    _, _, history = train_model(train_gen, valid_gen, params)

    hists.append((optimizer, history))
for name, history in hists:

    plt.plot(history.history['val_acc'], label=name)



plt.legend(loc='best', shadow=True)

plt.tight_layout()
loss_functions = ['categorical_crossentropy', 'kullback_leibler_divergence']

hists = []

params = best_params

for loss_funct in loss_functions:

    params['loss'] = loss_funct

    print("Train with loss function : {}".format(loss_funct))

    _, _, history = train_model(train_gen, valid_gen, params)

    hists.append((loss_funct, history))
for name, history in hists:

    plt.plot(history.history['val_acc'], label=name)



plt.legend(loc='best', shadow=True)

plt.tight_layout()

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

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Display some error results 



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

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1

    fig.tight_layout()

    

# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



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
# predict results

results = model.predict(X_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)