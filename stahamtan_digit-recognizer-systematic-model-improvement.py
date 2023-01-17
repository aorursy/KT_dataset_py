import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from IPython.display import clear_output
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import regularizers, initializers
from keras.optimizers import SGD, adam
from keras.activations import softmax
from keras.metrics import categorical_accuracy
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
def get_training_data():
    """
    This function reads the training data from the Kaggle directory.
    It returns X_train and y_train arrays.
    """
    train = pd.read_csv("../input/train.csv")
    y_train = train.iloc[:, 0].values
    X_train = train.iloc[:, 1:].values
    print("X_train.shape, y_train.shape", X_train.shape, y_train.shape)
    return X_train.astype('float32'), y_train

def get_test_data():
    """
    This function reads the test data from the Kaggle directory.
    It returns X_test array.
    """
    test = pd.read_csv("../input/test.csv")
    X_test = test.values
    print ("X_test.shape", X_test.shape)
    return X_test.astype('float32')
from math import floor
from functools import reduce
classes = np.arange(0, 10)

def plot_splits_distribution(y_dict):
    datasets = ['train', 'val', 'test']
    fig, axes = plt.subplots(figsize=(10,3), nrows=1, ncols=3, sharey=True)
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        ax.set_title(dataset)
        ax.hist(y_dict[dataset], density=True, label=dataset)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Classes')
        ax.set_xticks(classes)

    plt.tight_layout()

def random_sampling(X, y, classes, training_ratio=0.8, val_ratio = 0.1):
    training_size = len(X)
    indecis = np.arange(0, training_size)
    np.random.shuffle(indecis)
    
    training_last_index = floor(training_size*training_ratio)
    val_last_index = floor(training_size*(training_ratio+val_ratio))
    
    training_indecis = indecis[:training_last_index]
    val_indecis = indecis[training_last_index:val_last_index]
    test_indecis = indecis[val_last_index:]
    
    assert ((len(training_indecis)+len(val_indecis)+len(test_indecis))==training_size)
    
    X_train, y_train = X[training_indecis], y[training_indecis].reshape(-1, 1)
    X_val, y_val = X[val_indecis], y[val_indecis].reshape(-1, 1)
    X_test, y_test = X[test_indecis], y[test_indecis].reshape(-1, 1)

    print("X_train.shape, y_train.shape", X_train.shape, y_train.shape)
    print("X_val.shape, y_val.shape", X_val.shape, y_val.shape)
    print("X_test.shape, y_test.shape", X_test.shape, y_test.shape)
    
    X_dict = {'train': X_train,
              'val': X_val,
              'test': X_test}
    y_dict = {'train': y_train,
              'val': y_val,
              'test': y_test}
    
    plot_splits_distribution(y_dict)
    
    return X_dict, y_dict

def stratified_sampling(X, y, classes, training_ratio=0.8, val_ratio = 0.1):
    training_size = len(X)
    training_indecis = []
    val_indecis = []
    test_indecis = []

    for a_class in classes:
        #Get array indecis where y_train value is the same as the class in this iteration
        class_indecis = np.argwhere(y==a_class)[:,0]
        #Shuffle the indecis
        np.random.shuffle(class_indecis)
        #Compute the split points for training, validation and test sets
        class_size = len(class_indecis)
        training_last_index = floor(class_size*training_ratio)
        val_last_index = floor(class_size*(training_ratio + val_ratio))
        #Slice the class_indecis array for each set, then add to the list
        training_indecis.append(class_indecis[:training_last_index])
        val_indecis.append(class_indecis[training_last_index:val_last_index])
        test_indecis.append(class_indecis[val_last_index:])

    #A function to concatenate all arrays in a list
    reduce_func = lambda a,b: np.concatenate([a,b], axis=0)
    training_indecis = reduce(reduce_func, training_indecis)
    val_indecis = reduce(reduce_func, val_indecis)
    test_indecis = reduce(reduce_func, test_indecis)
    
    assert ((len(training_indecis)+len(val_indecis)+len(test_indecis))==training_size)
    
    X_train, y_train = X[training_indecis], y[training_indecis].reshape(-1, 1)
    X_val, y_val = X[val_indecis], y[val_indecis].reshape(-1, 1)
    X_test, y_test = X[test_indecis], y[test_indecis].reshape(-1, 1)

    print("X_train.shape, y_train.shape", X_train.shape, y_train.shape)
    print("X_val.shape, y_val.shape", X_val.shape, y_val.shape)
    print("X_test.shape, y_test.shape", X_test.shape, y_test.shape)
    
    X_dict = {'train': X_train,
              'val': X_val,
              'test': X_test}
    y_dict = {'train': y_train,
              'val': y_val,
              'test': y_test}
    
    plot_splits_distribution(y_dict)
    
    return X_dict, y_dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

class ImageDataScaler (BaseEstimator, TransformerMixin):
    def __init__(self, factor):
        self.factor = factor
    
    def fit(self, *_):
        return self
    
    def transform(self, X, *_):
        return X / self.factor

def preprocess(X_dict, y_dict):
    """
    A function to perform standard scaling on the X_train and OneHotEncoding on the y_train categories.
    It returns:
        X_scaler: A StandardScaler object fit to the X_train data
        X_train_scaled
        y_mlb: A MultiLabelBinarizer object fit to y_train
        y_train_categorical
    """
    X_train = X_dict['train']
    X_val = X_dict['val']
    X_test = X_dict['test']
    y_train = y_dict['train']
    y_val = y_dict['val']
    y_test = y_dict['test']
    
    y_mlb = MultiLabelBinarizer().fit(y_train)
    X_scaler = ImageDataScaler(factor=255).fit(X_train)
    
    X_train = X_scaler.transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    
    y_train = y_mlb.transform(y_train)
    y_val = y_mlb.transform(y_val)
    y_test = y_mlb.transform(y_test)
    
    print("X_train.shape, y_train.shape", X_train.shape, y_train.shape)
    print("X_val.shape, y_val.shape", X_val.shape, y_val.shape)
    print("X_test.shape, y_test.shape", X_test.shape, y_test.shape)
    
    X_dict = {'train': X_train,
              'val': X_val,
              'test': X_test}
    y_dict = {'train': y_train,
              'val': y_val,
              'test': y_test}
    transformers = {'X': X_scaler,
                    'y': y_mlb}
    
    return X_dict, y_dict, transformers

def preprocess_test_data(X_scaler, X_test):
    """
    A function to perform feature scaling on X_test using the X_scaler that is fit to the training data
    It returns X_test_scaled
    """
    X_test_scaled = X_scaler.transform(X_test)
    print ("X_test_scaled.shape", X_test_scaled.shape)
    return X_test_scaled
from mpl_toolkits.axes_grid1 import make_axes_locatable

rows, cols = 28, 28
def show_digit(idx, X, y):
    """
    A function to plot a digit with its labels from X and y arrays for a given index.
    """
    pixels = X[idx, :]
    label = y[idx, 0]
    image = pixels.reshape(rows, cols)
    plt.imshow(image, cmap='gray_r')
    plt.title(label)
    plt.show()

def show_n_images_prediction(indecis, X, X_scaler, model, dataset_name):
    """
    This function plots a series of digits specified in the indecis array. The X is the unscaled digit images array.
    Arguments:
        indecis: list or 1d numpy array of the indecis
        X: Numpy array of digit images of shapes (m, n_x) or (m, row, col, channel) or (m, channel, row, col)
        X_scaler: a StandardScaler object fit to X_train data
        model: a classification model with 'predict' method
    """
    X_scaled = X_scaler.transform(X)
    ncols = 5
    nrows = (len(indecis)-1) // ncols + 1
    fig = plt.figure(figsize=(2*ncols, 2.5*nrows))
    for i, idx in enumerate(indecis):
        pixels = X[idx, :].reshape(1, -1)
        probs = model.predict(np.expand_dims(X_scaled[idx, :], axis=0))
        pred = np.argmax(probs)
        image = pixels.reshape((rows, cols))
        fig.add_subplot(nrows, ncols, i+1)
        plt.imshow(image, cmap='gray_r')
        plt.title("{:.2f}% -> {}".format(probs[0, pred]*100, pred))
    plt.suptitle('{} images from {} with the model prediction'.format(len(indecis), dataset_name), fontsize=16)
    plt.show()

def plot_history(history, metric_name):
    """
    To visualize the loss and the metric variation vs. epochs
    """
    fig = plt.figure(figsize=(10,4))
    fig.add_subplot(1,2,1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    fig.add_subplot(1,2,2)
    plt.plot(history.history[metric_name], label='Training')
    plt.plot(history.history['val_'+metric_name], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    
def plot_confusion_matrix(cms, classes, titles, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')
    
    #print(cm)
    
    fig, axes = plt.subplots(figsize=(16,6), nrows=1, ncols=3)
    for i, cm in enumerate(cms):
        ax = axes[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        """
        fig.colorbar(im, ax=ax)
        ax.set_aspect('auto')
        """
        ax.set_title(titles[i])
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    
    plt.tight_layout()
    

def compute_accuracy(y_true, model, X):
    """
    This function computes and returns the model accuracy using this formula:
        accuracy = correct_predictions / total_predictions
    """
    y_pred = np.argmax(model.predict(X), axis=1).reshape(-1, 1)
    correct_predictions = np.sum(y_pred==y_true)
    total_predictions = len(y_pred)
    accuracy = correct_predictions/total_predictions
    return accuracy, y_pred

def visualize_results(model, history, metric_name,
                      X_dict, y_dict, transformers, X_test, num_images=10):
    """
    Arguments:
        * model -------- A trained keras Nueral Network classification model that can predict y(digit classes) from input X
        * history ------ The training history dictionary
        * metric_name -- the name of metric used during the training
        * X_train ------ 
    This function creates the following visualiztions:
        1. Training Loss (and a given metric) vs. Epochs
        2. A random sample of X_test images with their predictions
        3. Confusion matrix
    """
    
    plot_history(history, metric_name)
    classes = transformers['y'].classes_
    datasets = ['train', 'val', 'test']
    cms = []
    cm_titles = []
    for dataset in datasets:
        accuracy, y_pred = compute_accuracy(y_dict[dataset], model, X_dict[dataset])
        model_accuracy = "({}): Acc: {:.2f}% | Err: {:.2f}%".format(dataset, accuracy*100, (1-accuracy)*100) 
        cm=confusion_matrix(y_dict[dataset], y_pred)
        cms.append(cm)
        cm_titles.append(model_accuracy)
    
    plot_confusion_matrix(cms, classes, titles=cm_titles)
    
    indecis = np.random.randint(0, len(X_test), num_images)
    show_n_images_prediction(indecis, X_test, transformers['X'], model, 'X_test')
# Plotting loss during the training
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()
def baseline_model(X_train_scaled, y_train_categorical, learning_rate=0.0025, decay_rate=1e-6, loss_f='categorical_crossentropy', 
                   metrics=[categorical_accuracy], n_units=10, l2_rate=0, dropout_rate=0):
    m, input_dim = X_train_scaled.shape
    _, output_dim = y_train_categorical.shape
        
    model = Sequential()
    model.add(Dense(units=n_units,activation='relu',kernel_initializer=initializers.he_normal(),
                    kernel_regularizer=regularizers.l2(l2_rate), bias_initializer='zeros',input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=n_units,activation='relu',kernel_initializer=initializers.he_normal(),
                    kernel_regularizer=regularizers.l2(l2_rate), bias_initializer='zeros'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=n_units,activation='relu',kernel_initializer=initializers.he_normal(),
                    kernel_regularizer=regularizers.l2(l2_rate), bias_initializer='zeros'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=n_units,activation='relu',kernel_initializer=initializers.he_normal(),
                    kernel_regularizer=regularizers.l2(l2_rate), bias_initializer='zeros'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=output_dim,activation='softmax',kernel_initializer=initializers.he_normal(),
                    kernel_regularizer=regularizers.l2(l2_rate), bias_initializer='zeros'))
    
    optimizer_f = adam(lr=learning_rate, decay=decay_rate)
    model.compile(optimizer=optimizer_f, loss=loss_f, metrics=metrics)
    
    return model

def LeNet5_model(X_train_scaled, y_train_categorical, learning_rate=0.0025, decay_rate=1e-6, loss_f='categorical_crossentropy', 
                   metrics=[categorical_accuracy], l2_rate=0, dropout_rate=0):
    m, input_dim = X_train_scaled.shape[0], X_train_scaled.shape[1:]
    _, output_dim = y_train_categorical.shape
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim, activation='softmax'))
    
    optimizer_f = adam(lr=learning_rate, decay=decay_rate)
    model.compile(optimizer=optimizer_f, loss=loss_f, metrics=metrics)
    
    return model

def modify_regularizations(model, reg_funcs):
    layers = model.layers
    for i, layer in enumerate(layers):
        layer.kernel_regularizer = reg_funcs[i]
    return model
X_train, y_train = get_training_data()
X_test = get_test_data()
X_dict, y_dict = random_sampling(X_train, y_train, classes, training_ratio=0.8, val_ratio=0.1)
X_scaled, y_categorical, transformers = preprocess(X_dict, y_dict)
model1 = baseline_model(X_scaled['train'], y_categorical['train'])
batch_size = 128
epochs = 100
plot_losses = PlotLosses()
history = model1.fit(X_scaled['train'], y_categorical['train'], batch_size=batch_size,
                  epochs=epochs,  validation_data=(X_scaled['val'], y_categorical['val']), verbose=False,
                  callbacks=[plot_losses])
visualize_results(model1, history, 'categorical_accuracy', X_scaled, y_dict, transformers, X_test)
X_dict, y_dict = stratified_sampling(X_train, y_train, classes, training_ratio=0.8, val_ratio=0.1)
X_scaled, y_categorical, transformers = preprocess(X_dict, y_dict)
model2 = baseline_model(X_scaled['train'], y_categorical['train'])
batch_size = 128
epochs = 100
plot_losses = PlotLosses()
history = model2.fit(X_scaled['train'], y_categorical['train'], batch_size=batch_size,
                  epochs=epochs,  validation_data=(X_scaled['val'], y_categorical['val']), verbose=False,
                  callbacks=[plot_losses])
visualize_results(model2, history, 'categorical_accuracy', X_scaled, y_dict, transformers, X_test)
model3 = baseline_model(X_scaled['train'], y_categorical['train'], n_units=50 )
batch_size = 128
epochs = 100
plot_losses = PlotLosses()
history = model3.fit(X_scaled['train'], y_categorical['train'], batch_size=batch_size,
                  epochs=epochs,  validation_data=(X_scaled['val'], y_categorical['val']), verbose=False,
                  callbacks=[plot_losses])
visualize_results(model3, history, 'categorical_accuracy', X_scaled, y_dict, transformers, X_test)
model4 = baseline_model(X_scaled['train'], y_categorical['train'], n_units=50 ,l2_rate=0.0002)
batch_size = 128
epochs = 100
plot_losses = PlotLosses()
history = model4.fit(X_scaled['train'], y_categorical['train'], batch_size=batch_size,
                  epochs=epochs,  validation_data=(X_scaled['val'], y_categorical['val']), verbose=False,
                  callbacks=[plot_losses])
visualize_results(model4, history, 'categorical_accuracy', X_scaled, y_dict, transformers, X_test)
nrows=ncols=28
nchannels = 1
for key in X_scaled.keys():
    X_scaled[key] = X_scaled[key].reshape(-1, nrows, ncols, nchannels)

X_test = X_test.reshape(-1, nrows, ncols, nchannels)
model5 = LeNet5_model(X_scaled['train'], y_categorical['train'], dropout_rate=0.5)
batch_size = 128
epochs = 50
plot_losses = PlotLosses()
history = model5.fit(X_scaled['train'], y_categorical['train'], batch_size=batch_size,
                  epochs=epochs,  validation_data=(X_scaled['val'], y_categorical['val']), verbose=False,
                  callbacks=[plot_losses])
visualize_results(model5, history, 'categorical_accuracy', X_scaled, y_dict, transformers, X_test)
submission = pd.DataFrame(
    data=np.vstack([np.arange(1, len(X_test)+1), 
                    np.argmax(model5.predict(transformers['X'].transform(X_test)), axis=1)]).T,
    columns=['ImageId', 'Label']
)
show_n_images_prediction(list(range(0, 10)), X_test, transformers['X'], model5, "X_test")
submission.head(10)
submission.to_csv('STahamtan_LeNet5_MNIST_Submission.csv', index=False)