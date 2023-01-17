%%writefile utils.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Common packages
import numpy as np
import pandas as pd
import warnings

# ML
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

#Keras/Tensorflow
import tensorflow as tf
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

# Image processing
import imageio
import skimage
import skimage.io
import skimage.transform

img_folder='../input/data/imgs/'

categories = {}

def setup_onehot(df):
    categories['subspecies'] = np.unique(df['subspecies'])
    categories['health'] = np.unique(df['health'])
    
def read_data():
    bees=pd.read_csv('../input/data/bees_train.csv', 
                index_col=False,
                dtype={'subspecies':'category', 'health':'category','caste':'category'})
    bees_test_for_evaluation=pd.read_csv('../input/data/bees_test.csv', 
                index_col=False,  
                dtype={'caste':'category'})
    
    setup_onehot(bees)
    
    return bees, bees_test_for_evaluation

def read_img(file, img_folder, img_width, img_height, img_channels):
    """
    Read and resize img, adjust channels. 
    @param file: file name without full path
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.io.imread(img_folder + file)
        img = skimage.transform.resize(img, (img_width, img_height), mode='reflect', )
    return img[:,:,:img_channels]


def split(bees):
    """ 
    Split to train, test and validation. 
    
    @param bees: Total Bees dataset to balance and split
    @return:  train bees, validation bees, test bees
    """
    # Split to train and test before balancing
    train_bees, test_bees = train_test_split(bees, random_state=24)

    # Split train to train and validation datasets
    # Validation for use during learning
    train_bees, val_bees = train_test_split(train_bees, test_size=0.1, random_state=24)

    return(train_bees, val_bees, test_bees)
	
def load_images_and_target(train_bees, val_bees, test_bees, y_field_name, img_width, img_height, img_channels):
    """
    Load images for features, drop other columns
    One hot encode for label, drop other columns
    @return: train images, validation images, test images, train labels, validation labels, test labels
    """
    # Bees already splitted to train, validation and test
    # Load and transform images to have equal width/height/channels. 
    # Use np.stack to get NumPy array for CNN input

    # Train data
    train_X = np.stack(train_bees['file'].apply(lambda x: read_img(x, img_folder, img_width, img_height, img_channels)))
    #train_y = onehot_encoding(train_bees[y_field_name].values)
    train_y  = pd.get_dummies(train_bees[y_field_name], drop_first=False)

    # Validation during training data to calc val_loss metric
    val_X = np.stack(val_bees['file'].apply(lambda x: read_img(x, img_folder, img_width, img_height, img_channels)))
    #val_y = onehot_encoding(val_bees[y_field_name].values)
    val_y = pd.get_dummies(val_bees[y_field_name], drop_first=False)

    # Test data
    test_X = np.stack(test_bees['file'].apply(lambda x: read_img(x, img_folder, img_width, img_height, img_channels)))
    #test_y = onehot_encoding(test_bees[y_field_name].values)
    test_y = pd.get_dummies(test_bees[y_field_name], drop_first=False)

    return (train_X, val_X, test_X, train_y, val_y, test_y)	


def class_weights(y) :
    # Hint: usar
    # http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    #return compute_class_weight("balanced", np.unique(y), y)
    #return np.ones(np.unique(y).shape[0])
    return dict(enumerate(compute_class_weight("balanced", np.unique(y), y)))

def train(      model,
                train_X,
                train_y, 
                batch_size,
                epochs,
                validation_data_X,
				validation_data_y,
                steps_per_epoch,
                rotation_range,  # randomly rotate images in the range (degrees, 0 to rotation_range)
                zoom_range, # Randomly zoom image 
                width_shift_range,  # randomly shift images horizontally (fraction of total width)
                height_shift_range,  # randomly shift images vertically (fraction of total height)
                horizontal_flip,  # randomly flip images
                vertical_flip,
				patience,
				class_weights):
				
	generator = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to rotation_range)
				zoom_range = zoom_range, # Randomly zoom image 
				width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=height_shift_range,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=horizontal_flip,  # randomly flip images
				vertical_flip=vertical_flip)
				
				
	generator.fit(train_X)
	#Train
	##Callbacks
	earlystopper = EarlyStopping(monitor='loss', patience=patience, verbose=1,restore_best_weights=True)
    
	training = model.fit_generator(generator.flow(train_X,train_y, batch_size)
                        ,epochs=epochs
                        ,validation_data=[validation_data_X, validation_data_y]
                        ,steps_per_epoch=steps_per_epoch
                        ,callbacks=[earlystopper]
                        ,class_weight = class_weights)
								
	return training, model

def plot_images(data, attribute, samples) :
    if len(samples) < 2 or len(samples) > 5 : 
        raise ValueError('len(samples) must be in [2, 5]') 
        
    _, ax = plt.subplots(nrows = 1, ncols = len(samples), figsize = (20, 5))
    for i, img_idx in enumerate(samples) :
        attrname = data[attribute].iloc[img_idx]
        filename = '../input/data/imgs/' + data['file'].iloc[img_idx]
        img = imageio.imread(filename)
        ax[i].imshow(img)
        ax[i].set_title(attrname, fontsize = 16)
    plt.tight_layout()
    plt.show()

def eval_model(training, model, test_X, test_y, field_name):
    """
    Model evaluation: plots, classification report
    @param training: model training history
    @param model: trained model
    @param test_X: features 
    @param test_y: labels
    @param field_name: label name to display on plots
    """
    ## Trained model analysis and evaluation
    f, ax = plt.subplots(2,1, figsize=(5,5))
    ax[0].plot(training.history['loss'], label="Loss")
    ax[0].plot(training.history['val_loss'], label="Validation loss")
    ax[0].set_title('%s: loss' % field_name)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Accuracy
    ax[1].plot(training.history['acc'], label="Accuracy")
    ax[1].plot(training.history['val_acc'], label="Validation accuracy")
    ax[1].set_title('%s: accuracy' % field_name)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    
    # Accuracy by subspecies
    test_pred = model.predict(test_X)
    
    acc_by_subspecies = np.logical_and((test_pred > 0.5), test_y).sum()/test_y.sum()
    acc_by_subspecies.plot(kind='bar', title='Accuracy by %s' % field_name)
    plt.ylabel('Accuracy')
    plt.show()

    # Print metrics
    print("Classification report")
    test_pred = np.argmax(test_pred, axis=1)
    test_truth = np.argmax(test_y.values, axis=1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(metrics.classification_report(test_truth, test_pred, target_names=test_y.columns))

    # Loss function and accuracy
    test_res = model.evaluate(test_X, test_y.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])

def load_test(img_width, img_height, img_channels):
	X_test_partition=pd.read_csv('../input/data/bees_test.csv', 
					index_col=False,  
					dtype={'caste':'category'})
		
	test_images = np.stack(X_test_partition['file'].apply(lambda x: read_img(x, img_folder, img_width, img_height, img_channels)))
	
	return X_test_partition, test_images

def predict(model, data):	
	pred = np.argmax(model.predict(data), axis = 1) 
	pred = pred.reshape(-1,1)
	return pred
	
def load_test_and_generate_prediction_file(model, class_weights, class_name, img_width, img_height, img_channels):
	
	X_test_partition, test_images = load_test(img_width, img_height, img_channels)
	pred = predict(model, test_images)
	    
	test_ids = X_test_partition['id']	
	test_ids = np.array(test_ids).reshape(-1,1)

	output = np.stack((test_ids, pred), axis=-1)
	output = output.reshape([-1, 2])

	df = pd.DataFrame(output)
	df.columns = ['id','expected']
    
	df['expected'] = df['expected'].map(pd.Series(categories[class_name]))    
	df.to_csv("test_" + class_name + ".csv",index = False,index_label = False)	
	return df
import pandas as pd
import numpy as np
#import sys
#import os
#import random
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Lambda, GlobalAveragePooling2D, Input, Activation, BatchNormalization, GlobalMaxPooling2D
import tensorflow as tf

import utils
from keras.applications import vgg16, inception_v3, resnet50, mobilenet, densenet
from keras.preprocessing import image
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2, l1_l2
np.random.seed(28)
tf.set_random_seed(28)
img_width = 100
img_height = 100
img_channels = 3
bees, bees_test_for_evaluation = utils.read_data()
np.unique(['Italian']).reshape(-1,1)
np.unique(bees['subspecies'])
bees.head()
bees_test_for_evaluation.head()
# bees es el archivo a tomar
bees['subspecies'].head()
# Graficos embebidos.
%matplotlib inline
# parametros esteticos de seaborn
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (5, 4)})
bees.groupby('subspecies').size().plot(kind='bar', title='Cantidad de abejas x subespecie') 
plt.xlabel("Tipos de Subespecies")
plt.ylabel("Cantidad")
plt.show()
# Gráfico de tarta de subespecies de abejas
bees['subspecies'].value_counts().plot(kind='pie', autopct='%.2f', 
                                            figsize=(6, 6),
                                            title='Abejas x subespecie')
plt.show()
# Tabla de contingencia health según subspecies
pd.crosstab(index=bees['health'],
            columns=bees['subspecies'], margins=True)
# tabla de contingencia en porcentajes relativos total
pd.crosstab(index=bees['health'], columns=bees['subspecies'],
            margins=True).apply(lambda r: r/len(bees) *100,
                                axis=1)
# Gráfico de barras de la cantidad de abejas por estado de salud segun subespecie
pd.crosstab(index=bees['subspecies'],
            columns=bees['health']).plot(kind='bar', title = "Estado de Salud por cantidad de abejas segun subespecie", stacked=True)

# Gráfico de barras del porcentaje de abejas por estado de salud segun subespecie
pd.crosstab(index=bees['subspecies'],
            columns=bees['health']).apply(lambda r: r/r.sum() *100,
                          axis=1).plot(kind='bar', stacked=True, title = "% de enfermedades en subespecies", legend=False)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Health")
plt.show()

# plot_images(data, attribute, samples)
utils.plot_images(bees,'subspecies',[1,3,6,7])
# plot_images(data, attribute, samples)
utils.plot_images(bees,'subspecies',[15,19,42])
# plot_images(data, attribute, samples)
utils.plot_images(bees,'health',[2,6,8])
# plot_images(data, attribute, samples)
utils.plot_images(bees,'health',[10,19,43])
train_bees, val_bees, test_bees = utils.split(bees)
optimizer = Adam(lr=1e-5)
loss = 'categorical_crossentropy'
from enum import Enum
class Convolutional_Base(Enum):
    SIMPLE = 1
    COMPLEX = 2
    MORE_COMPLEX = 3
class Classifier(Enum):
    GLOBAL_MAX_POOLING = 1
    GLOBAL_AVERAGE_POOLING = 2
    FLATTEN = 3
def add_simple_convolutional_base(model, batch_normalization=False):
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (img_height, img_width, img_channels), name="conv_layer_1"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu', name="conv_layer_2"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu', name="conv_layer_3"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
def add_complex_convolutional_base(model, batch_normalization = False):
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding='Same', activation ='relu', input_shape = (img_height, img_width, img_channels), name="conv_layer_1.1"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding='Same', activation ='relu', name="conv_layer_1.2"))    
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_2.1"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_2.2")) 
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_3.1"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_3.2"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    
    # agregado...
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_4.1"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_4.2"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
def add_more_complex_convolutional_base(model, batch_normalization = False):
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding='Same', activation ='relu', input_shape = (img_height, img_width, img_channels), name="conv_layer_1.1"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding='Same', activation ='relu', name="conv_layer_1.2"))    
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding='Same', activation ='relu', name="conv_layer_1.3"))    
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_2.1"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_2.2")) 
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_2.3")) 
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_3.1"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_3.2"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding='Same', activation='relu', name="conv_layer_3.3"))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
def add_classifier_with_global_pooling(model, global_pooling, batch_normalization = False, dropout = False, kernel_regularizer = None):
    model.add(global_pooling)
    model.add(Dense(1024, activation='relu', kernel_regularizer = kernel_regularizer, name='dense1'))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Dropout(0.5)) if (dropout) else False
    model.add(Dense(1024, activation='relu', kernel_regularizer = kernel_regularizer, name='dense2'))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Dropout(0.5)) if (dropout) else False
    model.add(Dense(train_y.columns.size, activation='softmax', name='predictions'))
def add_classifier_with_flatten(model, batch_normalization = False, dropout = False, kernel_regularizer = None):
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu", kernel_regularizer = kernel_regularizer, name='dense1'))
    model.add(BatchNormalization()) if (batch_normalization) else False
    model.add(Dropout(0.5)) if (dropout) else False
    model.add(Dense(train_y.columns.size, activation = "softmax", name='predictions'))
def create_model(base_model, classifier, batch_normalization, dropout, kernel_regularizer, summary):
    model = Sequential()
    if (base_model == Convolutional_Base.MORE_COMPLEX):
        add_more_complex_convolutional_base(model, batch_normalization = True)
    if (base_model == Convolutional_Base.COMPLEX):
        add_complex_convolutional_base(model, batch_normalization = True)
    if (base_model == Convolutional_Base.SIMPLE):
        add_simple_convolutional_base(model, batch_normalization = True)
    
    if (classifier == Classifier.GLOBAL_MAX_POOLING):
        add_classifier_with_global_pooling(model, global_pooling = GlobalMaxPooling2D(), batch_normalization = batch_normalization, dropout = dropout, kernel_regularizer = kernel_regularizer)
    if (classifier == Classifier.GLOBAL_AVERAGE_POOLING):
        add_classifier_with_global_pooling(model, global_pooling = GlobalAveragePooling2D(), batch_normalization = batch_normalization, dropout = dropout, kernel_regularizer = kernel_regularizer)
    if (classifier == Classifier.FLATTEN):
        add_classifier_with_flatten(model, batch_normalization, dropout, kernel_regularizer)
    
    model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy']) 
    model.summary() if (summary) else False
    
    return model
def create_simple_model_with_flatten(summary = False):
    return create_model(base_model = Convolutional_Base.SIMPLE, classifier = Classifier.FLATTEN, batch_normalization = False, dropout = False, kernel_regularizer = None, summary = False)
def create_simple_model_with_average_pooling(summary = False):
    return create_model(base_model = Convolutional_Base.SIMPLE, classifier = Classifier.GLOBAL_AVERAGE_POOLING, batch_normalization = False, dropout = False, kernel_regularizer = None, summary = False)
def create_simple_model_with_max_pooling(summary = False):
    return create_model(base_model = Convolutional_Base.SIMPLE, classifier = Classifier.GLOBAL_MAX_POOLING, batch_normalization = False, dropout = False, kernel_regularizer = None, summary = False)
def create_complex_model_with_max_pooling():
    return create_model(base_model = Convolutional_Base.COMPLEX, classifier = Classifier.GLOBAL_MAX_POOLING, batch_normalization = False, dropout = False, kernel_regularizer = None, summary = False)
def create_complex_model_with_average_pooling():
    return create_model(base_model = Convolutional_Base.COMPLEX, classifier = Classifier.GLOBAL_AVERAGE_POOLING, batch_normalization = False, dropout = False, kernel_regularizer = None, summary = False)
def create_complex_model_with_flatten():
    return create_model(base_model = Convolutional_Base.COMPLEX, classifier = Classifier.FLATTEN, batch_normalization = False, dropout = False, kernel_regularizer = None, summary = False)
def create_best_simple_model_with_batch_normalization():
    return create_model(base_model = Convolutional_Base.SIMPLE, classifier = Classifier.FLATTEN, batch_normalization = True, dropout = False, kernel_regularizer = None, summary = False)
def create_best_complex_model_with_batch_normalization():
    return create_model(base_model = Convolutional_Base.COMPLEX, classifier = Classifier.FLATTEN, batch_normalization = True, dropout = False, kernel_regularizer = None, summary = False)
def create_model_with_dropout():
    return create_model(base_model = Convolutional_Base.COMPLEX, classifier = Classifier.FLATTEN, batch_normalization = True, dropout = True, kernel_regularizer = None, summary = False)
def create_model_with_l1_regularization():
    return create_model(base_model = Convolutional_Base.COMPLEX, classifier = Classifier.FLATTEN, batch_normalization = True, dropout = False, kernel_regularizer = l1(0.01), summary = False)
def create_model_with_l2_regularization():
    return create_model(base_model = Convolutional_Base.COMPLEX, classifier = Classifier.FLATTEN, batch_normalization = True, dropout = False, kernel_regularizer = l2(0.01), summary = False)
def create_simple_model_fine_tuned():
    return create_model(base_model = Convolutional_Base.SIMPLE, classifier = Classifier.GLOBAL_MAX_POOLING, batch_normalization = True, dropout = True, kernel_regularizer = l2(0.01), summary = False)
def create_complex_model_fine_tuned():
    return create_model(base_model = Convolutional_Base.COMPLEX, classifier = Classifier.GLOBAL_MAX_POOLING, batch_normalization = True, dropout = True, kernel_regularizer =l2(0.01), summary = False)
def create_classifier_global_max_pooling(base_model, kernel_regularizer = l2(0.01)):
    model = Sequential()
    model.add(base_model)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(1024, kernel_regularizer = kernel_regularizer, name = "dense1", activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, kernel_regularizer = kernel_regularizer, name = "dense2", activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(train_y.columns.size, activation='softmax', name='predictions'))
        
    model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])  
    model.summary()
    
    return model
def create_classifier_fully_connected(base_model):
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(train_y.columns.size, activation='softmax', name='predictions'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])  
    model.summary()
    
    return model
def add_preprocessing(pretrained_model, preprocess_input):
    inputs = Input((img_height, img_width, img_channels))
    x = Lambda(preprocess_input, name='preprocessing')(inputs)
    outputs = pretrained_model(x)
    base_model = Model(inputs, outputs)
     
    return base_model
def create_inceptionV3_model():
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape = (img_height, img_width, img_channels))
    for layer in base_model.layers:
        layer.trainable = False

    return base_model
def create_vgg16_model():
    base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = (img_height, img_width, img_channels))
    for layer in base_model.layers:
        layer.trainable = True
     
    return base_model
def create_resnet50_model():
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape = (img_height, img_width, img_channels))
    for layer in base_model.layers:
        layer.trainable = False

    return base_model
def create_densenet201_model():
    base_model = densenet.DenseNet201(weights='imagenet', include_top=False, input_shape = (img_height, img_width, img_channels))
    for layer in base_model.layers:
        layer.trainable = False
 
    return base_model
class VisualizeImageMaximizeFmap(object):
    def __init__(self,pic_shape):
        '''
        pic_shape : a dimention of a single picture e.g., (96,96,1)
        '''
        self.pic_shape = pic_shape
        
    def find_n_feature_map(self,layer_name,max_nfmap):
        '''
        shows the number of feature maps for this layer
        only works if the layer is CNN
        '''
        n_fmap = None
        for layer in model.layers:
            if layer.name == layer_name:
                weights = layer.get_weights()
                n_fmap=weights[1].shape[0]
        if n_fmap is None:
            print(layer_name + " is not one of the layer names..")
            n_fmap = 1
        n_fmap = np.min([max_nfmap,n_fmap])
        return(int(n_fmap))

    def find_image_maximizing_activation(self,iterate,input_img_data,
                                         picorig=False,
                                         n_iter = 30):
        '''
        The input image is scaled to range between 0 and 1
        picorig  : True  if the picture image for input is original scale
                         ranging between 0 and 225
                   False if the picture image for input is ranging [0,1]
        '''
            
        input_img_data = np.random.random((1, 
                                           self.pic_shape[0],
                                           self.pic_shape[1],
                                           self.pic_shape[2]))
        if picorig:
            ## if the original picture is unscaled and ranging between (0,225),
            ## then the image values are centered around 123 with STD=25
            input_img_data = input_img_data*25 + 123 
        ## I played with this step value but the final image looks to be robust
        step = 500 

        
        
        # gradient ascent
        loss_values = []
        for i in range(n_iter):
            loss_value, grads_value = iterate([input_img_data, 0])
            input_img_data += grads_value * step
            loss_values.append(loss_value) 
        return(input_img_data,loss_values)

    def create_iterate(self,input_img, layer_output,filter_index):
        '''
        layer_output[:,:,:,0] is (Nsample, 94, 94) tensor contains:
        W0^T [f(image)]_{i,j}], i = 1,..., 94, j = 1,..., 94
        
        layer_output[:,:,:,1] contains:
        W1^T [f(image)]_{i,j}], i = 1,..., 94, j = 1,..., 94
        
        W0 and W1 are different kernel!
        '''
        ## loss is a scalar 
        if len(layer_output.shape) == 4:
            ## conv layer 
            loss = K.mean(layer_output[:,  :, :, filter_index])
        elif len(layer_output.shape) ==2:
            ## fully connected layer
            loss = K.mean(layer_output[:, filter_index])
         
        # calculate the gradient of the loss evaluated at the provided image
        grads = K.gradients(loss, input_img)[0]
        # normalize the gradients
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # iterate is a function taking (input_img, scalar) and output [loss_value, gradient_value]
        iterate = K.function([input_img, K.learning_phase()], [loss, grads])
        return(iterate)

    def deprocess_image(self,x):
        # standardize to have a mean 0 and std  0.1 
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # Shift x to have a mean 0.5 and std 0.1
        # This means 95% of the x should be in between 0 and 1
        # if x is normal
        x += 0.5
        x = np.clip(x, 0, 1)

        # resclar the values to range between 0 and 255
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')

        return x

    def find_images(self,input_img,layer_names,layer_dict, max_nfmap,
                    picorig=True,n_iter=30):
        '''
        Input :

        input_img   : the alias of the input layer from the deep learning model
        layer_names : list containing the name of the layers whose feature maps to be used
        layer_dict  : symbolic outputs of each "key" layer (we gave them unique names).
        max_nfmap   : the maximum number of feature map to be used for each layer.
        pic_shape   : For example pic_shape = (96,96,1)

        Output : 
        dictionary 

        key = layer name 
        value = a list containing the tuple of (images, list of loss_values) that maximize each feature map
        '''
        argimage = {}
        ## Look for the image for each feature map of each layer one by one
        for layer_name in layer_names: ## the layer to visualize
            n_fmap = self.find_n_feature_map(layer_name,max_nfmap)
            layer_output = layer_dict[layer_name].output
            result = self.find_images_for_layer(input_img,
                                                layer_output,
                                                range(n_fmap),
                                                picorig=picorig,
                                                n_iter=n_iter)

            argimage[layer_name] = result
        return(argimage)

    def find_images_for_layer(self,input_img,layer_output,indecies,
                              picorig=False,n_iter=30):
        '''
        indecies : list containing index of 
                      --> filtermaps of CNN or 
                      --> nodes of fully-connected layer
        Output

        a list containing the tuple of (images, list of loss_values) 
        that maximize each feature map


        '''
        result_temp = []
        for filter_index in indecies: # filtermap to visualize
                iterate = self.create_iterate(input_img, layer_output,filter_index)
                input_img_data, loss_values = self.find_image_maximizing_activation(
                    iterate,input_img,
                    picorig=picorig,
                    n_iter=n_iter)
                result_temp.append((input_img_data,loss_values))
        return(result_temp)

    def plot_images_wrapper(self,argimage,n_row = 8, scale = 1):
        '''
        scale : scale up or down the plot size
        '''
        pic_shape = self.pic_shape
        if pic_shape[2] == 1:
            pic_shape = self.pic_shape[:2]
        layer_names = np.sort(list(argimage))


        for layer_name in layer_names:
            n_fmap = len(argimage[layer_name])
            n_col = np.ceil(n_fmap/float(n_row))
            fig = plt.figure(figsize=(n_col*scale,
                                      n_row*scale))
            fig.subplots_adjust(hspace=0.001,wspace=0.001)
            plt.title(layer_name + " n_featuremap=" + str(n_fmap))
            count = 1
            for value in argimage[layer_name]:
                input_img_data = value[0][0]
                img = self.deprocess_image(input_img_data)
                ax = fig.add_subplot(n_row,n_col,count,
                                    xticks=[],yticks=[])
                ax.imshow(img.reshape(*pic_shape),cmap="gray")
                count += 1
            plt.show()
def visualize_filter_model(model, layer_names = ['conv_layer_1.1', 'conv_layer_1.2', 'conv_layer_2.1', 'conv_layer_2.2', 'conv_layer_3.1', 'conv_layer_3.2', 'dense1', 'dense2', 'predictions']):
    input_img = model.layers[0].input
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    visualizer = VisualizeImageMaximizeFmap(pic_shape = (img_height, img_width, img_channels))
    max_nfmap = 3
    argimage = visualizer.find_images(input_img,
                                  layer_names,
                                  layer_dict, 
                                  max_nfmap)
    visualizer.plot_images_wrapper(argimage,n_row = 1, scale = 3)
rotation_range = 250      # rotación aleatoria en grados entre 0 a rotation_range
zoom_range = 0.5         # zoom aleatorio
width_shift_range = 0.5  # desplazamiento horizontal aleatorio (fracción del total)
height_shift_range = 0.5 # desplazamiento vertical aleatorio (fracción del total)
horizontal_flip = True   # transposición horizontal
vertical_flip = True     # transposición horizontal
batch_size = 100
epochs = 600
steps_per_epoch = 150
patience = 7
def training_model(model):
    return utils.train(model,
                train_X,
                train_y, 
                batch_size = batch_size,
                epochs = epochs,
                validation_data_X = val_X, 
                validation_data_y = val_y,
                steps_per_epoch = steps_per_epoch,
                rotation_range = rotation_range,
                zoom_range = zoom_range, 
                width_shift_range = width_shift_range,
                height_shift_range = height_shift_range,
                horizontal_flip = horizontal_flip,  
                vertical_flip = vertical_flip,
                patience = patience,
                class_weights = class_weights
            )
class_weights = utils.class_weights(bees['subspecies'])
train_X, val_X, test_X, train_y, val_y, test_y = utils.load_images_and_target(train_bees, 
                                                                              val_bees, 
                                                                              test_bees,
                                                                              'subspecies',
                                                                              img_width, 
                                                                              img_height, 
                                                                              img_channels)
def eval_model(training, model):
    utils.eval_model(training, model, test_X, test_y, 'subspecies')
def prediction_file(model):
    df = utils.load_test_and_generate_prediction_file(model, class_weights, 'subspecies', img_width, img_height, img_channels)
    df.head(5)
train_base_model = False
train_model_with_batch_normalization = False
train_model_with_dropout = False
train_model_with_l1_regularization = False
train_model_with_l2_regularization = False
train_model_fine_tuned = True
if (train_base_model):
    training, model = training_model(create_complex_model_with_average_pooling())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_complex_model_with_max_pooling())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_complex_model_with_flatten())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_simple_model_with_average_pooling())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_simple_model_with_max_pooling())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_simple_model_with_flatten())
    eval_model(training, model)
if (train_model_with_batch_normalization):
    training, model = training_model(create_best_complex_model_with_batch_normalization())
    eval_model(training, model)
if (train_model_with_batch_normalization):
    training, model = training_model(create_best_simple_model_with_batch_normalization())
    eval_model(training, model)
if (train_model_with_dropout):
    training, model = training_model(create_model_with_dropout())
    eval_model(training, model)
if (train_model_with_l1_regularization):
    training, model = training_model(create_model_with_l1_regularization())
    eval_model(training, model)
if (train_model_with_l2_regularization):
    training, model = training_model(create_model_with_l2_regularization())
    eval_model(training, model)
if (train_model_fine_tuned):
    training, model = training_model(create_complex_model_fine_tuned())
if (train_model_fine_tuned):
    eval_model(training, model)
    visualize_filter_model(model)
    prediction_file(model)
train_inception_v3 = False
train_vgg16 = False
train_resnet50 = False
train_densenet201 = False
if (train_inception_v3):
    training, model = training_model(create_classifier_global_max_pooling(create_inceptionV3_model()))
if (train_inception_v3):
    eval_model(training, model)
if (train_vgg16):
    training, model = training_model(create_classifier_global_max_pooling(create_vgg16_model()))
if (train_vgg16):
    eval_model(training, model)
    prediction_file(model)
if (train_resnet50):
    training, model = training_model(create_classifier_global_max_pooling(create_resnet50_model()))
if (train_resnet50):
    eval_model(training, model)
if (train_densenet201):
    training, model = training_model(create_classifier_global_max_pooling(create_densenet201_model()))
if (train_densenet201):
    eval_model(training, model)
class_weights = utils.class_weights(bees['health'])
train_X, val_X, test_X, train_y, val_y, test_y = utils.load_images_and_target(train_bees, 
                                                                              val_bees, 
                                                                              test_bees,
                                                                              'health',
                                                                              img_width, 
                                                                              img_height, 
                                                                              img_channels)
def eval_model(training, model):
    utils.eval_model(training, model, test_X, test_y, 'health')
def prediction_file(model):
    df = utils.load_test_and_generate_prediction_file(model, class_weights, 'health', img_width, img_height, img_channels)
    df.head(5)
train_base_model = False
train_model_with_batch_normalization = False
train_model_with_dropout = False
train_model_with_l1_regularization = False
train_model_with_l2_regularization = False
train_model_fine_tuned = False
if (train_base_model):
    training, model = training_model(create_complex_model_with_average_pooling())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_complex_model_with_max_pooling())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_complex_model_with_flatten())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_simple_model_with_average_pooling())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_simple_model_with_max_pooling())
    eval_model(training, model)
if (train_base_model):
    training, model = training_model(create_simple_model_with_flatten())
    eval_model(training, model)
if (train_model_with_batch_normalization):
    training, model = training_model(create_best_simple_model_with_batch_normalization())
    eval_model(training, model)
if (train_model_with_batch_normalization):
    training, model = training_model(create_best_complex_model_with_batch_normalization())
    eval_model(training, model)
if (train_model_with_dropout):
    training, model = training_model(create_model_with_dropout())
    eval_model(training, model)
if (train_model_with_l1_regularization):
    training, model = training_model(create_model_with_l1_regularization())
    eval_model(training, model)
if (train_model_with_l2_regularization):
    training, model = training_model(create_model_with_l2_regularization())
    eval_model(training, model)
if (train_model_fine_tuned):
    training, model = training_model(create_complex_model_fine_tuned())
if (train_model_fine_tuned):
    eval_model(training, model)
    visualize_filter_model(model)
    prediction_file(model)
train_inception_v3 = False
train_vgg16 = False
train_resnet50 = False
train_densenet201 = False
if (train_inception_v3):
    training, model = training_model(create_classifier_global_max_pooling(create_inceptionV3_model()))
if (train_inception_v3):
    eval_model(training, model)
if (train_vgg16):
    training, model = training_model(create_classifier_global_max_pooling(create_vgg16_model()))
if (train_vgg16):
    eval_model(training, model)
    prediction_file(model)
if (train_resnet50):
    training, model = training_model(create_classifier_global_max_pooling(create_resnet50_model()))
if (train_resnet50):
    eval_model(training, model)
if (train_densenet201):
    training, model = training_model(create_classifier_global_max_pooling(create_densenet201_model()))
if (train_densenet201):
    eval_model(training, model)