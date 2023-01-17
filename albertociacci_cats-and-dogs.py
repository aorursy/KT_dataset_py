import os
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models, layers, regularizers, metrics, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from scipy import stats
import shutil
import random
def standardize_0_1(ts):
    sts = (ts - np.min(ts))/(np.max(ts) - np.min(ts))
    return sts

def standardize_m_s(ts):
    sts = (ts - np.mean(ts))/(np.std(ts))
    return sts

def dataset_basic_info(generator, name):
        print('The ' + name + ' data set includes ' + str(generator.samples) + ' samples.')
        print('The ' + name + ' image shapes is ' + str(generator.image_shape))
        keys = [el for el in generator.class_indices.keys()]
        print('The ' + name + ' data set includes the following labels: ')
        print(keys)
        labels     = generator.labels
        cat_labels = []
        for i in range(len(labels)):
            for j in range(len(keys)):
                if (labels[i] == j):
                    cat_labels.append(keys[j])
                    break
        occurrences = []
        for key in keys:
            counter = 0
            for i in range(len(cat_labels)):
                if cat_labels[i] == key:
                    counter += 1
            occurrences.append(counter)
        print(name + ' data set labels frequencies:')
        weights = {}
        for i in range(len(keys)):
            print(keys[i] + ': ' + str(occurrences[i]) + ' (absolute), ' + str(round(occurrences[i]/float(generator.samples), 3)) + ' (relative).' )
            weights[i] = generator.samples/np.array(occurrences[i])*(1.0/float(len(keys)))
        
        return weights

def one_hot_encoding(data, exclusions):
    
    columns      = data.columns
    n_samples    = data.shape[0]
    one_hot_data = pd.DataFrame()
    for col in columns:
        if col not in exclusions:
                unique_vals = np.unique(data.loc[:,col])
                for val in unique_vals:
                    new_col_name = col + "_" + str(val)
                    zeros_array  = np.zeros(n_samples)
                    zeros_array[(data[col] == val)] = 1
                    one_hot_data[new_col_name] = zeros_array
    for col in exclusions:
        one_hot_data[col] = data[col]
    return one_hot_data
    
def build_model_ann(regression_problem, hidden_layers_neurons, hidden_activation_function, L1_coeffs, L2_coeffs, hidden_layers_dropout, final_layer_neurons, final_activation_function, shape, model_optimizer, loss_function, metrics):
    
    model = models.Sequential()
    
    for i in range(len(hidden_activation_function)):
        
        if (i == 0):
            model.add(layers.Dense(hidden_layers_neurons[i], 
                                   kernel_regularizer = regularizers.l1_l2(l1 = L1_coeffs[i], l2 = L2_coeffs[i]),                             
                                   activation=hidden_activation_function[i], 
                                   input_shape=(shape,)))
        else:
            model.add(layers.Dense(hidden_layers_neurons[i], 
                                   kernel_regularizer = regularizers.l1_l2(l1 = L1_coeffs[i], l2 =  L2_coeffs[i]),  
                                   activation=hidden_activation_function[i]))
        if (hidden_layers_dropout[i] > 0.0):
            model.add(layers.Dropout(hidden_layers_dropout[i]))
    if regression_problem:
            model.add(layers.Dense(final_layer_neurons))
    else:
            model.add(layers.Dense(final_layer_neurons,activation = final_activation_function))
            
    model.compile(optimizer = model_optimizer, loss = loss_function, metrics = metrics)
    
    return model

def build_model_cnn(regression_problem, conv_filters, conv_filter_shape, conv_activation_function, conv_padding, conv_pooling_type, conv_pooling_shape, hidden_layers_neurons, hidden_activation_function, L1_coeffs, L2_coeffs, hidden_layers_dropout, final_layer_neurons, final_activation_function, shape, model_optimizer, loss_function, metrics):
    
    model = models.Sequential()
    
    for i in range(len(conv_activation_function)):
        
        if (i == 0):
            model.add(layers.Conv2D(conv_filters[i],
                                    (conv_filter_shape[i][0],conv_filter_shape[i][1]), 
                                    activation = conv_activation_function[i], 
                                    padding    = conv_padding[i],
                                    input_shape = (shape[0],shape[1],shape[2])))             
        else:
            model.add(layers.Conv2D(conv_filters[i],
                                    (conv_filter_shape[i][0],conv_filter_shape[i][1]), 
                                    activation = conv_activation_function[i],
                                    padding    = conv_padding[i]))
        
        if (conv_pooling_type[i] == 'max'):
            model.add(layers.MaxPooling2D((conv_pooling_shape[i][0],conv_pooling_shape[i][1])))
        elif (conv_pooling_type[i] == 'avg'):
            model.add(layers.AveragePooling2D((conv_pooling_shape[i][0],conv_pooling_shape[i][1])))
        else:
            'no pooling'
            
    model.add(layers.Flatten())
    
    for i in range(len(hidden_activation_function)):

        model.add(layers.Dense(hidden_layers_neurons[i], 
                               kernel_regularizer = regularizers.l1_l2(l1 = L1_coeffs[i], l2 =  L2_coeffs[i]),  
                               activation=hidden_activation_function[i]))
        if (hidden_layers_dropout[i] > 0.0):
            model.add(layers.Dropout(hidden_layers_dropout[i]))
    if regression_problem:
            model.add(layers.Dense(final_layer_neurons))
    else:
            model.add(layers.Dense(final_layer_neurons,activation = final_activation_function))
            
    model.compile(optimizer = model_optimizer, loss = loss_function, metrics = metrics)
    
    model.summary()
    
    return model

def build_model_pretrained_cnn(pre_trained_model, regression_problem, hidden_layers_neurons, hidden_activation_function, L1_coeffs, L2_coeffs, hidden_layers_dropout, final_layer_neurons, final_activation_function, model_optimizer, loss_function, metrics):
    
    model = models.Sequential()
    model.add(pre_trained_model)
    model.add(layers.Flatten())
    
    for i in range(len(hidden_activation_function)):

        model.add(layers.Dense(hidden_layers_neurons[i], 
                               kernel_regularizer = regularizers.l1_l2(l1 = L1_coeffs[i], l2 =  L2_coeffs[i]),  
                               activation=hidden_activation_function[i]))
        if (hidden_layers_dropout[i] > 0.0):
            model.add(layers.Dropout(hidden_layers_dropout[i]))
    if regression_problem:
            model.add(layers.Dense(final_layer_neurons))
    else:
            model.add(layers.Dense(final_layer_neurons,activation = final_activation_function))
            
    model.compile(optimizer = model_optimizer, loss = loss_function, metrics = metrics)
    
    model.summary()
    
    return model

def display_input_images(generator, max_n_figures, batch_size, grid_size, fig_size):
    
    fig_counter = 0
    for image_batch, label_batch in generator: 
        plt.figure(figsize=(fig_size[0],fig_size[1]))
        for j in range(batch_size):
            ax   = plt.subplot(grid_size[0], grid_size[1], j + 1)
            plt.imshow(image_batch[j])
            if (label_batch[j] == 1):
                    plt.title("Dog")
            else:
                    plt.title("Cat")
            plt.axis("off")
        plt.show()
        fig_counter += 1
        if (fig_counter == max_n_figures): break

def analyze_performances(hst, epochs):
    history_dict             = hst.history
    loss_values              = history_dict['loss']
    validation_loss_values   = history_dict['val_loss']
    acc_values               = history_dict['accuracy']
    validation_acc_values    = history_dict['val_accuracy']
    prec_values              = history_dict['precision']
    validation_prec_values   = history_dict['val_precision']
    recall_values            = history_dict['recall']
    validation_recall_values = history_dict['val_recall']
    epochs                   = range(1,len(loss_values) + 1)
    fig, axes                = plt.subplots(1,4,figsize = (40,10))
    training_ts              = [loss_values, acc_values, prec_values, recall_values]
    validation_ts            = [validation_loss_values, validation_acc_values, validation_prec_values, validation_recall_values]
    metric_names             = ['loss', 'accuracy','precision','recall']
    for i in range(len(axes)):
        axes[i].plot(epochs,training_ts[i],color = 'r',label = 'training')
        axes[i].plot(epochs,validation_ts[i],color = 'b',label = 'validation')
        axes[i].set_xlabel('epoch')
        axes[i].set_ylabel(metric_names[i])
        axes[i].set_title(metric_names[i] + ' analysis')
        axes[i].set_xticks(np.arange(0,epochs[-1] + 1,5))
        axes[i].set_yticks(np.arange(0,1.1,0.1))
        axes[i].set_xlim([1,epochs[-1]])
        axes[i].set_ylim([np.min([np.min(training_ts[i]),np.min(validation_ts[i])]),np.max([np.max(training_ts[i]),np.max(validation_ts[i])])])
        axes[i].legend()
    plt.show()
        
training_path                  =  "/kaggle/input/dogs-cats-images/dataset/training_set/"
test_path                      =  "/kaggle/input/dogs-cats-images/dataset/test_set/"
validation_split               = 0.2
regression_problem             = False
target_img_shape_1             = 180
target_img_shape_2             = 180
target_img_channels            = 3
conv_filters                   = [32,64,128,128]      
conv_filter_shape              = [[3,3]]*4
conv_activation_function       = ['relu']*4
conv_padding                   = ['valid']*4
conv_pooling_type              = ['max']*4
conv_pooling_shape             = [[2,2]]*4
augment_data                   = True
rotation_range                 = 0.3
width_shift_range              = 0.2
height_shift_range             = 0.2
shear_range                    = 0.2
brightness_range               = [0.95,1.05]
zoom_range                     = 0.2
horizontal_flip                = True
fill_mode                      = 'nearest'
print_sample_input             = True
hidden_activation_function     = ['sigmoid']
hidden_layers_neurons          = [512]
hidden_layers_L1_coeffs        = [0.00]
hidden_layers_L2_coeffs        = [0.00]
hidden_layers_dropout          = [0.00]
final_activation_function      = 'sigmoid'
final_layer_neurons            = 1
model_optimizer                = 'Adam'
loss_function                  = 'binary_crossentropy'
metrics                        = ['accuracy',metrics.Precision(name='precision'),metrics.Recall(name='recall')]
n_epochs                       = 80
batch_size                     = 20
weight_labels                  = False
steps_per_epoch                = 100
validation_steps               = 50
vgg_hidden_activation_function = ['relu']
vgg_hidden_layers_neurons      = [256]
vgg_hidden_layers_L1_coeffs    = [0.00]
vgg_hidden_layers_L2_coeffs    = [0.00]
vgg_hidden_layers_dropout      = [0.00]
vgg_final_activation_function  = 'sigmoid'
vgg_final_layer_neurons        = 1
vgg_model_optimizer            = optimizers.RMSprop(lr=2e-5)
vgg_n_epochs                   = 35
vgg_steps_per_epoch            = 100
vgg_validation_steps           = 50
labels = ['cats','dogs']
new_training_path   = "../files/dogs-cats-images/dataset/training_set/"
new_validation_path = "../files/dogs-cats-images/dataset/validation_set/"
new_test_path       = "../files/dogs-cats-images/dataset/test_set/"
#shutil.rmtree(new_training_path)
#shutil.rmtree(new_validation_path) 
#shutil.rmtree(new_test_path)
[os.makedirs(new_training_path + label,exist_ok=True) for label in labels]
[os.makedirs(new_validation_path + label,exist_ok=True) for label in labels]
[os.makedirs(new_test_path + label,exist_ok=True) for label in labels]
for label in labels:
        training_filenames   = os.listdir(training_path + label + "/") 
        validation_filenames = random.sample(training_filenames, int(len(training_filenames)*validation_split))
        training_filenames   = [file for file in training_filenames if file not in validation_filenames]
        test_filenames       = os.listdir(test_path + label + "/") 
        for file in training_filenames:
            shutil.copy(training_path + label + "/" + file, new_training_path + label + "/")
        print('Training images transfer complete for label: ' + label + '. # transferred images: ' + str(len(training_filenames)))
        for file in validation_filenames:
            shutil.copy(training_path + label + "/" + file, new_validation_path + label + "/")
        print('Validation images transfer complete for label: ' + label + '. # transferred images: '  + str(len(validation_filenames)))
        for file in test_filenames:
            shutil.copy(test_path + label + "/" + file, new_test_path + label + "/")
        print('Test images transfer complete for label: ' + label + '. # transferred images: '  + str(len(test_filenames)))

if augment_data:
    train_datagen   = ImageDataGenerator(rescale            = 1./255,
                                         rotation_range     = rotation_range,
                                         width_shift_range  = width_shift_range,
                                         height_shift_range = height_shift_range,
                                         shear_range        = shear_range,
                                         brightness_range   = brightness_range,
                                         zoom_range         = zoom_range,
                                         horizontal_flip    = horizontal_flip,
                                         fill_mode          = fill_mode)
else:
    train_datagen   = ImageDataGenerator(rescale = 1./255)

validation_datagen   = ImageDataGenerator(rescale = 1./255)
test_datagen         = ImageDataGenerator(rescale = 1./255)
train_generator      = train_datagen.flow_from_directory(new_training_path,target_size = (target_img_shape_1, target_img_shape_2), batch_size = batch_size, class_mode = "binary")  
validation_generator = validation_datagen.flow_from_directory(new_validation_path,target_size = (target_img_shape_1, target_img_shape_2), batch_size = batch_size, class_mode = "binary") 
test_generator       = test_datagen.flow_from_directory(new_test_path,target_size = (target_img_shape_1, target_img_shape_2), batch_size = batch_size, class_mode = "binary") 
if print_sample_input:
    display_input_images(train_generator, 5, batch_size, [4,5], [20,20])
train_labels_weights_dict      = dataset_basic_info(train_generator, 'training')
validation_labels_weights_dict = dataset_basic_info(validation_generator, 'validation')
test_labels_weights_dict       = dataset_basic_info(test_generator, 'test')
model = build_model_cnn(regression_problem, conv_filters, conv_filter_shape, conv_activation_function, conv_padding, conv_pooling_type, conv_pooling_shape, hidden_layers_neurons, hidden_activation_function, 
                             hidden_layers_L1_coeffs, hidden_layers_L2_coeffs, hidden_layers_dropout, final_layer_neurons, final_activation_function, [target_img_shape_1, target_img_shape_2, target_img_channels], 
                             model_optimizer, loss_function, metrics)
early_exit      = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
best_checkpoint = ModelCheckpoint('.best_fit.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
#reduce_lr_loss  = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

if weight_labels:
        hst = model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = n_epochs, validation_data = validation_generator, validation_steps = validation_steps, class_weight=train_labels_weights_dict, callbacks =[early_exit, best_checkpoint])#, reduce_lr_loss] )
else:
        hst = model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = n_epochs, validation_data = validation_generator, validation_steps = validation_steps, callbacks =[early_exit, best_checkpoint])#, reduce_lr_loss] )
        
model.load_weights(filepath = '.best_fit.hdf5')
analyze_performances(hst, n_epochs)
test_loss_1, test_acc_1, test_prec_1, test_rec_1 = model.evaluate(test_generator)
print('The loss of the predictions on the test data set is: ' + str(round(test_loss_1,4)))
print('The accuracy of the predictions on the test data set is: ' + str(round(test_acc_1,4)))
pre_trained_conv = VGG16(weights = 'imagenet', include_top = False, input_shape = (target_img_shape_1, target_img_shape_2, target_img_channels))
model_pt = build_model_pretrained_cnn(pre_trained_conv, regression_problem, vgg_hidden_layers_neurons, vgg_hidden_activation_function, vgg_hidden_layers_L1_coeffs, vgg_hidden_layers_L2_coeffs, vgg_hidden_layers_dropout, vgg_final_layer_neurons, vgg_final_activation_function, vgg_model_optimizer, loss_function, metrics)
early_exit      = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
best_checkpoint = ModelCheckpoint('.best_fit_pre_trained.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

if weight_labels:
        vgg_hst = model_pt.fit(train_generator, steps_per_epoch = vgg_steps_per_epoch, epochs = vgg_n_epochs, validation_data = validation_generator, validation_steps = vgg_validation_steps, class_weight=train_labels_weights_dict, callbacks =[early_exit, best_checkpoint] )
else:
        vgg_hst = model_pt.fit(train_generator, steps_per_epoch = vgg_steps_per_epoch, epochs = vgg_n_epochs, validation_data = validation_generator, validation_steps = vgg_validation_steps, callbacks =[early_exit, best_checkpoint])
        
model_pt.load_weights(filepath = '.best_fit_pre_trained.hdf5')
analyze_performances(vgg_hst, vgg_n_epochs)
test_loss_2, test_acc_2, test_prec_2, test_rec_2 = model_pt.evaluate(test_generator)
print('The loss of the predictions on the test data set is: ' + str(round(test_loss_2,4)))
print('The accuracy of the predictions on the test data set is: ' + str(round(test_acc_2,4)))
print('The difference in accuracy (VGG16 - User model) in classifying test images is: ' + str(round(test_acc_2 - test_acc_1,4)))
print('The difference in precision (VGG16 - User model) in classifying test images is: ' + str(round(test_prec_2 - test_prec_1,4)))