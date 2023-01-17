# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# load help packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # additional plotting functionality

# Input data files are available in the "../input/" directory.
# For example, running the below code (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# load data
xray_data = pd.read_csv('../input/data-entry-csv/Data_Entry_2017.csv')

# see how many observations there are
num_obs = len(xray_data)
print('Number of observations:',num_obs)

# examine the raw data before performing pre-processing
xray_data.head(5) # view first 5 rows
#xray_data.sample(5) # view 5 randomly sampled rows
# had to learn this part from scratch, hadn't gone so deep into file paths before!
# looked at glob & os documentation, along with Kevin's methodology to get this part working
# note: DON'T adjust this code, it's short but took a long time to get right
# https://docs.python.org/3/library/glob.html
# https://docs.python.org/3/library/os.html
# https://www.geeksforgeeks.org/os-path-module-python/ 
    
from glob import glob
#import os # already imported earlier

my_glob = glob('../input/data/images*/images/*.png')
print('Number of Observations: ', len(my_glob)) # check to make sure I've captured every pathway, should equal 112,120
# Map the image paths onto xray_data
# Credit: small helper code fragment adapted from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn
full_img_paths = {os.path.basename(x): x for x in my_glob}
xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)
# Q: how many unique labels are there? A: many (836) because of co-occurence
# Note: co-occurence will turn out to be a real pain to deal with later, but there are several techniques that help us work with it successfully
num_unique_labels = xray_data['Finding Labels'].nunique()
print('Number of unique labels:',num_unique_labels)

# let's look at the label distribution to better plan our next step
count_per_unique_label = xray_data['Finding Labels'].value_counts() # get frequency counts per label
df_count_per_unique_label = count_per_unique_label.to_frame() # convert series to dataframe for plotting purposes

print(df_count_per_unique_label) # view tabular results
sns.barplot(x = df_count_per_unique_label.index[:20], y="Finding Labels", data=df_count_per_unique_label[:20], color = "green"), plt.xticks(rotation = 90) # visualize results graphically
# define dummy labels for one hot encoding - simplifying to 14 primary classes (excl. No Finding)
dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'] # taken from paper

# One Hot Encoding of Finding Labels to dummy_labels
for label in dummy_labels:
    xray_data[label] = xray_data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
xray_data.head(20) # check the data, looking good!
#[0,1,1,0,0]
# now, let's see how many cases present for each of of our 14 clean classes (which excl. 'No Finding')
clean_labels = xray_data[dummy_labels].sum().sort_values(ascending= False) # get sorted value_count for clean labels
print(clean_labels) # view tabular results

# plot cases using seaborn barchart
clean_labels_df = clean_labels.to_frame() # convert to dataframe for plotting purposes
sns.barplot(x = clean_labels_df.index[::], y= 0, data = clean_labels_df[::], color = "green"), plt.xticks(rotation = 90) # visualize results graphically
## MODEL CREATION PHASE STARTS HERE

# create vector as ground-truth, will use as actuals to compare against our predictions later
xray_data['target_vector'] = xray_data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])





xray_data.head(1) # take a look to ensure target_vector makes sense

print(xray_data['target_vector'][0])
# split the data into a training and testing set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(xray_data, test_size = 0.2, random_state = 1993)
# quick check to see that the training and test set were split properly
print('training set - # of observations: ', len(train_set))
print('test set - # of observations): ', len(test_set))
print('prior, full data set - # of observations): ', len(xray_data))
# IMAGE PRE-PROCESSING
# See Keras documentation: https://keras.io/preprocessing/image/

# Create ImageDataGenerator, to perform significant image augmentation
# Utilizing most of the parameter options to make the image data even more robust
from keras.preprocessing.image import ImageDataGenerator
data_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
# Credit: Helper function for image augmentation - Code sourced directly from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn

# This Flow_from function is actually based on the default function from Keras '.flow_from_dataframe', but is more flexible
# Base function reference: https://keras.io/preprocessing/image/
# Specific notes re function: https://github.com/keras-team/keras/issues/5152

def flow_from_dataframe(img_data_gen, _set, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(_set[path_col].values[0]) ## base_dir = full path to first image in batch
    print('## Ignore next message from keras, values are replaced anyways')
    print(base_dir)
    ##dflow_args = (target_size = image_size, color_mode = 'grayscale', batch_size = 32)
    ret = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    print(1,ret)
    ret.filenames = _set[path_col].values ## full path to all images in batch
    print(2,ret.filenames)
    ret.classes = np.stack(_set[y_col].values) ## all labels [0,1]
    print(3,ret.classes)
    ret.samples = _set.shape[0] #number of rows in batch
    print(4,ret.samples)
    ret.n = _set.shape[0] # number of rows in batch
    print(5,ret.n)
    ret._set_index_array() # absar
    ret.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(_set.shape[0]))
    print("final",ret)
    return ret
# Can use flow_from_dataframe() for training and validation - simply pass arguments through to function parameters
# Credit: Code adapted from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn

image_size = (224, 224) # image re-sizing target
train_gen = flow_from_dataframe(data_gen, train_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale', batch_size = 64)
valid_gen = flow_from_dataframe(data_gen, test_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale', batch_size = 128)

# define test sets
test_X, test_Y = next(flow_from_dataframe(data_gen, test_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale', 
                                          batch_size = 2048))
# print("train gen",next(train_gen))
# print("valid gen",next(valid_gen))
# print("text x",test_X)
# print("test y",test_Y)
# # this will copy the pretrained weights to our kernel
# !mkdir ~/.test
# !mkdir ~/.test/models
# !cp ../input/keras-pretrained-models/*notop* ~/.test/models/
# !cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.test/models/

# # importing the libraries
# from keras.models import Model
# from keras.layers import Flatten, Dense
# from keras.applications import VGG16
# #from keras.preprocessing import image

# IMAGE_SIZE = [224, 224]  # we will keep the image size as (64,64). You can increase the size for better results. 

# # loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
# vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

# # this will exclude the initial layers from training phase as there are already been trained.
# for layer in vgg.layers:
#     layer.trainable = False

# x = Flatten()(vgg.output)
# x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
# x = Dense(14, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

# model = Model(inputs = vgg.input, outputs = x)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

# # history = model.fit_generator(train_gen,
# #                    steps_per_epoch = 1401,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results. 
# #                    epochs = 5,  # change this for better results
# #                    validation_data = (test_X, test_Y),
# #                    validation_steps = 10)  # this should be equal to total number of images in validation set.

# # predict = model.predict_generator(train_gen,steps=175)


# print(predict)
# 


# print(predict[0])
## On to the fun stuff! Create a convolutional neural network model to train from scratch

# Import relevant libraries
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.models import Sequential

model=Sequential()

#1 conv layer
model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding="same",activation="relu",input_shape=test_X.shape[1:]))

#1 max pool layer
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(BatchNormalization())

#2 conv layer
model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="same",activation="relu"))

#2 max pool layer
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))


model.add(BatchNormalization())

#3 conv layer
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu"))

#4 conv layer
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu"))

#5 conv layer
model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu"))

#3 max pool layer
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(BatchNormalization())


model.add(Flatten())

#1 dense layer
model.add(Dense(500,activation="relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

#2 dense layer
model.add(Dense(500,activation="relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

#3 dense layer
model.add(Dense(500,activation="relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

#output layer
model.add(Dense(14, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# # set up a checkpoint for model training
# # https://keras.io/callbacks/
# from keras.callbacks import ModelCheckpoint

# checkpointer = ModelCheckpoint(filepath='weights.best.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only = True)
# callbacks_list = [checkpointer]
# # 
model.fit_generator(generator = train_gen, 
                    steps_per_epoch = 1401, 
                    epochs = 5,
                    validation_steps = 10,
                    validation_data = (test_X, test_Y)
                   )


# Make prediction based on our fitted model
# quick_model_predictions = model.predict(test_X, batch_size = 64, verbose = 1)
# Credit: Helper function for Plotting - Code sourced directly from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn

# import libraries
# from sklearn.metrics import roc_curve, auc

# # create plot
# fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
# for (i, label) in enumerate(dummy_labels):
#     fpr, tpr, thresholds = roc_curve(test_Y[:,i].astype(int), quick_model_predictions[:,i])
#     c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

# # Set labels for plot
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# fig.savefig('quick_trained_model.png')
## See previous code snippets for all references

# Run a longer, more detailed model
# model.fit_generator(generator = train_gen, steps_per_epoch = 50, epochs = 5, callbacks = callbacks_list, validation_data = (test_X, test_Y))

# # Make prediction based on our fitted model
# deep_model_predictions = model.predict(test_X, batch_size = 64, verbose = 1)

# # create plot
# fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
# for (i, label) in enumerate(dummy_labels):
#     fpr, tpr, thresholds = roc_curve(test_Y[:,i].astype(int), deep_model_predictions[:,i])
#     c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

# # Set labels for plot
# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# fig.savefig('deep_trained_model.png')
# The AUC curve looks good - it shows much tighter results than before in terms of the spread
# Clearly demonstrates the model works and has significant predictive power!
# Great ending, stopping here