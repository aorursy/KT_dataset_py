# Much of this work is inspired by the wonderful kernel 'Train Simple XRay CNN' by Kevin Mader
# Some code fragments are sourced or adapted directly from this Kernel
# I cite his work when appropriate, including URL/Dates, and it can also be referenced here: https://www.kaggle.com/kmader/train-simple-xray-cnn

# Much of my thinking is also guided by Google's nice explanation of AutoML for Vision. General principles are quite useful
# https://cloud.google.com/vision/automl/docs/beginners-guide

# Lastly, I found this article on how AI is changing radiology imaging quite interesting
# https://healthitanalytics.com/news/how-artificial-intelligence-is-changing-radiology-pathology
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
xray_data = pd.read_csv('../input/Data_Entry_2017.csv')

# see how many observations there are
num_obs = len(xray_data)
print('Number of observations:',num_obs)

# examine the raw data before performing pre-processing
xray_data.head(5) # view first 5 rows
#xray_data.sample(5) # view 5 randomly sampled rows112120
# had to learn this part from scratch, hadn't gone so deep into file paths before!
# looked at glob & os documentation, along with Kevin's methodology to get this part working
# note: DON'T adjust this code, it's short but took a long time to get right
# https://docs.python.org/3/library/glob.html
# https://docs.python.org/3/library/os.html
# https://www.geeksforgeeks.org/os-path-module-python/ 
    
from glob import glob
#import os # already imported earlier

my_glob = glob('../input/images*/images/*.png')
print('Number of Observations: ', len(my_glob)) # check to make sure I've captured every pathway, should equal 112,120
# Map the image paths onto xray_data
# Credit: small helper code fragment adapted from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn
full_img_paths = {os.path.basename(x): x for x in my_glob}
xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)
# Explore the dataset a bit

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
# now, let's see how many cases present for each of of our 14 clean classes (which excl. 'No Finding')
clean_labels = xray_data[dummy_labels].sum().sort_values(ascending= False) # get sorted value_count for clean labels
print(clean_labels) # view tabular results

# plot cases using seaborn barchart
clean_labels_df = clean_labels.to_frame() # convert to dataframe for plotting purposes
sns.barplot(x = clean_labels_df.index[::], y= 0, data = clean_labels_df[::], color = "green"), plt.xticks(rotation = 90) # visualize results graphically
## MODEL CREATION PHASE STARTS HERE

# create vector as ground-truth, will use as actuals to compare against our predictions later
xray_data['target_vector'] = xray_data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])
xray_data.head() # take a look to ensure target_vector makes sense
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
        )
# Credit: Helper function for image augmentation - Code sourced directly from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn

# This Flow_from function is actually based on the default function from Keras '.flow_from_dataframe', but is more flexible
# Base function reference: https://keras.io/preprocessing/image/
# Specific notes re function: https://github.com/keras-team/keras/issues/5152

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
# Can use flow_from_dataframe() for training and validation - simply pass arguments through to function parameters
# Credit: Code adapted from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn

image_size = (224, 224) # image re-sizing target
train_gen = flow_from_dataframe(data_gen, train_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale', batch_size = 32 )
valid_gen = flow_from_dataframe(data_gen, test_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale', batch_size = 32)

# define test sets
test_X, test_Y = next(flow_from_dataframe(data_gen, test_set, path_col = 'full_path', y_col = 'target_vector', target_size = image_size, color_mode = 'grayscale', 
                                          batch_size = 2048))
## On to the fun stuff! Create a convolutional neural network model to train from scratch

# Import relevant libraries
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import keras
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg19 import VGG19
import numpy as np
import keras
import tensorflow as tf
# Create CNN model
# Will use a combination of convolutional, max pooling, and dropout layers for this purpose
np.random.seed(1000)
#Instantiate an empty model
model = tf.keras.applications.VGG19(
    include_top=True,
    weights=None,
    input_shape=(224,224,1),
    pooling=max,
    classes=14,
)

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
# set up a checkpoint for model training
# https://keras.io/callbacks/
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='weights.best.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only = True)
callbacks_list = [checkpointer]
## Fit the model!
# Note: This first model will be fitted very quickly, will fit a higher accuracy modelthat requires more training time after this
# Documentation: https://keras.io/models/model/


model.fit_generator(generator = train_gen, steps_per_epoch = 700 , epochs = 20, callbacks = callbacks_list, validation_data = (test_X, test_Y))

## How I decided which fit function to use
# model.fit_generator() v. model.fit() - https://datascience.stackexchange.com/questions/34444/what-is-the-difference-between-fit-and-fit-generator-in-keras
# 2nd good example: https://medium.com/difference-engine-ai/keras-a-thing-you-should-know-about-keras-if-you-plan-to-train-a-deep-learning-model-on-a-large-fdd63ce66bd2
# Basically, for large data-sets like this one, it's best-practice to use model.fit_generator()
# This is because i) dataset can't be loaded into memory all at once, ii) we're already doing image augmentation, so this solution meshes nicely
# Takeaway - use model.fit_generator() for this dataset
# Make prediction based on our fitted model
quick_model_predictions = model.predict(test_X, batch_size = 64, verbose = 1)
# Credit: Helper function for Plotting - Code sourced directly from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn

# import libraries
from sklearn.metrics import roc_curve, auc

# create plot
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (i, label) in enumerate(dummy_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,i].astype(int), quick_model_predictions[:,i])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

# Set labels for plot
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('quick_trained_model.png')
## See previous code snippets for all references

# Run a longer, more detailed model
"""
model.fit_generator(generator = train_gen, steps_per_epoch = 50, epochs = 5, callbacks = callbacks_list, validation_data = (test_X, test_Y))

# Make prediction based on our fitted model
deep_model_predictions = model.predict(test_X, batch_size = 64, verbose = 1)

# create plot
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (i, label) in enumerate(dummy_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,i].astype(int), deep_model_predictions[:,i])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

# Set labels for plot
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('deep_trained_model.png')"""
# The AUC curve looks good - it shows much tighter results than before in terms of the spread
# Clearly demonstrates the model works and has significant predictive power!
# Great ending, stopping here