import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import cv2

from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from os import makedirs
from os.path import expanduser, exists, join

# Create the directories for the pretrained models
!ls ../input/keras-pretrained-models/   # just to be sure the data is here

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
    
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
# Set the train and test folder paths.
# NOTE: train and test are now in the 'dog-breed-identification' folder since the Keras Pretrained Models data/directory is added
train_folder = '../input/dog-breed-identification/train/'
test_folder = '../input/dog-breed-identification/test/'
# get the dog image ids and labels/breed
train_dogs = pd.read_csv('../input/dog-breed-identification/labels.csv')
train_dogs.head()
# Take a look at the class/breed distribution
ax=pd.value_counts(train_dogs['breed'],ascending=True).plot(kind='barh',
                                                       fontsize="40",
                                                       title="Class Distribution",
                                                       figsize=(50,100))
ax.set(xlabel="Images per class", ylabel="Classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)
plt.show()
# Get the top 20 breeds which is what we use in this notebook
top_breeds = sorted(list(train_dogs['breed'].value_counts().head(20).index))
train_dogs = train_dogs[train_dogs['breed'].isin(top_breeds)]
# Let's see what breeds are the top 20
print(top_breeds)
train_dogs.shape
# Get the labels of the top 20
target_labels = train_dogs['breed']
# One hot code the labels - need this for the model
one_hot = pd.get_dummies(target_labels, sparse = True)
one_hot_labels = np.asarray(one_hot)
# add the actual path name of the pics to the data set
train_dogs['image_path'] = train_dogs.apply( lambda x: (train_folder + x["id"] + ".jpg" ), axis=1)
train_dogs.head()
# Convert the images to arrays which is used for the model. Inception uses image sizes of 299 x 299
train_data = np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in train_dogs['image_path'].values.tolist()]).astype('float32')
# Split the data into train and validation. The stratify parm will insure  train and validation  
# will have the same proportions of class labels as the input dataset.
x_train, x_validation, y_train, y_validation = train_test_split(train_data, target_labels, test_size=0.2, stratify=np.array(target_labels), random_state=100)
# Need to know how many rows in each of the train/test split so we can 
# calculate steps_per_epoch and validatoin_steps for the model.fit_generator
print ('x_train shape = ', x_train.shape)
print ('x_validation shape = ', x_validation.shape)
# Calculate the value counts for train and validation data and plot to show a good stratify
# the plot should show an equal percentage split for each class
data = y_train.value_counts().sort_index().to_frame()   # this creates the data frame with train numbers
data.columns = ['train']   # give the column a name
data['validation'] = y_validation.value_counts().sort_index().to_frame()   # add the validation numbers
new_plot = data[['train','validation']].sort_values(['train']+['validation'], ascending=False)   # sort the data
new_plot.plot(kind='bar', stacked=True)
plt.show()
# Need to convert the train and validation labels into one hot encoded format
y_train = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).as_matrix()
# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=30, 
                                   # zoom_range = 0.3, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=10, seed=10)
# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=10, seed=10)
# Get the InceptionV3 model so we can do transfer learning
base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))
# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer and a logistic layer with 20 classes 
#(there will be 120 classes for the final submission)
x = Dense(512, activation='relu')(x)
predictions = Dense(20, activation='softmax')(x)
# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)
# first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
# Compile with Adam
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit_generator(train_generator,
                      steps_per_epoch = 175,
                      validation_data = val_generator,
                      validation_steps = 44,
                      epochs = 10,
                      verbose = 2)
# Use the sample submission file to set up the test data - x_test
# test_data = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
# Creae the x_test
# x_test = []
# for i in tqdm(test_data['id'].values):
#     img = cv2.imread('../input/dog-breed-identification/test/{}.jpg'.format(i))
#     x_test.append(cv2.resize(img, (299, 299)))
# Make it an array
# x_test = np.array(x_test, np.float32) / 255.
# Predict x_test
# predictions = model.predict(x_test, verbose=2)
# Set column names to those generated by the one-hot encoding earlier
# col_names = one_hot.columns.values
# Create the submission data.
# submission_results = pd.DataFrame(predictions, columns = col_names)
# Add the id as the first column
# submission_results.insert(0, 'id', test_data['id'])
# Save the submission
# submission_results.to_csv('submission.csv', index=False)
