# import libraries
import os
import shutil
import numpy as np
import glob   
import keras.backend as K
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import optimizers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
import scipy.misc
#from keras.applications import ResNet50
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

# for reading images
from matplotlib.pyplot import imshow
%matplotlib inline

# channels last is the format used by tensorflow 
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
# set to where the 'flowers' directory is located
data_dir = '../input/flowers-recognition/flowers'

# Training data dir
training_dir = 'Train'

# Test data dir
testing_dir = 'Test'

# Ratio of training and testing data
train_test_ratio = 0.8 


def split_dataset_into_test_and_train_sets(all_data_dir = data_dir, training_data_dir = training_dir, testing_data_dir=testing_dir, train_test_ratio = 0.8):

    # recreate test and train directories if they don't exist
    if not os.path.exists(training_data_dir):
        os.mkdir(training_data_dir)

    if not os.path.exists(testing_data_dir):
        os.mkdir(testing_data_dir)               
    
    num_training_files = 0
    num_testing_files = 0

    # iterate through the data directory 
    for subdir, dirs, files in os.walk(all_data_dir):
        
        category_name = os.path.basename(subdir)

        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name
        
        # creating subdirectory for each sub category
        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)   

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)
            
        file_list = glob.glob(subdir + '/*.jpg')

        print(str(category_name) + ' has ' + str(len(files)) + ' images') 
        random_set = np.random.permutation((file_list))
        
        # copy percentage of data from each category to train and test directory
        train_list = random_set[:round(len(random_set)*(train_test_ratio))] 
        test_list = random_set[-round(len(random_set)*(1-train_test_ratio)):]
        
        for lists in train_list : 
            shutil.copy(lists, training_data_dir + '/' + category_name + '/' )
            num_training_files += 1
  
        for lists in test_list : 
            shutil.copy(lists, testing_data_dir + '/' + category_name + '/' )
            num_testing_files += 1
  

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")
# split into train and test directories
split_dataset_into_test_and_train_sets()
# number of classes 
num_classes = 5

def get_model():
    
    # Get base model: ResNet50 
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    # freeze the layers in base model
    for layer in base_model.layers:
        layer.trainable = False
        
    # Get the output from the base model 
    base_model_ouput = base_model.output
    
    # Adding our own layers at the end
    # global average pooling: computes the average of all values in the feature map
    x = GlobalAveragePooling2D()(base_model_ouput)
    
    # fully connected and 5-softmax layer
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax', name='fcnew')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model
# Get the model
model = get_model()

# compile it
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# summary of model
model.summary()
# Using ImageDataGenerator for pre-processing

image_size = 224
batch_size = 64

# help(ImageDataGenerator)
train_data_gen = ImageDataGenerator(preprocessing_function = preprocess_input, 
                                    shear_range=0.2, zoom_range=0.2, 
                                    horizontal_flip=True)

# do only basic preprocessing for validation data (no data augmentation)
valid_data_gen = ImageDataGenerator(preprocessing_function = preprocess_input)

# create data generator objects
train_generator = train_data_gen.flow_from_directory(training_dir, (image_size,image_size), batch_size=batch_size, class_mode='categorical')
valid_generator = valid_data_gen.flow_from_directory(testing_dir, (image_size,image_size), batch_size=batch_size, class_mode='categorical')
# Training the newly added layers 
epochs = 10

# flow data (in batches) from directories (while simultaneously preprocessing/augmenting
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n//batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.n//batch_size,
    epochs=epochs,
    verbose=1)
epochs = 10

# training the model after 140 layers
split_at = 140
for layer in model.layers[:split_at]: layer.trainable = False
for layer in model.layers[split_at:]: layer.trainable = True
    
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Choosing lower learning rate for fine-tuning
# learning rate is generally 10-1000 times lower than normal learning rate when we are fine tuning the initial layers
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n//batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.n//batch_size,
    epochs=epochs,
    verbose=1)
