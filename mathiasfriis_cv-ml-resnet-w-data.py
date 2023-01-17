# ---------------------------------------------------------------------------------------
# -------------------------- copy images to dirs with label name ------------------------
# ---------------------------------------------------------------------------------------
import pandas as pd
import shutil
import os
import sys

# https://www.kaggle.com/c/dog-breed-identification/discussion/48908

def sepImgsToDirsFromPath(lbls_path, data_path, new_dir_path, file_prefix = "Image"):

    labels = pd.read_csv(lbls_path, header=None)
    labels.rename(columns={0:'Label'},inplace=True)
    labels['ID'] = labels.index + 1
    c = labels.columns
    labels[[c[0], c[1]]] = labels[[c[1], c[0]]]
    
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    
    for filename, class_name in labels.values:
        
        # add zero in front of classes
        if class_name < 10:
            prefix_zero = '0'
        else:
            prefix_zero = ''
            
        # Create subdirectory with class_name
        if not os.path.exists(new_dir_path + 'class_' + prefix_zero + str(class_name)):
            os.mkdir(new_dir_path + 'class_' + prefix_zero + str(class_name))
            
        src_path = data_path + file_prefix + str(filename) + '.jpg'
        dst_path = new_dir_path + 'class_' + prefix_zero + str(class_name) + '/' + file_prefix + str(filename) + '.jpg'
     
        try:
            shutil.copy(src_path, dst_path)
        except IOError as e:
            print('Unable to copy file {} to {}'.format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'.format(src_path, dst_path, sys.exc_info()))
            
    print("{} is ready with {} images in {} classes".format(new_dir_path, labels.shape[0], labels.values[-1][-1]))
    return labels.shape[0], labels.values[-1][-1]
    
# ---------- end sepImgsToDirsFromPath() ----------------
            
# what to do?
# https://towardsdatascience.com/a-bunch-of-tips-and-tricks-for-training-deep-neural-networks-3ca24c31ddc8

# train data
train_data_file_path = '../input/au-eng-cvml2020/Train/TrainImages/'
train_lbs_file_path = '../input/au-eng-cvml2020/Train/trainLbls.csv'
train_sep_dir = './train/'

# validation data
valid_data_file_path = '../input/au-eng-cvml2020/Validation/ValidationImages/'
valid_lbs_file_path = '../input/au-eng-cvml2020/Validation/valLbls.csv'
valid_sep_dir = './valid/'

# test data
test_data_file_path = '../input/au-eng-cvml2020/Test/'

# test data
num_train_images, num_train_classes = sepImgsToDirsFromPath(train_lbs_file_path, train_data_file_path, train_sep_dir)
num_valid_images, num_valid_classes = sepImgsToDirsFromPath(valid_lbs_file_path, valid_data_file_path, valid_sep_dir)

if num_valid_classes != num_train_classes:
    print('Error: not the same amount validation classes and training classes')
else:
    num_classes = num_train_classes
    print('--- ready! ---')
# --- IMPORTS ---
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from tensorflow import keras
from PIL import Image
import re
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import backend as K 
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from subprocess import check_output

# --- CONSTANTS ---
CSV_length = 4096;
N_classes = 29;
N_train_samples = 1000
N_validation_samples = 100
N_epochs = 50
#IM_SIZE = 256
#IM_SIZE = 128;
IM_SIZE = 64;


# --- INSTALL TALOS FOR HYPERPARAMETER OPTIMIZATION ---
#!pip install talos


"""# --- LOAD IMAGES ---

# Read training images
filelistTraining = sorted(glob.glob('../input/au-eng-cvml2020/Train/TrainImages/*.jpg'));
filelistTraining.sort(key=lambda f: int(re.sub('\D', '', f))) # Sort, so it goes 1,2,3 instead of 1,10,100, etc...
trainingImages = np.array([np.array(Image.open(fname)) for fname in filelistTraining]);

# Read validaiton images
filelistValidation = sorted(glob.glob('../input/au-eng-cvml2020/Validation/ValidationImages/*.jpg'));
filelistValidation.sort(key=lambda f: int(re.sub('\D', '', f))) # Sort, so it goes 1,2,3 instead of 1,10,100, etc...
validationImages = np.array([np.array(Image.open(fname)) for fname in filelistValidation]);

# Read test images
filelistTest = sorted(glob.glob('../input/au-eng-cvml2020/Test/TestImages/*.jpg'));
filelistTest.sort(key=lambda f: int(re.sub('\D', '', f))) # Sort, so it goes 1,2,3 instead of 1,10,100, etc...
testImages = np.array([np.array(Image.open(fname)) for fname in filelistTest]);

N_training = trainingImages.shape[0];
N_validation = validationImages.shape[0];
N_test = testImages.shape[0];
print("trainingImages.shape = ");
print(trainingImages.shape);
print("N_training = ");
print(N_training);

imagesPreprocessed = False;"""
# --- READ LABELS ---
trainingLabels = pd.read_csv('../input/au-eng-cvml2020/Train/trainLbls.csv', header=None);
#trainingVectors = pd.read_csv('../input/au-eng-cvml2020/Train/trainVectors.csv', header=None).T;


validationLabels = pd.read_csv('../input/au-eng-cvml2020/Validation/valLbls.csv', header=None);
#validationVectors = pd.read_csv('../input/au-eng-cvml2020/Validation/valVectors.csv', header=None).T;

testVectors = pd.read_csv('../input/au-eng-cvml2020/Test/testVectors.csv', header=None).T;

vectorsPreprocessed = False;
"""# --- PREPROCESS IMAGES ---

# Resize images
if IM_SIZE < trainingImages.shape[1]:
    print("trainingImages.shape PRE resizing = ")
    print(trainingImages.shape)

    _antialias = False

    trainingImages = tf.image.resize(
    trainingImages, (IM_SIZE,IM_SIZE), preserve_aspect_ratio=False,
    antialias=_antialias, name=None
    )

    validationImages = tf.image.resize(
    validationImages, (IM_SIZE,IM_SIZE), preserve_aspect_ratio=False,
    antialias=_antialias, name=None
    )

    testImages = tf.image.resize(
    testImages, (IM_SIZE,IM_SIZE), preserve_aspect_ratio=False,
    antialias=_antialias, name=None
    )

print("trainingImages.shape POST resizing = ")
print(trainingImages.shape)
print("validationImages.shape POST resizing = ")
print(validationImages.shape)
print("testImages.shape POST resizing = ")
print(testImages.shape)

# Normalize images

trainingImages = (trainingImages/255.0 - 0.5)*2
validationImages = (validationImages/255.0 - 0.5)*2
testImages = (testImages/255.0 - 0.5)*2"""

"""trainingImages = (trainingImages/255.0)
validationImages = (validationImages/255.0)
testImages = (testImages/255.0)"""

"""trainingImages /=255
validationImages /= 255
testImages /= 255"""
"""# --- SHOW EXAMPLE OF PROCESSED TRAINING IMAGE ---

print(trainingImages.shape)
plt.figure()
plt.imshow(trainingImages[0])
plt.show()"""
# --- ENCODE LABELS ---
from sklearn.preprocessing import LabelEncoder 
 
le = LabelEncoder() 
le.fit(trainingLabels) 
train_labels_enc = le.transform(trainingLabels) 
validation_labels_enc = le.transform(validationLabels) 
 
#print(trainingLabels[1495:1505], train_labels_enc[1495:1505])
# --- PREPARE DATA ---
batch_size = 80
rescale = None

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=rescale,
        shear_range=0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        data_format='channels_last',
        samplewise_center=True, 
        samplewise_std_normalization=True
)

# this is the augmentation configuration we will use for testing:
# only rescaling
validation_datagen = ImageDataGenerator(
    rescale=rescale,
    data_format='channels_last',
    samplewise_center=True, 
    samplewise_std_normalization=True
)

# Fit ImageDataGenerators to their respective data sets.
# HOW TO DO THIS? MAYBE STANDARDIZE BEFORE LOADING INTO DIRECTORIES?

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        './train',
        target_size=(IM_SIZE, IM_SIZE),
        batch_size=batch_size,
        class_mode='categorical')


validation_generator = validation_datagen.flow_from_directory(
        './valid',
        target_size=(IM_SIZE, IM_SIZE),
        batch_size=batch_size,
        class_mode='categorical')
"""# --- TEST TRAINING DATA GENERATOR ---
print(trainingImages[0].shape)
array_to_img(trainingImages[0])
img_test = trainingImages[0]
img_val = validationImages[0]
x = img_to_array(img_test)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# Plot original image
plt.figure();
plt.subplot(2,1,1);
plt.imshow(img_test);
plt.subplot(2,1,2);
plt.imshow(img_val);

# the .flow() command below generates batches of randomly transformed images
# and plots them
plt.figure();
i = 0
for batch in train_datagen.flow(x, batch_size=1,
                          save_to_dir=None, save_prefix='data_gen_test', save_format='jpeg'):
    #print(i)
    plt.subplot(3,3,i+1);
    plt.imshow(batch[0]);
    i += 1
    if i >= 9:
        break  # otherwise the generator would loop indefinitely
    plt.title('Outputs from Training Generator')
        
plt.show();"""
# --- IMPORT ResNet WITHOUT TOP LAYERS ---
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
K.set_learning_phase(1)

_weights = 'imagenet'

restnet = ResNet50(include_top=False, weights=_weights, input_shape=(IM_SIZE,IM_SIZE,3))

output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)

restnet = Model(inputs = restnet.input, outputs=output)

for layer in restnet.layers:
    layer.trainable = False
restnet.summary()
# --- ADD FULLY CONNECTED LAYERS ---

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

"""model = Sequential()
model.add(restnet)
model.add(Dense(32, activation='relu', input_dim=32768))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(N_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-6),
              metrics=['accuracy'])
model.summary()"""

"""model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])"""

"""# initiate Adam optimizer
opt = Adam(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])"""


#model.add(Dense(N_classes+1, activation='sigmoid'))

"""
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
              """
"""
# initiate Adam optimizer
opt = Adam(lr=0.001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])
              """
#model.summary()

"""# --- FIT MODEL ---
history = model.fit_generator(train_generator, 
                              steps_per_epoch=100, 
                              epochs=100,
                              validation_data=validation_generator, 
                              validation_steps=50, 
                              verbose=1)

# --- SAVE MODEL ---
model.save('CV_ML_Kaggle_weights.h5')"""
# --- SET LAST CONVOLUTIONAL LAYER AS TRAINABLE ---
restnet.trainable = True
set_trainable = False
for layer in restnet.layers:
    if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
layers = [(layer, layer.name, layer.trainable) for layer in restnet.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model_finetuned = Sequential()

# add Base Layers
model_finetuned.add(restnet)

"""model_finetuned.add(Dense(512, activation='relu', input_dim=32768))
model_finetuned.add(Dropout(0.3))
model_finetuned.add(Dense(512, activation='relu'))
model_finetuned.add(Dropout(0.3))
model_finetuned.add(Dense(N_classes, activation='softmax'))"""

# Try with architecture as in [https://arxiv.org/ftp/arxiv/papers/1903/1903.10035.pdf, p 9]
model_finetuned.add(BatchNormalization(momentum=0.1, epsilon=1e-5))
model_finetuned.add(Dropout(0.25))
model_finetuned.add(Dense(128, activation='relu', input_dim=32768))
model_finetuned.add(BatchNormalization(momentum=0.1, epsilon=1e-5))
model_finetuned.add(Dropout(0.5))
model_finetuned.add(Dense(64, activation='relu'))
model_finetuned.add(Dense(N_classes, activation='softmax'))



model_finetuned.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'])

"""model_finetuned.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])"""

"""model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])"""

model_finetuned.summary()
try:
    # Load pre-trained weights
    model.load_weights('../input/model-weights-cnn-50ish/model_weights_CNN_50ish.h5')
    print('Loaded pre-trained weights')
except IOError as e:
    # Train model
    stoppingCriterion = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=5, verbose=1, mode='max', baseline=None, restore_best_weights=True);

    print('Training model.')
    history_1 = model_finetuned.fit_generator(train_generator, 
                                      steps_per_epoch=100, 
                                      epochs=3000,
                                      validation_data=validation_generator, 
                                      validation_steps=100, 
                                      verbose=1)
    # Save model
    model_finetuned.save('ResNet_finetuned_model.h5')
    
    # save history object
    with open('ResNet_finetuned_history.json', 'w') as f:
        json.dump(hist.history, f)
import os
import regex
import shutil

# --- RENAME TEST FILES ---
# Rename test files, so they're called 0001, 0010, 0100 etc.. instead of 1, 10 100, etc...

testDir = 'test/testImages'

# Create subdirectory.
if not os.path.exists(testDir):
    os.mkdir(testDir)

print('Renaming files...')
# Get list of files.
filelistTest = sorted(glob.glob('../input/au-eng-cvml2020/Test/TestImages/*jpg'));
for fname in filelistTest:
    # Extract number of file.
    filename_string = os.path.splitext(fname)[0]
    number = int(re.findall(r'\d+', fname)[-1])
    
    # Create new filename
    newFilename = 'Image'
    
    # Add correct number of zeros.
    if number < 10:
        newFilename += '000'
    elif number <100:
        newFilename += '00'
    elif number < 1000:
        newFilename += '0'
    
    # add number
    newFilename += str(number)
    newFilename +='.jpg'
    
    #print('newFilename = ')
    #print(newFilename)
    
    shutil.copy(fname, os.path.join(testDir, newFilename))
print('All files renamed')
# --- PREDICT ---
    
test_generator = validation_datagen.flow_from_directory(
        './test',
        target_size=(IM_SIZE, IM_SIZE),
        batch_size=batch_size,
shuffle = False)

#print(test_generator.directory)
#print(test_generator.filenames)
    
predictions_images = model.predict_generator(test_generator)
print(predictions_images)
print("predictions_images.shape = ")
print(predictions_images.shape)
# Print our model's predictions.

predictions_images = np.argmax(predictions_images, axis=1)
# add 1, because classes go from 1 and up, not 0
predictions_images = predictions_images + 1
print(predictions_images)
print('class_min = ')
print(min(predictions_images))

print('class_max = ')
print(max(predictions_images))
# --- EXPORT TO CSV ---from subprocess import check_output
data_to_submit = pd.DataFrame({
    'ID':range(1,len(predictions_images)+1),
    'Label':predictions_images
})

data_to_submit.to_csv('Results_raw_images_ResNet_finetuned.csv', index = False)
# Noget med noget training phase, som skal sætte til 1. K.set_learning_phase(1)