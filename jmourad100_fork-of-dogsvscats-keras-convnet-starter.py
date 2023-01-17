import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from keras import layers, models, optimizers
from keras.models import Model
from keras.layers import (Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation,
                          BatchNormalization, Concatenate)
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os, cv2, random, re, csv
from tqdm import tqdm

# %matplotlib inline
# import keras
# keras.backend.backend() #=> Tensorflow
TRAIN_DIR = '../input/train'
TEST_DIR = '../input/test'

ROWS = 200
COLS = 200
CHANNELS = 3

# because of the limited resources we have, we have to adapt the BATCH_SIZE 
# With image size and complexity of the model (nb params)
BATCH_SIZE=50
EPOCHS=40
# This function resizes the images to 64x64 and samples 2000 images (8%) of the data.
# I also separated cats and dogs for exploratory analysis

train_images = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR)]
train_dogs = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images = [TEST_DIR+"/"+i for i in os.listdir(TEST_DIR)]

#### For testing purposes
# train_images = train_dogs[:4000] + train_cats[:4000]
# test_images = test_images[:1000]
# random.shuffle(train_images)

# # Helper function to sort the image files based on the numeric value in each file name.
# def atoi(text): return int(text) if text.isdigit() else text
# def natural_keys(text): return [ atoi(c) for c in re.split('(\d+)', text) ]
# train_dogs.sort(key=natural_keys);
# train_cats.sort(key=natural_keys);
####

def read_image(file_path):
    img= cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

# Alternative function
from keras.preprocessing import image
def read_image_(file_path):
        img = image.load_img(file_path, target_size=(ROWS, COLS))
        return image.img_to_array(img)

def prep_data(images):
    X = [] # images as arrays
    y = [] # labels
    for image_file in tqdm(images):
        image = read_image_(image_file)
        X.append(image)
        if 'dog' in image_file: y.append(1)
        elif 'cat' in image_file: y.append(0)      
    return X, y

print("Processing Train images")
X_train, y_train = prep_data(train_images)

print("Train: {} images with shape {}".format(len(X_train),X_train[0].shape))
print("Test: {} images".format(len(test_images)))
# We're dealing with classification problem here - (1) dogs (0) cats
labels = [1 if 'dog' in l else 0 for l in train_images]
sns.countplot(labels)
plt.title('Cats and Dogs');
# A quick side-by-side comparison of the animals
for idx in range(2):
    idx = idx + np.random.randint(low=1, high=100); # To randomize images
    cat = read_image(train_cats[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(15, 5))
    f = plt.imshow(pair)
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
    plt.show()
def convBatchActivMax_block(_input, N_Filters, N, kernel, blockNumber):
    # N is used to Variate number of filters for each block
    x = Conv2D(N_Filters* N, kernel_size=kernel, padding='same', activation='relu', name='block{}_conv{}_{}'.format(blockNumber, 1, kernel))(_input)
    x = Conv2D(N_Filters* N, kernel_size=kernel, padding='same', name='block{}_conv{}_{}'.format(blockNumber, 2, kernel))(x)
    x = BatchNormalization(name="block{}_BatchNorm_{}".format(blockNumber,kernel))(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((2,2), strides=(2,2), name='block{}_pool_{}'.format(blockNumber, kernel))(x)
    return x

def build_model(N_Filters=32):
    input_layer = Input((ROWS, COLS, CHANNELS), name="InputLayer")
    
    #----- Branch 1-------
    ######################
    # Block 1
    x1 = convBatchActivMax_block(input_layer, N_Filters, 1, 3, 1)
    # Block 2
    x1 = convBatchActivMax_block(x1, N_Filters, 2, 3, 2)
    # Block 3
    x1 = convBatchActivMax_block(x1, N_Filters, 3, 3, 3)
    
    #----- Branch 2-------
    ######################
    x2 = convBatchActivMax_block(input_layer, N_Filters, 1, 5, 1)
    # Block 2
    x2 = convBatchActivMax_block(x2, N_Filters, 2, 5, 2)
    # Block 3
    x2 = convBatchActivMax_block(x2, N_Filters, 3, 5, 3)
    
    OutConcat = Concatenate()([x1,x2])
    x = Conv2D(N_Filters*3, 1, activation='relu')(OutConcat)
    
    x = Flatten(name='flatten')(x)
    x = Dense(N_Filters*10, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(N_Filters*4, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input_layer, output)
    model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model
model = build_model()
model.summary()
#################
# Plot The Model
#################
from keras.utils import plot_model 
plot_model(model, to_file='keras-baseline-architecture.png')

from IPython.display import Image
Image(filename='keras-baseline-architecture.png') 
# First split the data in two sets, 80% for training, 20% for Val)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=1)

# Augmentation configuration to use for training and validation
train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=40,
                width_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Prepare generators for training and validation sets
train_generator = train_datagen.flow(np.array(X_train), y_train, batch_size=BATCH_SIZE)
validation_generator = validation_datagen.flow(np.array(X_val), y_val, batch_size=BATCH_SIZE)
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
Callbacks = [
    EarlyStopping(monitor='val_loss', mode = 'min',patience=10, verbose=1),
    ModelCheckpoint('BestModel.hdf5', monitor='val_loss', mode='min', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=5, min_lr=0.0001, verbose=1)
]

history = model.fit_generator(
    train_generator, 
    steps_per_epoch = len(X_train) // BATCH_SIZE,
    callbacks = Callbacks,
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = len(X_val) // BATCH_SIZE
)
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# model.save_weights('model_wieghts.h5')
# model.save('model_keras.h5')
from keras.models import load_model
best_model = load_model('BestModel.hdf5')

# making_test_data() differs from prep_data(), because here we need the image & it's id
def making_test_data():
    testing_data = []
    for img_path in tqdm(test_images):
        img_num = img_path.split('/')[-1].split('.')[0]
        image = read_image(img_path)
        testing_data.append([np.array(image), img_num])      
    return testing_data # List of lists of images and there id's

test_data = making_test_data()
with open('submission_file.csv','w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv','a') as f:
    # Predicting image by image
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = (data[0] / 255)
        data = img_data.reshape(1, ROWS, COLS, 3)
        out = best_model.predict([data])[0][0]
        f.write('{},{}\n'.format(img_num,out))
# Free some space
import gc
del X_train, X_val, y_train, y_val, test_data, labels
gc.collect()
X_test, _ = prep_data(test_images)
X_test = np.array(X_test)
X_test /= 255.
predictions = best_model.predict(X_test)
for i in range(0,10):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

    plt.imshow(image.array_to_img(X_test[i]))
    plt.show()
with open('dogs-v-cat-results.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['id', 'label'])
    for i, path in enumerate(test_images):
        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        writer.writerow([name, predictions[i, 0]])

