import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
import shutil

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D,Activation, BatchNormalization
from tensorflow.keras.layers import Input, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
tf.random.set_seed(42)
image_dir = '../input/zndmasks/images'
train = pd.read_csv('../input/zndmasks/train_labels.csv')
sub = pd.read_csv('../input/zndmasks/sample_sub_v2.csv')
train
sub
def create_dir(names: [str], path: str):
    """
    Creates the directories passed in the names arg list
    in the specified path within the current working directory.
    Args:
        `names`: A list of directory names to be created.
        `path`: The path where to create the directories.
    """
    for dir_name in names:
        new_dir = os.path.join(path, dir_name)
        img_path = os.path.join(new_dir)
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        else:
            print(f'{img_path}: Exists!') 


def move_images(image_list: [str], source_dir: str, dest_dir: str):
    """
    Move images in image_list from source path to dest path.
    Args:
        `image_list`: A list of image files to be moved between directories.
        `source_dir`: The source/current directory holding the files.
        `dest_dir`: The new directory where the files will be transfered.
    """
    for img in image_list:
        shutil.move(source_dir+img, dest_dir)
dirs = ['train', 'test']
#create_dir(dirs, '/kaggle/working/')
create_dir(dirs, '/tmp/')
img_names = os.listdir('../input/zndmasks/images')
train_img_names = train.image.tolist()
test_img_names = []

for img in img_names:
    if img not in train_img_names:
        test_img_names.append(img)
        
train_img_names
test_img_names
print(len(train_img_names))
print(len(test_img_names))
from distutils.dir_util import copy_tree

fromDirectory='../input/zndmasks/images'
toDirectory='/tmp/images'

copy_tree(fromDirectory,toDirectory)
# Move train images to the train dir
move_images(train_img_names, '/tmp/images/', '/tmp/train/')

# Move test images to the test dir
move_images(test_img_names, '/tmp/images/', '/tmp/test/')
# Plotting the image class distributions in the train set
train['target'].value_counts().plot.bar()
plt.title('Image class distributions')
plt.xlabel('Classes (0: No mask) (1: Mask)')
plt.ylabel('Number of images in train set')
# Further splitting the train images into train and validation images.
train['target'] = train['target'].replace({0: 'No_mask', 1: 'Mask'})

tr_data, val_data = train_test_split(train, test_size=0.20, random_state=42)
tr_data = tr_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
tr_data
total_train = tr_data.shape[0]  #total train images
total_validate = val_data.shape[0] #total images on validation set
batch_size = 8
# Train generator
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    tr_data, 
    '/tmp/train/', 
    x_col = 'image',
    y_col = 'target',
    target_size = IMAGE_SIZE,
    class_mode = 'categorical',
    batch_size = batch_size
)

# Validation generator
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    val_data, 
    '/tmp/train/', 
    x_col='image',
    y_col='target',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
plt.figure(figsize=(12, 12))
for i in range(0, 8):
    plt.subplot(2, 4, i+1)
    for X_batch, Y_batch in train_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                                               include_top=False,
                                               weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
x =  Dense(1024,activation='relu')(x) # complex for better results.
x = Dropout(0.8)(x) 
x = Dense(1024,activation='relu')(x) 
x = Dropout(0.8)(x) 
x = Dense(512,activation='relu')(x) 
x = Dropout(0.8)(x) 
preds = Dense(2,activation='softmax')(x)

model = Model(inputs=base_model.input,outputs=preds) #specify the inputs and outputs

model.summary()
earlystop = EarlyStopping(patience=5) # Stop if validation loss doesn't improve after 5 epochs

# Gradually reduce the learning rate if validation loss doesn't improve after 5 steps
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001
                                           )
modelcheckpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)

callbacks = [earlystop, modelcheckpoint,learning_rate_reduction]

base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Training the model
epochs = 10 if FAST_RUN else 200

history = model.fit_generator(
    train_generator, 
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = total_validate//batch_size,
    steps_per_epoch = total_train//batch_size,
    callbacks = callbacks
)
model.save_weights("mobilenet.h5")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
test_filenames = os.listdir('/tmp/test/')
test_df = pd.DataFrame({
    'image': test_filenames
})
nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    '/tmp/test/', 
    x_col='image',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
predictions = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
from keras.models import load_model
saved_model = load_model('best_model.h5')
pred_new = saved_model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
pred_probabilities = pred_new.max(1)
test_df['target'] = pred_probabilities
sub2 = test_df.copy()
sub2.to_csv('submission2.csv', index=False)
predicted_probabilities = predictions.max(1)
test_df['target'] = predicted_probabilities
test_df
sub = test_df.copy()
sub.to_csv('submission.csv', index=False)