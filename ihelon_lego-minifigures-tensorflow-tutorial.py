import os

import math

import random



import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import seaborn as sn

import albumentations as A

import tensorflow as tf

from tensorflow.keras.applications import mobilenet_v2 as tf_mobilenet_v2

from tensorflow.keras import layers as tf_layers

from tensorflow.keras import models as tf_models

from tensorflow.keras import callbacks as tf_callbacks

from sklearn import metrics as sk_metrics

from sklearn import model_selection as sk_model_selection
# The directory to the dataset

BASE_DIR = '../input/lego-minifigures-classification/'

PATH_INDEX = os.path.join(BASE_DIR, "index.csv")

PATH_TEST = os.path.join(BASE_DIR, "test.csv")

PATH_METADATA = os.path.join(BASE_DIR, "metadata.csv")
# Try to set random seet that our experiment repeated between (We have some problem to set seed with GPU)

def set_seed(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    tf.random.set_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    os.environ['TF_DETERMINISTIC_OPS'] = 'true'

    



SEED = 42

set_seed(SEED)
# Read information about dataset

df = pd.read_csv(PATH_INDEX)



tmp_train, tmp_valid = sk_model_selection.train_test_split(

    df, test_size=0.3, random_state=SEED, stratify=df['class_id']

)



# Get train file paths

train_paths = tmp_train['path'].values

# Get train labels

train_targets = tmp_train['class_id'].values

# Create full train paths (base dir + concrete file)

train_paths = list(map(lambda x: os.path.join(BASE_DIR, x), train_paths))



# Get valid file paths

valid_paths = tmp_valid['path'].values

# Get valid labels

valid_targets = tmp_valid['class_id'].values

# Create full valid paths (base dir + concrete file)

valid_paths = list(map(lambda x: os.path.join(BASE_DIR, x), valid_paths))



df_test = pd.read_csv(PATH_TEST)

test_paths = df_test['path'].values

test_paths = list(map(lambda x: os.path.join(BASE_DIR, x), test_paths))

test_targets = df_test['class_id'].values
# Total number of classes in the dataset

df_metadata = pd.read_csv(PATH_METADATA)

n_classes = df_metadata.shape[0]

print('Number of classes: ', n_classes)
# DataGenerator allows you not to load the entire dataset to memory at once, but to do it in batches   

# Each time we have only one batch of pictures in memory



class DataGenerator(tf.keras.utils.Sequence):

    def __init__(

        self, 

        paths, 

        targets, 

        image_size=(224, 224), 

        batch_size=64, 

        shuffle=True, 

        transforms=None

    ):

        # the list of paths to files

        self.paths = paths

        # the list with the true labels of each file

        self.targets = targets

        # images size

        self.image_size = image_size

        # batch size (the number of images)

        self.batch_size = batch_size

        # if we need to shuffle order of files

        # for validation we don't need to shuffle, for training - do

        self.shuffle = shuffle

        # Augmentations for our images. It is implemented with albumentations library

        self.transforms = transforms

        # Preprocess function for the pretrained model. 

        # CHANGE IT IF USING OTHER THAN MOBILENETV2 MODEL

        self.preprocess = tf_mobilenet_v2.preprocess_input

        

        # Call function to create and shuffle (if needed) indices of files

        self.on_epoch_end()

        

    def on_epoch_end(self):

        # This function is called at the end of each epoch while training

        

        # Create as many indices as many files we have

        self.indexes = np.arange(len(self.paths))

        # Shuffle them if needed

        if self.shuffle:

            np.random.shuffle(self.indexes)

            

    def __len__(self):

        # We need that this function returns the number of steps in one epoch

        

        # How many batches we have

        return len(self.paths) // self.batch_size

    

    

    def __getitem__(self, index):

        # This function returns batch of pictures with their labels

        

        # Take in order as many indices as our batch size is

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        

        # Take image file paths that are included in that batch

        batch_paths = [self.paths[k] for k in indexes]

        # Take labels for each image

        batch_y = [self.targets[k] - 1 for k in indexes]

        batch_X = []

        for i in range(self.batch_size):

            # Read the image

            img = cv2.imread(batch_paths[i])

            # Resize it to needed shape

            img = cv2.resize(img, self.image_size)

            # Convert image colors from BGR to RGB

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply transforms (see albumentations library)

            if self.transforms:

                img = self.transforms(image=img)['image']

            # Normalize image

#             img = img / 255.

            img = self.preprocess(img)

            

            batch_X.append(img)

            

        return np.array(batch_X), np.array(batch_y)

# albumentations transformations for training data. We don't need this transformations for the validation



def get_train_transforms():

    return A.Compose(

        [

            A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=0.5),

            A.Cutout(num_holes=8, max_h_size=25, max_w_size=25, fill_value=0, p=0.25),

            A.Cutout(num_holes=8, max_h_size=25, max_w_size=25, fill_value=255, p=0.25),

            A.HorizontalFlip(p=0.5),

            A.RandomContrast(limit=(-0.3, 0.3), p=0.5),

            A.RandomBrightness(limit=(-0.4, 0.4), p=0.5),

            A.Blur(p=0.25),

        ], 

        p=1.0

    )
IMAGE_SIZE = (512, 512)

# We use not big batch size to train

TRAIN_BATCH_SIZE = 4

# If we use batch size != 1, we can miss some images if len(valid) % batch_size != 0

VALID_BATCH_SIZE = 1 



# Initialize the train data generator

train_generator = DataGenerator(

    train_paths, 

    train_targets, 

    batch_size=TRAIN_BATCH_SIZE, 

    image_size=IMAGE_SIZE,

    shuffle=True, 

    transforms=get_train_transforms()

)



# Initialize the valid data generator

valid_generator = DataGenerator(

    valid_paths, 

    valid_targets, 

    image_size=IMAGE_SIZE,

    batch_size=VALID_BATCH_SIZE, 

    shuffle=False,

)
def denormalize_image(image):

    return ((image + 1) * 127.5).astype(int)



# Let's visualize some batches of the train data

plt.figure(figsize=(10, 10))

for i_batch in range(4):

    images, labels = train_generator[i_batch]

    for i in range(4):

        plt.subplot(4, 4, 4 * i_batch + i + 1)

        plt.imshow(denormalize_image(images[i]))

        plt.title(labels[i])

        plt.axis('off')
# Let's visualize some batches of the valid data

plt.figure(figsize=(10, 10))

for i_batch in range(16):

    images, labels = valid_generator[i_batch]

    plt.subplot(4, 4, i_batch + 1)

    plt.imshow(denormalize_image(images[0]))

    plt.title(labels[0])

    plt.axis('off')
def create_model(n_classes):

    # We take pretrained MobileNetV2 (see Keras docs)

    base_model = tf_mobilenet_v2.MobileNetV2()

    x = base_model.layers[-2].output

    # Take penultimate layer of the MobileNetV2 model and connect this layer with Dropout

    x = tf_layers.Dropout(.5)(x)

    # Add additional Dense layer, with number of neurons as number of our classes

    # Use softmax activation because we have one class classification problem

    outputs = tf_layers.Dense(n_classes, activation='softmax')(x)

    # Create model using MobileNetV2 input and our created output

    model = tf_models.Model(base_model.inputs, outputs)





    # Compile model using Adam optimizer and categorical crossentropy loss

    model.compile(

        optimizer=tf.keras.optimizers.Adam(0.0001),

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']

    )

    

    return model





model = create_model(n_classes)
# checkpoint to saving the best model by validation loss

callback_save = tf_callbacks.ModelCheckpoint(

    'best.hdf5',

    monitor="val_loss",

    save_best_only=True,

    mode="min",

)



# checkpoint to stop training if model didn't improve valid loss for 3 epochs

callback_early_stopping = tf_callbacks.EarlyStopping(

    monitor="val_loss",

    patience=3,

)
EPOCHS = 50



# Train model using data generators

history = model.fit(

    train_generator,

    validation_data=valid_generator,

    epochs=EPOCHS,

    callbacks=[

        callback_save, 

        callback_early_stopping

    ],

    verbose=0,

)
# Visualize train and valid loss 

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)

plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='valid loss')

plt.xticks(fontsize=14)

plt.xlabel("Epoch number", fontsize=15)

plt.yticks(fontsize=14)

plt.ylabel("Loss value", fontsize=15)

plt.legend(fontsize=15)

plt.grid()



# Visualize train and valid accyracy 

plt.subplot(1, 2, 2)

plt.plot(history.history['accuracy'], label='train acc')

plt.plot(history.history['val_accuracy'], label='valid acc')

plt.xticks(fontsize=14)

plt.xlabel("Epoch number", fontsize=15)

plt.yticks(fontsize=14)

plt.ylabel("Accuracy score", fontsize=15)

plt.legend(fontsize=15)

plt.grid();
# Load the best model (we create for checkpoint to save the best model)

model = tf_models.load_model('best.hdf5')
TEST_BATCH_SIZE = 1



test_generator = DataGenerator(

    test_paths, 

    test_targets, 

    image_size=IMAGE_SIZE,

    batch_size=TEST_BATCH_SIZE, 

    shuffle=False,

)
# Save the model predictions and true labels

y_pred = []

y_test = []

for _X_test, _y_test in test_generator:

    y_pred.extend(model.predict(_X_test).argmax(axis=-1))

    y_test.extend(_y_test)



# Calculate needed metrics

print(f'Accuracy score on test data:  {sk_metrics.accuracy_score(y_test, y_pred)}')

print(f'Macro F1 score on test data:  {sk_metrics.f1_score(y_test, y_pred, average="macro")}')
# Load metadata to get classes people-friendly names

labels = df_metadata['minifigure_name'].tolist()



# Calculate confusion matrix

confusion_matrix = sk_metrics.confusion_matrix(y_test, y_pred)

# confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)

df_confusion_matrix = pd.DataFrame(confusion_matrix, index=labels, columns=labels)



# Show confusion matrix

plt.figure(figsize=(12, 12))

sn.heatmap(df_confusion_matrix, annot=True, cbar=False, cmap='Oranges', linewidths=1, linecolor='black')

plt.xlabel('Predicted labels', fontsize=15)

plt.xticks(fontsize=12)

plt.ylabel('True labels', fontsize=15)

plt.yticks(fontsize=12);
# Save image, label, prediction for false predictions 

error_images = []

error_label = []

error_pred = []

error_prob = []

for _X_test, _y_test in test_generator:

    pred = model.predict(_X_test).argmax(axis=-1)

    if pred[0] != _y_test:

        error_images.extend(_X_test)

        error_label.extend(_y_test)

        error_pred.extend(pred)

        error_prob.extend(model.predict(_X_test).max(axis=-1))
# Visualize missclassified samples

plt.figure(figsize=(16, 16))

w_size = int(len(error_images) ** 0.5)

h_size = math.ceil(len(error_images) / w_size)

for ind, image in enumerate(error_images):

    plt.subplot(h_size, w_size, ind + 1)

    plt.imshow(denormalize_image(image))

    pred_label = labels[error_pred[ind]]

    pred_prob = error_prob[ind]

    true_label = labels[error_label[ind]]

    plt.title(f'predict: {pred_label} ({pred_prob:.2f}) true: {true_label}')

    plt.axis('off')