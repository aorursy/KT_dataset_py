import cv2

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
# Inspect the input folder

!cd ../input/ ;ls
# View the number of data

base_dir = '/kaggle/input'

train_image_dir = os.path.join(base_dir, 'train_images')

test_image_dir = os.path.join(base_dir, 'test_images')

train_list = os.listdir(train_image_dir)

test_list = os.listdir(test_image_dir)

print('Number of training images:', len(train_list))

print('Number of test images:', len(test_list))



# See some names

print(train_list[:5])
# Read in training labels and inspect

label_all = pd.read_csv(os.path.join(base_dir,'train.csv'))

label_all.head()
# Inspect 0002cc93b.jpg

sample_image_name = '0002cc93b.jpg'

sample_image = plt.imread(os.path.join(train_image_dir, sample_image_name))

print(sample_image.shape)

plt.imshow(sample_image)
# Fill nan with empty string, which is prepared for decoding the run-length encoding(rle)

label_all.EncodedPixels.fillna('', inplace=True)

label_all.head()
# Seperate colunms of ImageId and ClassId, in order to combine four EncodedPixels of the same image

new_df = label_all.ImageId_ClassId.str.split('_', expand=True)

label_all['ImageId'] = new_df[0]

label_all['ClassId'] = new_df[1]

label_all.head()
# Group label all labels for each image together to form a dictionary {Id : list of 4 EncodedPixels }

labels = {}

# for imageId in label_all.ImageId[0:][::4]:

#     labels[imageId] = label_all.EncodedPixels[label_all.ImageId==imageId].to_numpy()



'THe above is too slow...'



for i in range(0,len(label_all.ImageId),4):

    labels[label_all.ImageId[i]] = list(label_all.EncodedPixels[i:i+4])
# Decode the run-length encoding 

def decode_rle(encodes: list, shape=sample_image.shape[0:2]):

    """

    encodes : a list of 4 run-length encodes of a target image ID. 

    Return decoded mask of 4 classed defects in an array of shape 4, x, y.

    """

    x, y = shape

    decoded_images = np.empty((4, x, y))

    

    for j in range(4):

        decoded = np.zeros((x*y))

        

        encode = encodes[j]

        if encode:

            split_num = encode.split()

            for i in range(0, len(split_num), 2):

                start_pixel = int(split_num[i])

                run_length = int(split_num[i+1])

                decoded[start_pixel-1 : start_pixel-1+run_length] = 1

        decoded_image = decoded.reshape(y, x).T

        

        decoded_images[j] = decoded_image

        

    return decoded_images



# Test with sample image

decoded_sample_image = decode_rle(labels[sample_image_name])

plt.imshow(decoded_sample_image[0], cmap='gray')

print(decoded_sample_image.shape)
# Encode rle from masks

def encode_rle(masks):

    """

    mask: shape is (classes, height, width)

    Return a list of 4 rle for a same image.

    """

    classes, height, width = masks.shape

    encoded_rle = ['', '', '', '']

    

    for i in range(classes):

        longer = masks[i,].T.flatten()

        longer = np.concatenate(([0], longer, [0]))

        condition = (longer[1:] != longer[:-1])

        pixels_that_changes = np.where(condition)[0] + 1 

        # this return the starts and ends point of the offset difference

        # [0] is to reduce the array dimension

        # + 1 is to change from index to pixel number 

        

        # Change even bit into the difference between adjacent odd and even, which is the length

        pixels_that_changes[1::2] -= pixels_that_changes[:-1:2]

        

        encoded_rle[i] = " ".join(str(x) for x in pixels_that_changes)

    return encoded_rle



# Test

print(encode_rle(decoded_sample_image))
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = tf.keras.backend.flatten(y_true)

    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = tf.keras.backend.flatten(y_true)

    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * tf.keras.backend.sum(intersection) + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
# Design model: U-net

from tensorflow.keras import Model

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

from tensorflow.keras.losses import binary_crossentropy



def build_model(input_shape):

    inputs = Input(input_shape)



    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (inputs)

    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (c1)

    p1 = MaxPooling2D((2, 2)) (c1)



    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (p1)

    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (p2)

    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (c3)

    p3 = MaxPooling2D((2, 2)) (c3)



    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (p3)

    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (p4)

    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (c5)

    p5 = MaxPooling2D(pool_size=(2, 2)) (c5)



    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (p5)

    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (c55)



    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)

    u6 = concatenate([u6, c5])

    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (u6)

    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (c6)



    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)

    u71 = concatenate([u71, c4])

    c71 = Conv2D(32, (3, 3), activation='elu', padding='same') (u71)

    c61 = Conv2D(32, (3, 3), activation='elu', padding='same') (c71)



    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (u7)

    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (c7)



    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (u8)

    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (c8)



    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (u9)

    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (c9)



    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)



    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

    

    return model
# Instantiate model 

model = build_model((256, 1600, 1))

model.summary()
class DataGenerator(tf.keras.utils.Sequence):

    

    def __init__(self, list_IDs, labels=None, batch_size=32, dim=(256, 1600), n_channels=3, n_classes=4,

                 shuffle=True, data_dir=train_image_dir, train=False):

        'Initialization'

        self.list_IDs = list_IDs

        self.labels = labels

        self.batch_size = batch_size

        self.dim = dim

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.data_dir = data_dir

        self.train = train

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X = self.__data_generation_X(list_IDs_temp)

        

        if self.train:

            y = self.__data_generation_y(list_IDs_temp)

            return X, y

        else:

            return X



        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation_X(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            img = cv2.imread(os.path.join(self.data_dir, ID), cv2.IMREAD_GRAYSCALE)

            img = img.astype(np.float32)

            img = np.expand_dims(img, axis=-1)

            # Normalise image 

            X[i,] = img / 255

        return X

    

    def __data_generation_y(self, list_IDs_temp):

        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_temp):

            # Store class

            y[i,] = decode_rle(self.labels[ID]).transpose(1,2,0)

            # Transpose 4, x, y to x, y, 4

        return y        
# List train and validation data, preparing for DataGenerator

train_ID = train_list[:10000]

valid_ID = train_list[10000:]



# Generate data

params = {'labels': labels,

          'dim': (256, 1600),

          'batch_size': 16,

          'n_channels': 1,

          'n_classes': 4,

          'shuffle': True,

          'data_dir': train_image_dir,

          'train': True}



# Generators

training_generator = DataGenerator(train_ID, **params)

validation_generator = DataGenerator(valid_ID, **params)
from tensorflow.keras.callbacks import ModelCheckpoint

# Set checkpoint

checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_dice_coef', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



# Train model on dataset

history = model.fit_generator(generator=training_generator,

                      validation_data=validation_generator,

                      validation_steps=int(len(valid_ID) / 16),  # add this to prevent first epoch freeze at the last step

                      callbacks=[checkpoint],

                      use_multiprocessing=False,

                      workers=1,

                      epochs=5)
# Get the test list from the submission file

submission_df = pd.read_csv('../input/sample_submission.csv')

submission_df['ImageId'] = submission_df.ImageId_ClassId.apply(lambda x: x.split('_')[0])

test_list = submission_df.ImageId.unique()

test_list
# Predict: send prediction data in minibath and save all results in a df [imageId, predicted_rle]



model.load_weights('model.h5')



predict_params = {'dim': (256, 1600),

                  'batch_size': 1,

                  'n_channels': 1,

                  'n_classes': 4,

                  'shuffle': False,

                  'data_dir': test_image_dir}

test_batch_size = 500



result_df = []



# Loop over serveral batches to save RAM

for batch_index in range(0, len(test_list), test_batch_size):

    image_Ids = test_list[batch_index : min(len(test_list), batch_index+test_batch_size)]

    

    # Loop within one batch to read in test images

    test_generator = DataGenerator(image_Ids, **predict_params)



    # Make prediction using the batch

    test_results = model.predict_generator(test_generator, workers=1, use_multiprocessing=False, verbose=1)



    # Loop over the result to extract

    for ID, prediction in zip(image_Ids, test_results):

        current_ID_df = submission_df[submission_df.ImageId == ID].copy()



        prediction = prediction.round().astype(int)

        predict_rle = encode_rle(prediction.transpose(2,0,1))



        current_ID_df['EncodedPixels'] = predict_rle

        result_df.append(current_ID_df) 
# Use submission file imageId_classId to search for result to generate output

result_df = pd.concat(result_df)

result_df.drop(columns='ImageId', inplace=True)

result_df.to_csv('submission.csv', index=False)

    
result_df.head(15)