# Install tf 2.0

!pip install tensorflow==2.0.0
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



import tensorflow as tf



from tensorflow.keras import Model, Input

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Conv2D, MaxPool2D,Concatenate,concatenate,Lambda, Flatten, Dense, BatchNormalization, ZeroPadding2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l2



import os
tf.__version__
# Hyperparameters

batch_size = 32

epochs = 10

img_height = 128

img_width = 128

val_split = 0.1

AUTOTUNE = tf.data.experimental.AUTOTUNE

val_size = np.floor(batch_size*val_split)

num_predict_img = 10
# Load file-labels data

train_dir = '../input/humpback-whale-identification/train/'

test_dir = '../input/humpback-whale-identification/test/'

df = pd.read_csv('../input/humpback-whale-identification/train.csv')



# Get the classes with more than 10 images

df_classesCount = df.groupby('Id').count().sort_values('Image',ascending=False)

class_names = df_classesCount[df_classesCount['Image']>10]

class_names = class_names.index

num_classes = len(class_names)

print('Number of classes:',num_classes)



# Get test image file names

list_test = os.listdir(test_dir)
# Helper functions

def getDFBatch(batch_size=32):

    """Create dataframe with pairs of image files, where half same class with target 1 

    and half different classes with target 0"""

    image_pairs = [[],[]]

    targets = [1]*(batch_size//2)+[0]*(batch_size//2)

    for i in range(batch_size//2):

        # Choose one class

        target = np.random.choice(class_names,replace=False)

        # Choose two image files from the same class

        image_pair = np.random.choice(df[df['Id']==target]['Image'],2,replace=False)

        image_pairs[0].insert(0,image_pair[0])

        image_pairs[1].insert(0,image_pair[1])

        # Choose two classes

        target = np.random.choice(class_names,2,replace=False)

        # Choose two image files from different classes

        im1 = np.random.choice(df[df['Id']==target[0]]['Image'],replace=False)

        im2 = np.random.choice(df[df['Id']==target[1]]['Image'],replace=False)

        image_pairs[0].append(im1)

        image_pairs[1].append(im2)

    return pd.DataFrame({'left_image':image_pairs[0],'right_image':image_pairs[1],'targets':targets})



    

def processDS(left_file,right_file,target):

    """Process on training dataset elements to get the images and target"""

    # Read images

    left_file_path=train_dir+left_file

    right_file_path=train_dir+right_file

    img_left = tf.io.read_file(left_file_path)

    img_right = tf.io.read_file(right_file_path)

    

    # Convert to grayscale

    img_left = tf.image.decode_jpeg(img_left, channels=1)

    img_right = tf.image.decode_jpeg(img_right, channels=1)

    # Convert to floats in the [0,1] range.

    img_left = tf.image.convert_image_dtype(img_left, tf.float32)

    img_right = tf.image.convert_image_dtype(img_right, tf.float32)

    # Resize the image to the desired size.

    img_left = tf.image.resize(img_left, [img_width, img_height])

    img_right = tf.image.resize(img_right, [img_width, img_height])

    

    return [img_left,img_right],target



def processTestDS(left_file,right_file):

    """Process on testing dataset elements to get the images"""

    # Read images

    left_file_path=train_dir+left_file

    right_file_path=test_dir+right_file

    img_left = tf.io.read_file(left_file_path)

    img_right = tf.io.read_file(right_file_path)

    

    # Convert to grayscale

    img_left = tf.image.decode_jpeg(img_left, channels=1)

    img_right = tf.image.decode_jpeg(img_right, channels=1)

    # Convert to floats in the [0,1] range.

    img_left = tf.image.convert_image_dtype(img_left, tf.float32)

    img_right = tf.image.convert_image_dtype(img_right, tf.float32)

    # Resize the image to the desired size.

    img_left = tf.image.resize(img_left, [img_width, img_height])

    img_right = tf.image.resize(img_right, [img_width, img_height])

    

    images = [img_left,img_right]

    images = tf.reshape(images, (2,img_height,img_width,1))

    

    return images



def getBatch(batch_size=32,df_img_files=None):

    """Create a dataset for training/testing data"""

    df_batch = getDFBatch(batch_size=batch_size)

    

    if df_img_files is None:

        ds = tf.data.Dataset.from_tensor_slices((df_batch['left_image'].values,df_batch['right_image'].values,\

                                             df_batch['targets'].values))

        labeled_ds = ds.map(processDS, num_parallel_calls=AUTOTUNE)

        return labeled_ds

    else:

        ds = tf.data.Dataset.from_tensor_slices((df_img_files['left_image'].values,\

                                                 df_img_files['right_image'].values))

        labeled_ds = ds.map(processTestDS, num_parallel_calls=AUTOTUNE)

        return labeled_ds

# Helper functions

def siameseModel(input_shape):

    # Define the tensors for the two input images

    inputs = Input(([2]+input_shape))

    left_input = inputs[:,0,:,:,:]

    right_input = inputs[:,1,:,:,:]

    

    # Convolutional Neural Network

    model = Sequential()

    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (7,7), activation='relu', kernel_regularizer=l2(2e-4)))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (4,4), activation='relu', kernel_regularizer=l2(2e-4)))

    model.add(MaxPool2D())

    model.add(Conv2D(256, (4,4), activation='relu', kernel_regularizer=l2(2e-4)))

    model.add(Flatten())

    model.add(Dense(num_classes, activation='sigmoid', kernel_regularizer=l2(1e-3)))

    

    # Generate the encodings (feature vectors) for the two images

    encoded_l = model(left_input)

    encoded_r = model(right_input)

    

    # Add a customized layer to compute the absolute difference between the encodings

    L1_layer = Lambda(lambda tensors:tf.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([encoded_l, encoded_r])

    

    # Add a dense layer with a sigmoid unit to generate the similarity score

    prediction = Dense(1,activation='sigmoid')(L1_distance)

    

    # Connect the inputs with the outputs

    siamese_net = Model(inputs=inputs,outputs=prediction)

      

    # Compile model

    siamese_net.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    siamese_net.summary()

    

    # return the model

    return siamese_net



# Build model

model=siameseModel([img_height,img_width,1])
# Train model

history = []

for i in range(epochs):

    train_batch = getBatch(batch_size)

    train_batch = train_batch.shuffle(batch_size)

    # create validation dataset

    valid_ds = train_batch.take(val_size)

    valid_ds = valid_ds.batch(val_size)

    # create training dataset

    train_ds = train_batch.skip(val_size)

    train_ds = train_ds.batch(batch_size-val_size)

    

    train_hist = model.train_on_batch(train_ds)

    val_hist = model.test_on_batch(valid_ds)

    history.append([train_hist, val_hist])

    

    if i%2==0:

        print('Epoch {} ------------'.format(i))

        print('Training loss and accuracy')

        print(train_hist)

        print('Validation loss and accuracy')

        print(val_hist)

    
# Predict



# prepare test data

idx = np.random.randint(len(df), size=num_predict_img)

new_df = df.iloc[idx]

test_classes = new_df['Id']

new_df = new_df.drop(columns='Id')

new_df['right_image']=list_test[0]

new_df.rename(columns={'Image': 'left_image'}, inplace=True)

test_batch=getBatch(df_img_files=new_df)

test_batch=test_batch.batch(num_predict_img)



# Get the scores

r=model.predict(test_batch)

print(r)



# Get the best class matching

best_idx = np.argmax(r.reshape(-1))

print('Best match is: ', test_classes.iloc[best_idx])
# Visualize the best matching

best_img_file = new_df['left_image'].iloc[best_idx]

plt.subplot(1,2,1)

img = mpimg.imread(test_dir+list_test[0])

plt.imshow(img)

plt.title('Original')

plt.subplot(1,2,2)

img = mpimg.imread(train_dir+best_img_file)

plt.imshow(img)

plt.title('Matching')

plt.show()