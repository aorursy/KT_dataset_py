# used to change filepaths

import os



import matplotlib as mpl

import matplotlib.pyplot as plt

from IPython.display import display

%matplotlib inline



import pandas as pd

import numpy as np



# import Image from PIL

from PIL import Image



from skimage.feature import hog

from skimage.color import rgb2grey



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



# import train_test_split from sklearn's model selection module

from sklearn.model_selection import train_test_split 



# import SVC from sklearn's svm module

from sklearn.svm import SVC



# import accuracy_score from sklearn's metrics module

from sklearn.metrics import roc_curve, auc, accuracy_score



import seaborn as sns

# sns.set(style="darkgrid")



# import keras library

import keras



# import Sequential from the keras models module

from keras.models import Sequential



# import Dense, Dropout, Flatten, Conv2D, MaxPooling2D from the keras layers module

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
DF_PATH = '/kaggle/input/goodreads-best-books/'

IMAGES_PATH = '/kaggle/input/goodreads-best-books/images/images/'

IMAGE_WIDTH = 100

IMAGE_HEIGHT = 100

IMAGE_CHANNELS = 3



TEST_SIZE = 0.2

VAL_SIZE = 0.2

MAX_POOL_DIM = 2

KERNEL_SIZE = 3



# How to get reproducible results in keras-StackOverflow

# Seed value

# Apparently you may use different seed values at each stage

RANDOM_STATE = 22



# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

import os

os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)



# 2. Set the `python` built-in pseudo-random generator at a fixed value

import random

random.seed(RANDOM_STATE)



# 3. Set the `numpy` pseudo-random generator at a fixed value

import numpy as np

np.random.seed(RANDOM_STATE)



# 4. Set the `tensorflow` pseudo-random generator at a fixed value

import tensorflow as tf

tf.set_random_seed(RANDOM_STATE)
def get_image(file_name, 

              root = IMAGES_PATH, 

              extension = '.jpg', 

              resize = False,

              reduce_channels = False):

    """

    Converts an image number into the file path where the image is located, 

    opens the image, and returns the image as a numpy array.

    """

#     filename = "{}.jpg".format(row_id)

    file_path = os.path.join(root, str(file_name) + extension)

    img = Image.open(file_path)

    if resize == True:

        img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT))

#     img = skimage.transform.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), 

#                                      mode='reflect')

    if reduce_channels == True:

        return np.array(img)[:,:,:IMAGE_CHANNELS]

    else:

        return np.array(img)
books_df = pd.read_csv(DF_PATH + 'book_data.csv')

# image filename is connected to initial index in dataframe 

books_df['img_name'] = books_df.index

books_df.info()
books_df.head()
image_files = list(os.listdir(IMAGES_PATH))

print("Number of image files: {}".format(len(image_files)))
books_df['image_url'].notnull().value_counts()
books_df = books_df[books_df['image_url'].notnull()].reset_index(drop = True)

books_df.info()
books_df.head()
random_check = books_df.sample(1, random_state = RANDOM_STATE).iloc[0]['img_name']

books_df.sample(1, random_state = RANDOM_STATE)
# Open the image

img = Image.open(IMAGES_PATH + str(random_check) + '.jpg') 



# Get the image size

print("The image size is: {}".format(img.size))



# Just having the image as the last line in the cell will display it in the notebook

img
file_names = list(books_df['img_name'].apply(lambda row: str(row) + '.jpg'))

print("Matching image names: {}".format(len(set(file_names).intersection(image_files))))
books_df['book_rating'].apply(lambda row: round(row, 0)).astype('float64').value_counts()
books_df['book_rating_round'] = 0

G1 = 3.6 # ~ 1

G2 = 3.9 # ~ 2

G3 = 4.2 # ~ 3

G4 = 4.5 # ~ 4

G5 = 4.8 # ~ 5

G6 = 5 # ~ 6



books_df['book_rating_round'][(books_df['book_rating'] >= 0) & 

                              (books_df['book_rating'] <= G1)] = G1

books_df['book_rating_round'][(books_df['book_rating'] > G1) & 

                              (books_df['book_rating'] <= G2)] = G2

books_df['book_rating_round'][(books_df['book_rating'] > G2) & 

                              (books_df['book_rating'] <= G3)] = G3

books_df['book_rating_round'][(books_df['book_rating'] > G3) & 

                              (books_df['book_rating'] <= G4)] = G4

books_df['book_rating_round'][(books_df['book_rating'] > G4) & 

                              (books_df['book_rating'] <= G5)] = G5

books_df['book_rating_round'][(books_df['book_rating'] > G5) & 

                              (books_df['book_rating'] <= G6)] = G6
books_df[['book_rating', 'book_rating_round']]
books_df['book_rating_round'].value_counts().sort_index()
ax = sns.countplot(x = books_df['book_rating_round'])

plt.tight_layout()

plt.show()
# There is an error with couple images. Removing from analysis

img_error_list = ['2833', '17637', '33060', '44763', '49439']

books_sample_df = books_df[-books_df['img_name'].isin(img_error_list)]
# Sample when building the model 

# books_sample_df = books_sample_df.sample(n = 10000, 

#                                          random_state = RANDOM_STATE)
books_sample_df.shape
width = []

height = []

channels = []

for i in books_sample_df['img_name']: 

    img = get_image(i).shape

    height.append(img[0])

    width.append(img[1])

    if (len(img) < 3):

        channels.append(0)

    else:

        channels.append(img[2])
books_sample_df['img_width'] = width

books_sample_df['img_height'] = height

books_sample_df['img_channels'] = channels
books_sample_df.head()
books_sample_df.boxplot(column = ['img_width', 'img_height', 'img_channels'])
books_sample_df['img_channels'].value_counts(dropna=False)
books_sample_df = books_sample_df[-(books_sample_df['img_channels'] == 0)]
books_sample_df['img_channels'].value_counts(dropna=False)
labels = books_sample_df[['img_name', 'book_rating_round']].set_index('img_name')

labels.head()
labels['book_rating_round'].value_counts().sort_index()
# Initialize standard scaler

standarization = StandardScaler()



image_list = []

for i in labels.index:

    # Load image

    img = get_image(i, resize = True, reduce_channels = True)

    # For each channel, apply standard scaler's fit_transform method

    for channel in range(img.shape[2]):

        img[:, :, channel] = standarization.fit_transform(img[:, :, channel])

        

    # Append to list of all images

    image_list.append(img)

    

# Convert image list to single array

X = np.array(image_list)



# Print shape of X matrix

print(X.shape)
labels["book_rating_round"]
y = pd.get_dummies(labels["book_rating_round"])

y.head()
# Split out test sets and intermediary sets

x_interim, x_test, y_interim, y_test = train_test_split(X,

                                           y,

                                           test_size = 0.2,

                                           random_state = RANDOM_STATE,

                                           stratify = y)



# Split remaining data into train and validation sets

x_train, x_val, y_train, y_val = train_test_split(x_interim,

                                           y_interim,

                                           test_size=0.4,

                                           random_state = RANDOM_STATE,

                                           stratify = y_interim)



# Examine number of samples in train, test, and validation sets

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_val.shape[0], 'validate samples')

print(x_test.shape[0], 'test samples')
# set model constants

num_classes = len(labels['book_rating_round'].value_counts()) 

num_classes
# Define model as Sequential

model = Sequential()



# First convolutional layer with 16 filters

model.add(Conv2D(filters = 16, 

                 kernel_size = KERNEL_SIZE, 

                 activation = 'relu', 

                 input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))



# Add a second 2D convolutional layer with 32 filters

model.add(Conv2D(filters = 32, 

                 kernel_size = KERNEL_SIZE, 

                 activation = 'relu'))



# Reduce dimensionality through max pooling

model.add(MaxPooling2D(pool_size = MAX_POOL_DIM))



# Third convolutional layer with 64 filters

model.add(Conv2D(filters = 32, 

                 kernel_size = KERNEL_SIZE, 

                 activation='relu'))



# Dropout to prevent over fitting

model.add(Dropout(rate = 0.25, 

                  seed = RANDOM_STATE))



# Add another max pooling layer

model.add(MaxPooling2D(pool_size = MAX_POOL_DIM))



# Necessary flatten step preceeding dense layer

model.add(Flatten())



# Fully connected layer

model.add(Dense(128, 

                activation='relu'))



# Additional dropout to prevent overfitting

model.add(Dropout(rate = 0.5, 

                  seed = RANDOM_STATE))



# Prediction layers

model.add(Dense(num_classes, 

                activation="softmax", # Multilabel classification

                name='Predictions'))
# show model summary

model.summary()
# Model compilation

model.compile(

    # Loss as categorical_crossentropy for multi-label classification

    loss = "categorical_crossentropy",

    # Optimizer as Stochastic Gradient Descent

    optimizer = keras.optimizers.SGD(lr = 0.001),

    # Metric as accuracy

    metrics = ['accuracy']

)
# Model training

model.fit(

    x_train,

    y_train,

    epochs = 20,

    verbose = 0,

    validation_data = (x_val, y_val)

)
# Evaluate model on validation set

val_score = model.evaluate(x_val, y_val, verbose=0)

print('Validation loss:', val_score[0])

print('Validation accuracy:', val_score[1])



print("")



# Evaluate model on holdout test set

test_score = model.evaluate(x_test, y_test, verbose=0)

# print loss score

print('Eval loss:', test_score[0])

# print accuracy score

print('Eval accuracy:', test_score[1])
# Save model history on each epoch

history = model.history.history
fig = plt.figure(1)

plt.subplot(211)

# Plot the Train accuracy

plt.plot(history['accuracy'])

plt.title('Train accuracy and loss')

plt.ylabel('Accuracy')

plt.subplot(212)

# plot the Train loss

plt.plot(history['loss'], 'r')

plt.xlabel('Epoch')

plt.ylabel('Loss value');
fig = plt.figure(1)

plt.subplot(211)

# Plot the Validation accuracy

plt.plot(history['val_accuracy'])

plt.title('Validation accuracy and loss')

plt.ylabel('Accuracy')

plt.subplot(212)

# Plot the Validation loss

plt.plot(history['val_loss'], 'r')

plt.xlabel('Epoch')

plt.ylabel('Loss value');
# Predicted probabilities for x_test

y_proba = model.predict(x_test)



print("First five probabilities:")

print(y_proba[:5])

print("")



# Predicted classes for x_test - Highest probability

y_pred = model.predict_classes(x_test)



print("First five class predictions:")

print(y_pred[:5])

print("")
# Palette must be given in sorted order

palette = [0, 1, 2, 3, 4, 5]

# Key gives the new values you wish palette to be mapped to

key = np.array([3.6, 3.9, 4.2, 4.5, 4.8, 5])

index_pred = np.digitize(y_pred.ravel(), palette, right=True)



y_pred_transformed = key[index_pred].reshape(y_pred.shape)

print(y_pred_transformed)
ax = sns.countplot(x = y_pred_transformed)

plt.tight_layout()

plt.show()
ax = sns.countplot(x = labels['book_rating_round'])

plt.tight_layout()

plt.show()
def convolution(image, kernel):

    """3x3 convolution"""

    result = np.zeros(image.shape)

    # Output array

    for ii in range(image.shape[0] - 3):

        for jj in range(image.shape[1] - 3):

            result[ii, jj] = (image[ii:ii+3, jj:jj+3] * kernel).sum()



    # Result

    return(result)
model.layers
# Third convolution layer

cnv1 = model.layers[3]
weights1 = cnv1.get_weights()

len(weights1)
kernels1 = weights1[0]

kernels1.shape
kernel1_1 = kernels1[:,:,0,0]

kernel1_1.shape
plt.imshow(kernel1_1)

plt.colorbar()
test_image = x_test[2,:, :, 0]

plt.imshow(test_image)

plt.colorbar()
# Convolve with the fourth image in test_data

out = convolution(test_image, kernel1_1)



# Visualize the result

plt.imshow(out)

plt.colorbar()

plt.show()