import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





from sklearn.model_selection import train_test_split



import random

from random import randint

import re



from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical #convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout,BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping



from tensorflow import keras

from tensorflow.keras import layers









#%tensorflow_version 2.x

import tensorflow as tf



device_name = tf.test.gpu_device_name()

if "GPU" not in device_name:

    print("GPU device not found")

print('Found GPU at: {}'.format(device_name))



print("GPU", "available (YESS!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")
# creat data dirctories, we my use it later



DATA_DIR = '../input/dog-breed-identification'





TRAIN_DIR = DATA_DIR + '/train'                           

TEST_DIR = DATA_DIR + '/test'                             



TRAIN_CSV = DATA_DIR + '/labels.csv'                     

TEST_CSV = DATA_DIR + '/sample_submission.csv' 
# Checkout the labels of our data

import pandas as pd

labels_df= pd.read_csv(TRAIN_CSV)

labels_df.head()
labels_df.describe()
labels = labels_df["breed"].to_numpy() # convert labels column to NumPy array

labels[:15] 
num_images = len(labels_df["id"])

print('Number of images in Training file:', num_images)

no_labels=len(labels)

print('Number of dog breeds in Training file:', no_labels)
# Make bar chart



bar = labels_df["breed"].value_counts(ascending=True).plot.barh(figsize = (30,120))

plt.title("Distribution of the Dog Breeds", fontsize = 20)

bar.tick_params(labelsize=20)

plt.show()
# Create pathnames from image ID's

filenames = [TRAIN_DIR + "/" + fname + ".jpg" for fname in labels_df["id"]]

filenames[:10]
# display some dogs with their labels

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25),

                          subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):

    ax.imshow(plt.imread(filenames[i]))

    ax.set_title(labels_df.breed[i])

plt.tight_layout()

plt.show()

# check image size

from matplotlib.pyplot import imread

image = imread(filenames[42]) # read in an image

image.shape
# See if number of labels matches the number of filenames

if len(labels) == len(filenames):

  print("Number of labels matches number of filenames!")

else:

  print("Number of labels does not match number of filenames, check data directories.")
# Find the unique label values

unique_breeds = np.unique(labels)

len(unique_breeds)
one_hot_labels = [label == np.array(unique_breeds) for label in labels]

one_hot_labels[0] 
# Setup X & y variables

X = filenames

y = one_hot_labels
# Define image size

IMAGE_SIZE = 331







def process_image(image_path):

  """

  Takes an image file path and turns it into a Tensor.

  """

  # Read in image file

  image = tf.io.read_file(image_path)

  

  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)

  image = tf.image.decode_jpeg(image, channels=3)

  



  # Convert the colour channel values from 0-225 values to 0-1 values

  image = tf.image.convert_image_dtype(image, tf.float32)

  

  # Resize the image to our desired size 

  image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])



  return image

#Display one dog

#plt.imshow(process_image(filenames[0]))

#plt.title(labels_df.breed[0])



one_image=process_image(filenames[0])

one_image

#print(one_image.shape)
#diplay dogs after processing



fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25),

                        subplot_kw={'xticks': [], 'yticks': []})



for i, ax in enumerate(axes.flat):

    ax.imshow(process_image(filenames[i]))

    ax.set_title(labels_df.breed[i])

plt.tight_layout()

plt.show()





def data_agumentation(image=process_image):

    image=tf.image.random_flip_up_down(image)

    image=tf.image.random_flip_left_right(image)

    return image









#not used..low accuraccy
# Split them into training and validation using NUM_IMAGES 

X_train, X_val, y_train, y_val = train_test_split(X[:1000],

                                                  y[:1000], 

                                                  test_size=0.15,

                                                  random_state=42)



len(X_train), len(y_train), len(X_val), len(y_val)
# Create a simple function to return a tuple (image, label)

def get_image_label(image_path, label):

  """

  Takes an image file path name and the associated label,

  processes the image and returns a tuple of (image, label).

  """

  image = process_image(image_path)

  return image, label
# Define the batch size, 32 is a good default

BATCH_SIZE = 32



# Create a function to turn data into batches

def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):

  """

  Creates batches of data out of image (x) and label (y) pairs.

  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.

  Also accepts test data as input (no labels).

  """

  # If the data is a test dataset, we probably don't have labels

  if test_data:

    print("Creating test data batches...")

    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths

    data_batch = data.map(process_image).batch(batch_size)

    return data_batch

  

  # If the data if a valid dataset, we don't need to shuffle it

  elif valid_data:

    print("Creating validation data batches...")

    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths

                                               tf.constant(y))) # labels

    data_batch = data.map(get_image_label).batch(batch_size)

    return data_batch



  else:

    # If the data is a training dataset, we shuffle it

    print("Creating training data batches...")

    # Turn filepaths and labels into Tensors

    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths

                                              tf.constant(y))) # labels

    

    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images

    #data = data.shuffle(buffer_size=len(x))



    # Create (image, label) tuples (this also turns the image path into a preprocessed image)

    data = data.map(get_image_label)



    # Turn the data into batches

    data_batch = data.batch(BATCH_SIZE)

  return data_batch
# Create training and validation data batches

train_data = create_data_batches(X_train, y_train)

val_data = create_data_batches(X_val, y_val, valid_data=True)
# Create a function for viewing images in a data batch

def show_25_images(images, labels):

  """

  Displays 25 images from a data batch.

  """

  # Setup the figure

  plt.figure(figsize=(10, 10))

  # Loop through 25 (for displaying 25 images)

  for i in range(25):

    # Create subplots (5 rows, 5 columns)

    ax = plt.subplot(5, 5, i+1)

    # Display an image

    plt.imshow(images[i])

    # Add the image label as the title

    plt.title(unique_breeds[labels[i].argmax()])

    # Turn gird lines off

    plt.axis("off")
# Visualize training images from the training data batch

train_images, train_labels = next(train_data.as_numpy_iterator())

show_25_images(train_images, train_labels)
# Visualize validation images from the validation data batch

val_images, val_labels = next(val_data.as_numpy_iterator())

show_25_images(val_images, val_labels)
# Setup input shape to the model

INPUT_SHAPE = [None, IMAGE_SIZE, IMAGE_SIZE, 3] # batch, height, width, colour channels



# Setup output shape of the model

OUTPUT_SHAPE = len(unique_breeds) # number of unique labels

#create model



def create_model():

    pretrained_model = tf.keras.applications.InceptionV3(input_shape=(IMAGE_SIZE,IMAGE_SIZE, 3), include_top=False)

    pretrained_model.trainable = True

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(OUTPUT_SHAPE, activation='softmax')

      ])

    

    #Define the optimizer

    optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)









    model.compile(

        optimizer=optimizer,

        loss = tf.keras.losses.CategoricalCrossentropy(),

        metrics=['accuracy']

  )

    

  # Build the model

    model.build(INPUT_SHAPE) # Let the model know what kind of inputs it'll be getting

    return model  

  



# Create a model and check its details

model = create_model()

model.summary() 
# Create early stopping (once our model stops improving, stop training)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",

                                                  patience=3) # stops after 3 rounds of no improvements
# Set a learning rate annealer

learning_rate_redcuing=ReduceLROnPlateau(monitor='val_accuracy', 

                                         patience=3,

                                         verbose=1,

                                         factor=0.5,

                                         min_lr=0.00001)
# How many rounds should we get the model to look through the data?

NUM_EPOCHS = 100
# Build a function to train and return a trained model

def train_model(NUM_EPOCHS, model):

    """

    Trains a given model and returns the trained version.

    """



    # Fit the model to the data passing it the callbacks we created

    history=model.fit(x=train_data,

                epochs=NUM_EPOCHS,

                validation_data=val_data,

                steps_per_epoch=len(X_train) // BATCH_SIZE,

                validation_freq=1, # check validation metrics every epoch

                callbacks=[learning_rate_redcuing, early_stopping])

    return history

  

# Fit the model to the data

history = train_model(NUM_EPOCHS,model)



final_accuracy = history.history["val_accuracy"][-5:]

print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
def display_training_curves(training, validation, title, subplot):

  ax = plt.subplot(subplot)

  ax.plot(training)

  ax.plot(validation)

  ax.set_title('Model '+ title)

  ax.set_ylabel(title)

  ax.set_xlabel('Epoch')

  ax.legend(['Training', 'Validation'])



plt.subplots(figsize=(10,10))

plt.tight_layout()

display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 211)

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 212)
# Make predictions on the validation data (not used to train on)

predictions = model.predict(val_data, verbose=1) # verbose shows us how long there is to go

predictions

# Check the shape of predictions

predictions.shape
# First prediction

print(predictions[0])

print(f"Max value (probability of prediction): {np.max(predictions[0])}") # the max probability value predicted by the model

print(f"Sum: {np.sum(predictions[0])}") # because we used softmax activation in our model, this will be close to 1

print(f"Max index: {np.argmax(predictions[0])}") # the index of where the max value in predictions[0] occurs

print(f"Predicted label: {unique_breeds[np.argmax(predictions[0])]}") # the predicted label
# Create a function to unbatch a batched dataset

def unbatchify(data):

  """

  Takes a batched dataset of (image, label) Tensors and returns separate arrays

  of images and labels.

  """

  images = []

  labels = []

  # Loop through unbatched data

  for image, label in data.unbatch().as_numpy_iterator():

    images.append(image)

    labels.append(unique_breeds[np.argmax(label)])

  return images, labels



# Unbatchify the validation data

val_images, val_labels = unbatchify(val_data)

val_images[0], val_labels[0]
# Turn prediction probabilities into their respective label (easier to understand)

def get_pred_label(prediction_probabilities):

  """

  Turns an array of prediction probabilities into a label.

  """

  return unique_breeds[np.argmax(prediction_probabilities)]



# Get a predicted label based on an array of prediction probabilities

pred_label = get_pred_label(predictions[0])

pred_label
def plot_pred(prediction_probabilities, labels, images, n=1):

  """

  View the prediction, ground truth label and image for sample n.

  """

  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

  

  # Get the pred label

  pred_label = get_pred_label(pred_prob)

  

  # Plot image & remove ticks

  plt.imshow(image)

  plt.xticks([])

  plt.yticks([])



  # Change the color of the title depending on if the prediction is right or wrong

  if pred_label == true_label:

    color = "green"

  else:

    color = "red"



  plt.title("Predicted label :{} ({:2.0f}%) \n True label :{}".format(pred_label,

                                      np.max(pred_prob)*100,

                                      true_label),

                                      color=color)
# View an example prediction, original image and truth label

plot_pred(prediction_probabilities=predictions, n=0,

          labels=val_labels,

          images=val_images)
def plot_pred_conf(prediction_probabilities, labels, n=1):

    """

    Plots the top 10 highest prediction confidences along with

    the truth label for sample n.

    """

    pred_prob, true_label = prediction_probabilities[n], labels[n]



    # Get the predicted label

    pred_label = get_pred_label(pred_prob)



    # Find the top 10 prediction confidence indexes

    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]

    

    # Find the top 10 prediction confidence values

    top_10_pred_values = pred_prob[top_10_pred_indexes]

    

    # Find the top 10 prediction labels

    top_10_pred_labels = unique_breeds[top_10_pred_indexes]



    # Setup plot

    top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 

                     top_10_pred_values, color="gray")

        

    plt.xticks(np.arange(len(top_10_pred_labels)),

             labels=top_10_pred_labels, rotation="vertical")

    

     # Change color of true label

    #if np.isin(true_label, top_10_pred_labels):

    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")

 

    





plot_pred_conf(prediction_probabilities=predictions,

               labels=val_labels,

               n=0)
# Let's check a few predictions and their different values

i_multiplier = 0

num_rows = 3

num_cols = 2

num_images = num_rows*num_cols

plt.figure(figsize=(5*2*num_cols, 5*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_pred(prediction_probabilities=predictions,

            labels=val_labels,

            images=val_images,

            n=i+i_multiplier)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_pred_conf(prediction_probabilities=predictions,

                labels=val_labels,

                n=i+i_multiplier)

plt.tight_layout(h_pad=1.0)

plt.show()
import datetime

import shutil



#shutil.rmtree("./models")



#Make directory

shutil.os.mkdir("./models")





def save_model(model, suffix=None):

    """

    Saves a given model in a models directory and appends a suffix (str)

    for clarity and reuse.

    """

    # Create model directory with current time

    modeldir = os.path.join("models",

                          datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))

    model_path = modeldir + "-" + suffix + ".h5" # save format of model

    print(f"Saving model to: {model_path}...")

    model.save(model_path)

    return model_path
def load_model(model_path):

  """

  Loads a saved model from a specified path.

  """

  print(f"Loading saved model from: {model_path}")

  model = tf.keras.models.load_model(model_path)

                                     

  return model
# Save our model trained on 1000 images

save_model(model, suffix="1000 images model")
# Load our model trained on 1000 images

#model_1000_images = load_model('models/20200815-22211597530072-1000 images model.h5')
# Evaluate the pre-saved model

model.evaluate(val_data)
# Evaluate the loaded model

#model_1000_images.evaluate(val_data)
test_filenames = [TEST_DIR +"/"+ fname for fname in os.listdir(TEST_DIR)]
# Create test data batch

test_data = create_data_batches(test_filenames, test_data=True)
#take one hour to complete



# Make predictions on test data batch using the loaded full model

test_predictions = model.predict(test_data, verbose=1)
# Check out the test predictions

test_predictions[:10]