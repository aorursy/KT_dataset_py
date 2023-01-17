# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import tensorflow_hub as hub

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import Image

import matplotlib.pyplot as plt

print('TensorFlow Version :', tf.__version__)

print('TensorFlow_Hub Version :', hub.__version__)



# Check if we're using GPU

print('GPU','Available!!, Noice' if tf.config.list_logical_devices('GPU') else 'Not Available')
labels_csv = pd.read_csv('../input/dog-breed-identification/labels.csv')
labels_csv[70:80]
labels_csv.describe()
print(labels_csv['breed'].value_counts()[:10])

labels_csv['breed'].value_counts()[:20].plot.bar(figsize=(20,10))
labels_csv['breed'].value_counts(normalize=True).plot.bar(logx=False, figsize=(20,10))

print(labels_csv['breed'].value_counts().median())
Image('../input/dog-breed-identification/train/0042188c895a2f14ef64a918ed9c7b64.jpg')
Image('../input/dog-breed-identification/train/01e787576c003930f96c966f9c3e1d44.jpg')
# Create pathnames from image ID's

filenames = []
filenames = []

for filename in labels_csv['id']:

    filenames.append('../input/dog-breed-identification/train/' + filename + '.jpg')

filenames[:10]
# Check if number of filenames are equal to number of actual image files.

import os

if len(os.listdir('../input/dog-breed-identification/train/')) == len(filenames):

    print('Yes ! they match')

else:

    print('No, they don\'t')
print(labels_csv['breed'][9000])

Image(filenames[9000])
labels = labels_csv['breed']

labels = np.array(labels)

labels



# Or we can ,

# labels = labels_csv['breed'].to_numpy()
len(labels)
# Check if number of labels are equal to number of filenames.

import os

if len(filenames) == len(labels):

    print('Yessssss ! no missing values ;-)')

else:

    print('Nooooo ! Look\'s like we have missing values to deal with')
# Let's find unique label values.

unique_breeds = np.unique(labels)

unique_breeds

len(unique_breeds)
print(labels[0])

labels[0] == unique_breeds
boolean_labels = [label == unique_breeds for label in labels]

boolean_labels[:2]
from sklearn.model_selection import train_test_split

X = filenames

y = boolean_labels
# Splitting into train and validation set.

X_train, X_valid, y_train, y_valid = train_test_split(X[:1000], y[:1000], test_size = 0.2, random_state = 42)



# Checking the dimensions of train and validation set

len(X_train), len(X_valid), len(y_train), len(y_valid)
# Let's look if everything is fine.

X_train[:1], y_train[:1]
from matplotlib.pyplot import imread

image = imread(filenames[30])

image.shape
image.max(), image.min()
tf.constant(image)
# Define image size

#IMG_SIZE = 224



# Create a function that preprocessess images

def image_process(image_path):

    '''

    Takes an image filepath and converts image into a tensor

    '''

    # Read image file

    image = tf.io.read_file(image_path)

    

    # Turn the jpg image into numerical tensor with 3 colour channels (red, green, blue)

    image = tf.image.decode_jpeg(image, channels = 3)

    

    # Convert the colour channel values from 0-255 to 0-1 values

    image = tf.image.convert_image_dtype(image, tf.float32)

    

    # Resize the image (224,224)

    image = tf.image.resize(image, size = [224,224])

    

    return image
# Creating a function that return a tuple of image and tensor [(image, tensor)]

def get_image_label(image_path, label):

    '''

    Takes an image filepath name and the associated label,

    processes the image and returns a tuple of (image, label)

    '''

    image = image_process(image_path)

    return image, label
# Creating a function to turn data into batches.

def data_batcher(X, y=None, batch_size = 32, valid_data = False, test_data = False):

    '''

    Creates batches of data out of image (X) and label (y) pairs.

    It shuffles if it's training data, but won't if it's validation data.

    Also accepts test data as input (it doesn't have labels).

    '''

    

    # If the data is test data, we won't have labels

    if test_data:

        print('Creating test data batches...')

    

        # Only filepaths, not labels

        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))

        data_batch = data.map(image_process).batch(32)

        return data_batch



    # If the data is valid dataset, we don't shuffle it.

    elif valid_data:

        print('Creating validation data batches...')

        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths

                                                   tf.constant(y)))# labels

        data_batch = data.map(get_image_label).batch(32) 

        return data_batch

    # If the data is training data set

    else:

        print('Creating training data batches...')

        # Turn filepaths and labels into tensors

        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths

                                                   tf.constant(y))) # labels

        

        # Shuffle the pathnames and labels before mapping image processor function

        # ... is faster than shuffling images

        data = data.shuffle(buffer_size = len(X))

        

        # Create a image, label tuple and turns the image path into a preprocessed image

        data = data.map(get_image_label)

        

        # Turning the training data into batches.

        data_batch = data.batch(32) 

    return data_batch
# Create training and validation data batches.

train_data = data_batcher(X_train, y_train)

val_data = data_batcher(X_valid, y_valid, valid_data = True)
# Check different attributes of our data batches.

train_data.element_spec, val_data.element_spec
# Creating a function to view images in a data batch

def show_lim_images(images,labels):

    '''

    Displays a plot of given number of images and their labels from a data batch.

    '''

    # Setting up the fig

    plt.figure(figsize=(10,10))

    # Loop through 25 for displaying 25 images

    for i in range(25):

        #create subplots (5 rows, 5 columns)

        ax = plt.subplot(5,5,i+1)

        # Display an image

        plt.imshow(images[i])

        # Add the image lable as title.

        plt.title(unique_breeds[labels[i].argmax()])

        # Turn the gridlines off

        plt.axis('off')
train_images, train_labels = next(train_data.as_numpy_iterator()) 

show_lim_images(train_images, train_labels)
valid_images, valid_labels = next(val_data.as_numpy_iterator()) 

show_lim_images(valid_images, valid_labels)
# Setup input shape to the model.

INPUT_SHAPE = [None, 224, 224,3] #batch, height, width, colour channels.

# Setup the output shape of the model.

OUTPUT_SHAPE = len(unique_breeds)

# Setup model URL from tensorflow hub.

MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4'
def create_model(input_shape = INPUT_SHAPE, output_shape = OUTPUT_SHAPE, model_url = MODEL_URL):

    print('Building model with : ', MODEL_URL, '...')

    

    #Setting up the model layers

    model = tf.keras.Sequential([

        hub.KerasLayer(MODEL_URL), #1st layer/input layer

        tf.keras.layers.Dense(units = OUTPUT_SHAPE,

                              activation='softmax')])#2nd/output layer

    

    # Compiling the model

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),

                  optimizer=tf.keras.optimizers.Adam(),

                  metrics=['accuracy'])

        

    # Building the model

    model.build(INPUT_SHAPE)

    

    return model    
model = create_model()

model.summary()
# Create early stopping 

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",

                                                  patience=3) 
# Build a function to train and return a trained model (100 epochs)

def train_model():    

    """

    Trains a given model and returns the trained version.

    """

    # Create a model

    model = create_model()

    

    # Create new TensorBoard session everytime we train a model

    #tensorboard = create_tf_callback()



    # Fit the model to the data passing it the callbacks we created

    model.fit(x=train_data,

          epochs=100,

          validation_data=val_data,

          validation_freq=1, # check validation metrics every epoch

          callbacks=[early_stopping])

  

    # Return the fitted model.

    return model
# Fit the model to the data

model = train_model()
# Make predictions on the validation data (not used to train on)

predictions = model.predict(val_data, verbose=1)

predictions[0] # Predictions of one image
def pred(index):

    '''

    Takes index value from the predictions and returns 

    highest confidence level index of the highest confidence value

    and dog breed.

    '''

    max_value = np.max(predictions[index])

    max_value_index = predictions[index].argmax()

    breed_at_that_index = unique_breeds[max_value_index]

    print(f"Confidence Level for first image : {max_value}")

    print(f"Index for the Max Value : {max_value_index}")

    print(f"Breed at that Index :  {breed_at_that_index}")
pred(0)
# Turn prediction probabilities into their respective label (easier to understand)

def get_pred_label(prediction_probabilities):

  """

  Turns an array of prediction probabilities into a label.

  """

  return unique_breeds[np.argmax(prediction_probabilities)]



# Get a predicted label based on an array of prediction probabilities

pred_label = get_pred_label(predictions[0])
# Create a function to unbatch.

def unbatch(data):

    '''

    Takes a dataset(which is in batches), and unbatches it. 

    '''

    images = []

    labels = []

    for image, label in data.unbatch().as_numpy_iterator():

        images.append(image)

        labels.append(unique_breeds[np.argmax(label)])

    return images, labels
val_images, val_labels = unbatch(val_data)

val_images[0], val_labels[0]
def plot(pred_probs, labels, images, n=1):

    '''

    View prediction, actual truth, and image for sample n

    '''

    pred_prob, true_label, image = pred_probs[n], labels[n], images[n]

    

    #Getting pred 

    pred_label = get_pred_label(pred_prob)

    

    # Plot image & remove ticks

    plt.imshow(image)

    plt.xticks([])

    plt.yticks([])

    

    # Changing colour, depending upon whether the prediction is right or wrong.

    if pred_label == true_label:

        color='green'

    else:

        color='red'

    

    

    # Change plot title

    plt.title('{} - {:2.0f}%\n{}'.format(pred_label, 

                                   np.max(pred_prob)*100,

                                   true_label),

                                   color=color)
plot(predictions, val_labels, val_images,73)
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

                     top_10_pred_values, 

                     color="black")

  plt.xticks(np.arange(len(top_10_pred_labels)),

             labels=top_10_pred_labels,

             rotation="vertical")



  # Change color of true label

  if np.isin(true_label, top_10_pred_labels):

    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("red")

  else:

    pass
plot_pred_conf(prediction_probabilities=predictions,

               labels=val_labels,

               n=20)
# Let's check a few predictions and their different values

i_multiplier = 0

num_rows = 3

num_cols = 2

num_images = num_rows*num_cols

plt.figure(figsize=(5*2*num_cols, 5*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot(pred_probs=predictions,

            labels=val_labels,

            images=val_images,

            n=i+i_multiplier)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_pred_conf(prediction_probabilities=predictions,

                labels=val_labels,

                n=i+i_multiplier)

plt.tight_layout(h_pad=1.0)

plt.show()
from datetime import datetime

def save_model(model, suffix=None):

  """

  Saves a given model in a models directory and appends a suffix (str)

  for clarity and reuse.

  """

  # Create model directory with current time

  modeldir = os.path.join("",

                          datetime.now().strftime("%Y%m%d-%H%M%s"))

  model_path = modeldir + "-" + suffix + ".h5" # save format of model

  print(f"Saving model to: {model_path}...")

  model.save(model_path)

  return model_path
def load_model(model_path):

  """

  Loads a saved model from a specified path.

  """

  print(f"Loading saved model from: {model_path}")

  model = tf.keras.models.load_model(model_path,

                                     custom_objects={"KerasLayer":hub.KerasLayer})

  return model
# Save our model trained on 1000 images

save_model(model, suffix="1000-images-Adam")
# Turn full training data in a data batch

full_data = data_batcher(X, y)
# Instantiate a new model for training on the full dataset

full_model = create_model()
# Create full model callbacks

full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",

                                                             patience=3)
# Fit the full model to the full training data

full_model.fit(x=full_data,

               epochs=5,

               callbacks=[full_model_early_stopping])
save_model(full_model, suffix="all-images-Adam")
# Load test image filenames (since we're using os.listdir(), these already have .jpg)

test = "/kaggle/input/dog-breed-identification/test/"

test_filenames = [test + fname for fname in os.listdir(test_path)]



test_filenames[:10]
# Create test data batch

test_data = data_batcher(test_filenames, test_data=True)
# Make predictions on test data batch using the loaded full model

test_preds = full_model.predict(test_data,

                                             verbose=1)
# Creating pandas DataFrame with empty columns

subm_df = pd.DataFrame(columns=["id"] + list(unique_breeds))

# Append test image ID's to predictions DataFrame

test = "/kaggle/input/dog-breed-identification/test/"

subm_df["id"] = [os.path.splitext(path)[0] for path in os.listdir(test)]

# Add the prediction probabilities to each dog breed column

subm_df[list(unique_breeds)] = test_predictions

subm_df.head()
# Taking a .csv output for submission

subm_df.to_csv("Submissions.csv",

                 index=False)