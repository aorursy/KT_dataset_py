from glob import glob

import tensorflow as tf

import tensorflow_hub as hub
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train='../input/chest-xray-pneumonia/chest_xray/train/'

val='../input/chest-xray-pneumonia/chest_xray/val/'

test='../input/chest-xray-pneumonia/chest_xray/test/'
for expression in os.listdir(train):

    print(str(len(os.listdir(train + "" + expression))) + "  " + expression ) 
categories=os.listdir(f'{train}')

categories
categoriesval=os.listdir(f'{val}')

categoriesval
traindf = pd.DataFrame()

for cat in categories:

    files = glob(train+cat +"/*")

    tempdf = pd.DataFrame({'filepath':files,'category':cat.split("/")[-1]})

    traindf = pd.concat([traindf,tempdf])
valdf = pd.DataFrame()

for cat in categories:

    files = glob(val+cat +"/*")

    tempdf = pd.DataFrame({'filepath':files,'category':cat.split("/")[-1]})

    valdf = pd.concat([valdf,tempdf])
traindf.head()
valdf.head()
traindf.category.value_counts()
valdf.category.value_counts()
data=pd.concat([traindf,valdf])
data.shape,data.category.value_counts()
gby_cnt = data.groupby("category").aggregate('count').reset_index().sort_values(by='filepath',ascending=False)
gby_cnt.plot(kind='bar',x = 'category',y = 'filepath',title = 'Counts from Each Category');
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

from PIL import Image

for i in range(16):

    path = data.sample(1)['filepath'].values[0]

    category = path.split("/")[1]

    ex_img = Image.open(path)

    ax = plt.subplot(4, 4, i + 1)

    ax.imshow(ex_img)

    plt.axis('off')



plt.tight_layout();
import numpy as np

labels = data["category"].to_numpy() 

# labels = np.array(labels) # does same thing as above

labels
# Find the unique label values

unique_label = np.unique(labels)

len(unique_label)
# Turn every label into a boolean array

boolean_labels = [label == unique_label for label in labels]

boolean_labels[:2]
# Example: Turning boolean array into integers

print(labels[0]) # original label

print(np.where(unique_label == labels[0])) # index where label occurs

print(boolean_labels[0].argmax()) # index where label occurs in boolean array

print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs
# Setup X & y variables

X = data.filepath.values

y = boolean_labels
len(X),len(y)
# Let's split our data into train and validation sets

from sklearn.model_selection import train_test_split



# Split them into training and validation of total size NUM_IMAGES

X_train, X_val, y_train, y_val = train_test_split(X,

                                                  y,

                                                  test_size=0.2,

                                                  random_state=42)



len(X_train), len(y_train), len(X_val), len(y_val)
# Let's have a geez at the training data

X_train[45], y_train[:2]
# Convert image to NumPy array

from matplotlib.pyplot import imread

image = imread(X_train[42])

image.shape
image.max(), image.min()
# turn image into a tensor

tf.constant(image)[:2]
# Define image size

IMG_SIZE = 224



# Create a function for preprocessing images

def process_image(image_path, img_size=IMG_SIZE):

  """

  Takes an image file path and turns the image into a Tensor.

  """

  # Read in an image file

  image = tf.io.read_file(image_path)

  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)

  image = tf.image.decode_jpeg(image, channels=3)

  # Convert the colour channel values from 0-255 to 0-1 values

  image = tf.image.convert_image_dtype(image, tf.float32)

  # Resize the image to our desired value (224, 224)

  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])



  return image
# Create a simple function to return a tuple (image, label)

def get_image_label(image_path, label):

  """

  Takes an image file path name and the assosciated label,

  processes the image and reutrns a typle of (image, label).

  """

  image = process_image(image_path)

  return image, label
# Demo of the above

(process_image(X[42]), tf.constant(y[42]))
# Define the batch size, 32 is a good start

BATCH_SIZE = 32



# Create a function to turn data into batches

def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):

  """

  Creates batches of data out of image (X) and label (y) pairs.

  Shuffles the data if it's training data but doesn't shuffle if it's validation data.

  Also accepts test data as input (no labels).

  """

  # If the data is a test dataset, we probably don't have have labels

  if test_data:

    print("Creating test data batches...")

    data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)

    data_batch = data.map(process_image).batch(BATCH_SIZE)

    return data_batch

  

  # If the data is a valid dataset, we don't need to shuffle it

  elif valid_data:

    print("Creating validation data batches...")

    data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths

                                               tf.constant(y))) # labels

    data_batch = data.map(get_image_label).batch(BATCH_SIZE)

    return data_batch



  else:

    print("Creating training data batches...")

    # Turn filepaths and labels into Tensors

    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),

                                               tf.constant(y)))

    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images

    data = data.shuffle(buffer_size=len(X))



    # Create (image, label) tuples (this also turns the iamge path into a preprocessed image)

    data = data.map(get_image_label)



    # Turn the training data into batches

    data_batch = data.batch(BATCH_SIZE)

  return data_batch
# Create training and validation data batches

train_data = create_data_batches(X_train, y_train)

val_data = create_data_batches(X_val, y_val, valid_data=True)
# Check out the different attributes of our data batches

train_data.element_spec, val_data.element_spec
import matplotlib.pyplot as plt



# Create a function for viewing images in a data batch

def show_25_images(images, labels):

  """

  Displays a plot of 25 images and their labels from a data batch.

  """

  # Setup the figure

  plt.figure(figsize=(10, 10))

  # Loop through 25 (for displaying 25 images)

  for i in range(10):

    # Create subplots (5 rows, 5 columns)

    ax = plt.subplot(5, 5, i+1)

    # Display an image 

    plt.imshow(images[i])

    # Add the image label as the title

    plt.title(unique_label[labels[i].argmax()])

    # Turn the grid lines off

    plt.axis("off")
# # Now let's visualize the data in a training batch

train_images, train_labels = next(train_data.as_numpy_iterator())

show_25_images(train_images, train_labels)
# # Now let's visualize our validation set

val_images, val_labels = next(val_data.as_numpy_iterator())

show_25_images(val_images, val_labels)
# Setup input shape to the model

INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channels



# Setup output shape of our model

OUTPUT_SHAPE = len(unique_label)



# Setup model URL from TensorFlow Hub

MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
# Create a function which builds a Keras model

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):

  print("Building model with:", MODEL_URL)



  # Setup the model layers

  model = tf.keras.Sequential([

    hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)

    tf.keras.layers.Dense(units=OUTPUT_SHAPE,

                          activation="softmax") # Layer 2 (output layer)

  ])



  # Compile the model

  model.compile(

      loss=tf.keras.losses.CategoricalCrossentropy(),

      optimizer=tf.keras.optimizers.Adam(),

      metrics=["accuracy"]

  )



  # Build the model

  model.build(INPUT_SHAPE)



  return model
model = create_model()

model.summary()
# Load TensorBoard notebook extension

%load_ext tensorboard
import datetime



# Create a function to build a TensorBoard callback

def create_tensorboard_callback():

  # Create a log directory for storing TensorBoard logs

  logdir = os.path.join("../input/chest-xray-pneumonia/chest_xray",

                        # Make it so the logs get tracked whenever we run an experiment

                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  return tf.keras.callbacks.TensorBoard(logdir)
# Create early stopping callback

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",

                                                  patience=3)
# Build a function to train and return a trained model

def train_model():

  """

  Trains a given model and returns the trained version.

  """

  # Create a model

  model = create_model()



  # Create new TensorBoard session everytime we train a model

  tensorboard = create_tensorboard_callback()



  # Fit the model to the data passing it the callbacks we created

  model.fit(x=train_data,

            epochs=20,

            validation_data=val_data,

            validation_freq=1,

            callbacks=[early_stopping])

  # Return the fitted model

  return model
# Fit the model to the data

model = train_model()
# Make predictions on the validation data (not used to train on)

predictions = model.predict(val_data, verbose=1)

predictions
# First prediction

index = 42

print(predictions[index])

print(f"Max value (probability of prediction): {np.max(predictions[index])}")

print(f"Sum: {np.sum(predictions[index])}")

print(f"Max index: {np.argmax(predictions[index])}")

print(f"Predicted label: {unique_label[np.argmax(predictions[index])]}")
# Turn prediction probabilities into their respective label (easier to understand)

def get_pred_label(prediction_probabilities):

  """

  Turns an array of prediction probabilities into a label.

  """

  return unique_label[np.argmax(prediction_probabilities)]



# Get a predicted label based on an array of prediction probabilities

pred_label = get_pred_label(predictions[81])

pred_label
# Create a function to unbatch a batch dataset

def unbatchify(data):

  """

  Takes a batched dataset of (image, label) Tensors and reutrns separate arrays

  of images and labels.

  """

  images = []

  labels = []

  # Loop through unbatched data

  for image, label in data.unbatch().as_numpy_iterator():

    images.append(image)

    labels.append(unique_label[np.argmax(label)])

  return images, labels



# Unbatchify the validation data

val_images, val_labels = unbatchify(val_data)

val_images[0], val_labels[0]
def plot_pred(prediction_probabilities, labels, images, n=1):

  """

  View the prediction, ground truth and image for sample n

  """

  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]



  # Get the pred label

  pred_label = get_pred_label(pred_prob)



  # Plot image & remove ticks

  plt.imshow(image)

  plt.xticks([])

  plt.yticks([])



  # Change the colour of the title depending on if the prediction is right or wrong

  if pred_label == true_label:

    color = "green"

  else:

    color = "red"

  

  # Change plot title to be predicted, probability of prediction and truth label

  plt.title("{} {:2.0f}% {}".format(pred_label,

                                    np.max(pred_prob)*100,

                                    true_label),

                                    color=color)
plot_pred(prediction_probabilities=predictions,

          labels=val_labels,

          images=val_images,

          n=77)
def plot_pred_conf(prediction_probabilities, labels, n=1):

  """

  Plus the top 10 highest prediction confidences along with the truth label for sample n.

  """

  pred_prob, true_label = prediction_probabilities[n], labels[n]



  # Get the predicted label

  pred_label = get_pred_label(pred_prob)



  # Find the top 10 prediction confidence indexes

  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]

  # Find the top 10 prediction confidence values

  top_10_pred_values = pred_prob[top_10_pred_indexes]

  # Find the top 10 prediction labels

  top_10_pred_labels = unique_label[top_10_pred_indexes]



  # Setup plot

  top_plot = plt.bar(np.arange(len(top_10_pred_labels)),

                     top_10_pred_values,

                     color="grey")

  plt.xticks(np.arange(len(top_10_pred_labels)),

             labels=top_10_pred_labels,

             rotation="vertical")

  

  # Change color of true label

  if np.isin(true_label, top_10_pred_labels):

    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")

  else:

    pass
plot_pred_conf(prediction_probabilities=predictions,

               labels=val_labels,

               n=9)
# Let's check out a few predictions and their different values

i_multiplier = 20

num_rows = 3

num_cols = 2

num_images = num_rows*num_cols

plt.figure(figsize=(10*num_cols, 5*num_rows))

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
for expression in os.listdir(train):

    print(str(len(os.listdir(train + "" + expression))) + "  " + expression ) 





categoriestest=os.listdir(f'{test}')

testdf = pd.DataFrame()

for cat in categories:

    files = glob(test+cat +"/*")

    tempdf = pd.DataFrame({'filepath':files,'category':cat.split("/")[-1]})

    testdf = pd.concat([testdf,tempdf])
# Create test data batch

test_data = create_data_batches(testdf.filepath.values, test_data=True)
# Make predictions on test data batch using the  model

test_predictions = model.predict(test_data,

                                             verbose=1)
test_predictions[:10]
# Create a pandas DataFrame with empty columns

preds_df = pd.DataFrame(columns=["id"] + list(unique_label))

preds_df.head()
# Append test image ID's to predictions DataFrame

test_ids = testdf.filepath.values

preds_df["id"] = test_ids
preds_df.head()
# Add the prediction probabilities to each dog breed column

preds_df[list(unique_label)] = test_predictions

preds_df.head()
# Save our predictions dataframe to CSV for submission to Kaggle

preds_df.to_csv("sub.csv",

                index=False)