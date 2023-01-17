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
#Import necessary tools
import tensorflow as tf
print("Tensor Flow version:", tf.__version__)
import tensorflow_hub as hub
print("Tensor Flow Hub version:", hub.__version__)

#Check for GPU availability
print("GPU","available (YESS!!)" if tf.config.list_physical_devices("GPU") else "not available")
#check out the labels of our data
import pandas as pd
labels_csv=pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
print(labels_csv.describe())
print(labels_csv.head())
#How many images of each breed?
labels_csv["breed"].value_counts().plot.bar(figsize=(20,10))
# Lets view an image 
from IPython.display import Image
Image("/kaggle/input/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg")
filenames= ["/kaggle/input/dog-breed-identification/train/"+ fname + ".jpg" for fname in labels_csv["id"]]
# Check the first 10 filenames
filenames[:10]
import numpy as np
labels= labels_csv["breed"].to_numpy()
print(labels[:10])
len(labels)
# Find the unique label values
unique_breeds = np.unique(labels)
len(unique_breeds)
# Turn one label into an array of booleans
print(labels[0])
labels[0] == unique_breeds # use comparison operator to create boolean array
#Turn every label into boolean labels
boolean_labels=[label==np.array(unique_breeds) for label in labels]
boolean_labels[:2]
# Example: Turning a boolean array into integers
print(labels[0]) # original label
print(np.where(unique_breeds == labels[0])[0][0]) # index where label occurs
print(boolean_labels[0].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs
# Setup X & y variables
X = filenames
y = boolean_labels
X
# now lets make a function to preprocess the image

# Define the size
IMG_SIZE=224

#Create a function for preprocess images
def process_image(image_path):
  """
  Take image file path and turn it into tensor
  """
  #Read in image file
  image =tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image=tf.image.decode_jpeg(image,3)
  # Convert the colour channel values from 0-225 values to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired size (224, 244)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image
#Create a function to return a tuple
def get_image_label(image_path,label):
  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)
  return image, label
#Define batch size 32 is default
BATCH_SIZE=32

#Create a function to turn data to batches
def create_data_batches(x,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """

  #If data is a test dataset , we probably don't have labels
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  # If the data if a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    
    return data_batch
  else:
    # If the data is a training dataset, we shuffle it
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels  
    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(x))

    # Create (image, label) tuples (this also turns the image path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the data into batches
    data_batch = data.batch(BATCH_SIZE)
    return data_batch 
import matplotlib.pyplot as plt

#Create a function for viewing images in a data batch
def show_25_images(images,labels):
  """
  Displays 25 images from a data batch.
  """
  #Setup the figure
  plt.figure(figsize=(10,10))

  #loop through 25 for 25 images
  for i in range(25):
    #Create a subplot 5 rows and 5 columns
    ax=plt.subplot(5,5,i+1)
    #Display an image
    plt.imshow(images[i])
    # Add the image label as the title
    plt.title(unique_breeds[labels[i].argmax()])
    # Turn gird lines off
    plt.axis("off")

# Setup input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channels

# Setup output shape of the model
OUTPUT_SHAPE = len(unique_breeds) # number of unique labels

# Setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/4"
#create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,model_url=MODEL_URL):
  print("Building model with: ",model_url)

  #Setup the model layers
  model=tf.keras.Sequential([
                             hub.KerasLayer(model_url), #Layer 1 (input layer)
                             tf.keras.layers.Dense(units=output_shape,
                                                   activation="softmax") #Layer 2 (output layer)
                            ])
  
  #Compile the model
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(), #Our model wants to reduce this (how wrong its guesses are)
      optimizer=tf.keras.optimizers.Adam(), #A friend tells our model how to improve its guesses
      metrics=["accuracy"]
  )

  #Build the model
  model.build(input_shape)

  return model
# Create a model and check its details
model = create_model()
model.summary()
# Load tensoboard notebook extension
%load_ext tensorboard
import datetime

#Create a function to build a Tensorboard callback
def create_tensorboard_callback():
  #Create a dir for storing Tensorboard logs
  logdir=os.path.join("/kaggle/input/dog-breed-identification/logs",
                      #MAke it so that logs can be tracked whenever twe ran an experiment
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)
# Create early stopping callback(once our model stop improving, stop training)
early_stopping=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                patience=3) #stops after 3 rounds of no improvements
# How many rounds should we get the model to look through the data?
NUM_EPOCHS = 100 #@param {type:"slider", min:10, max:100, step:10}
#Build a function for training a model
def train_model():
  """
  Trains a given model and returned a trained version
  """
  #Create a model
  model = create_model()

  #tensorboard session evertime we train a model. For call backs
  tensorboard=create_tensorboard_callback()

  #Fit the model to the data passing it the callbacks we created
  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard,early_stopping])
  
  return model
  

# Create a data batch with the full data set
full_data = create_data_batches(X, y)
# Remind ourselves of the size of the full dataset
len(X), len(y)
X[:10]

full_data
full_model=create_model()
#Create full model callbacks
full_model_tensorboard=create_tensorboard_callback()
#no validation set when training on all the data, so we cant model val accuracy
full_model_early_stopping=tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)
# Fit the full model to the full data
full_model.fit(x=full_data,
               epochs=NUM_EPOCHS,
               callbacks=full_model_early_stopping)
# Load test image filenames
test_path = "/kaggle/input/dog-breed-identification/test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]
test_filenames[:10]
len(test_filenames)
# Create test data batch
test_data = create_data_batches(test_filenames, test_data=True)
#Make predictions on test data . will take an hour for 10000+ images
test_predictions=full_model.predict(test_data,verbose=1)
test_predictions.shape
preds_df=pd.DataFrame(columns=["id"]+list(unique_breeds))
preds_df.head()
# Append test image ID's to predictions DataFrame
test_ids=[os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids
preds_df.head()
# Add the prediction probabilities to each dog breed column
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()
