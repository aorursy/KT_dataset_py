import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import os
import datetime
DATA_PATH = "/kaggle/input/dog-breed-identification/"
MODELS_PATH = "/kaggle/working/models/"
LOGS_PATH = "/kaggle/working/logs/"
OUTPUT_PATH = "/kaggle/working/output/"

if not os.path.isdir(MODELS_PATH):
    os.makedirs(MODELS_PATH)
if not os.path.isdir(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
labels_csv = pd.read_csv(DATA_PATH + "labels.csv")

#getting filenames
filenames = [DATA_PATH + f"train/{fname}.jpg" for fname in labels_csv["id"]]
filenames[:10]
labels = labels_csv.breed.values
labels
#unique breeds labels
unique_breeds = np.unique(labels)
print(len(unique_breeds))
print(unique_breeds)
# label into one-hot array
print(labels[0])
labels[0] == unique_breeds
one_hot_labels = [label == unique_breeds for label in labels]
one_hot_labels[:2]
X = filenames
y = one_hot_labels
NUM_IMAGES = 10222
X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)

len(X_train), len(X_val), len(y_train), len(y_val)
X_train[:2], y_train[:2]
IMG_SIZE = 224


# Function for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
    """
  Takes an image filepath and turns it into a Tensor
  """
    # Read the image file
    image = tf.io.read_file(image_path)
    # Turn the jpeg image into numerical Tensor with 3 color channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the color channels values range from 0-255 to 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired values (224, 224)
    image = tf.image.resize(image, size=(img_size, img_size))
    # Return the modified image
    return image
def get_image_label(image_path, label):
    """
  Takes an image filepath name and the associated label, processes the image and return a tuple of (image, label)
  """
    image = process_image(image_path)
    return image, label
BATCH_SIZE = 32
def create_data_batches(X,
                        y=None,
                        batch_size=BATCH_SIZE,
                        valid_data=False,
                        test_data=False):
    """
  Creates batches of data out of image (X) and label (y) pairs. Shuffles the data if it's validation data.
  Also accepts test data as input (no labels).
  """
   
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X)))  # only filepaths (no labels)
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((
            tf.constant(X),  # filepaths
            tf.constant(y)))  # labels
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X), tf.constant(y)))
        
        data = data.shuffle(buffer_size=len(X))

       
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)

        return data_batch
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)
train_data.element_spec, val_data.element_spec
def show_25_images(images, labels):
    """
  Displays a plot of a 25 of images and their labels from a data batch.
  """
   
    plt.figure(figsize=(10, 10))
    
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        
        plt.imshow(images[i])
        
        plt.title(unique_breeds[labels[i].argmax()])
       
        plt.axis("off")
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)
val_images, val_labels = next(val_data.as_numpy_iterator())

INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE,
               3]  # batch, hieght, width, color channels

# Setup output shape of our model
OUTPUT_SHAPE = len(unique_breeds)

# Setup the MobileNetV2 model URL from TensorFlow hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
# Function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE,
                 output_shape=OUTPUT_SHAPE,
                 model_url=MODEL_URL):
    print("Building model with:", model_url)

    # Setup the model layers
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url),  # layer 1 (input layer)
        tf.keras.layers.Dense(units=output_shape,
                              activation="softmax")  # layer 2 (output layer)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    # Build the model
    model.build(input_shape)

    return model
model = create_model()
model.summary()
%load_ext tensorboard
def create_tensorboard_callback():
    
    logdir = os.path.join(
        LOGS_PATH,  
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)
NUM_EPOCHS = 100


# Function to train and return a trained model
def train_model(num_epochs=NUM_EPOCHS):
    """
  Trains a given model and return the trained version.
  """
    # Create a model
    model = create_model()

    # Create a new TensorBoard session everytime we train a model
    tensorboard = create_tensorboard_callback()

    # Fit the model to the data passing it the callbacks we created
    model.fit(x=train_data,
              epochs=NUM_EPOCHS,
              validation_data=val_data,
              validation_freq=1,
              callbacks=[tensorboard, early_stopping])

    # Return the fitted model
    return model
model = train_model()
model_path = save_model(model, suffix="1000_images_mobilenetv2_Adam")
full_model_path=model_path
loaded_full_model = load_model(model_path)
loaded_full_model = load_model(full_model_path)
# Load test image filenames
test_path = DATA_PATH + "test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]
test_filenames[:10]
# Create test data batch
test_data = create_data_batches(test_filenames, test_data=True)
test_data

test_predictions = loaded_full_model.predict(test_data, verbose=1)

np.savetxt(OUTPUT_PATH + "preds_array.csv", test_predictions, delimiter=",")

test_predictions = np.loadtxt(OUTPUT_PATH + "preds_array.csv", delimiter=",")
test_predictions.shape


preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df

test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids
preds_df.head()
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()
# Save  
preds_df.to_csv(OUTPUT_PATH +
                "Coe6_101703122_Avani.csv",
                index=False)