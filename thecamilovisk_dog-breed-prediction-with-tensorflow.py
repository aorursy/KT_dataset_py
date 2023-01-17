# Import necessary tools

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
print("TF version:", tf.__version__)

print("TF Hub version:", hub.__version__)



# Check for GPU availability

print("GPU", "available" if tf.config.list_physical_devices("GPU") else "not")
DATA_PATH = "/kaggle/input/dog-breed-identification/"

MODELS_PATH = "/kaggle/working/models/"

LOGS_PATH = "/kaggle/working/logs/"

OUTPUT_PATH = "/kaggle/working/output/"



# Make sure that the required directories path exists

if not os.path.isdir(MODELS_PATH):

    os.makedirs(MODELS_PATH)

if not os.path.isdir(LOGS_PATH):

    os.makedirs(LOGS_PATH)

if not os.path.isdir(OUTPUT_PATH):

    os.makedirs(OUTPUT_PATH)
labels_csv = pd.read_csv(DATA_PATH + "labels.csv")

display(labels_csv.describe())

display(labels_csv.head())
# How manu images are there of each breed?

labels_csv.breed.value_counts().plot.bar(figsize=(20, 10))
labels_csv.breed.value_counts().median()
# Let's view an image

from IPython.display import Image

Image(DATA_PATH + "train/001513dfcb2ffafc82cccf4d8bbaba97.jpg")
filenames = [DATA_PATH + f"train/{fname}.jpg" for fname in labels_csv["id"]]

filenames[:10]
# Check whether number of filenames matches number of actual image files

if len(os.listdir(DATA_PATH + "train")) == len(filenames):

    print("Filenames match actual amount of files! Proceed.")

else:

    print(

        "Filenames do not match actual amount of files! Check target directory."

    )
# One more check

print(labels_csv.breed[9000])

Image(filenames[9000])
labels = labels_csv.breed.values

labels
len(labels)
# See if number of labels matches the number of filenames

if len(labels) == len(filenames):

    print("Number of labels matches number of filenames!")

else:

    print(

        "Number of labels does note matches number of filenames! Check data directory."

    )
# Find the uniques label values

unique_breeds = np.unique(labels)

print(len(unique_breeds))

print(unique_breeds)
# Turn a single label into an array of booleans (one-hot array)

print(labels[0])

labels[0] == unique_breeds
# Turn every label into a boolean array

one_hot_labels = [label == unique_breeds for label in labels]

one_hot_labels[:2]
# Setup X & y

X = filenames

y = one_hot_labels
# Set number of images to use for experimenting

NUM_IMAGES = 1000
# Split our data into training and validation of total size NUM_IMAGES

X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],

                                                  y[:NUM_IMAGES],

                                                  test_size=0.2,

                                                  random_state=42)



len(X_train), len(X_val), len(y_train), len(y_val)
# Let's have a look on our training data

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
# Simple function to return a tuple (image, label)

def get_image_label(image_path, label):

    """

  Takes an image filepath name and the associated label, processes the image and return a tuple of (image, label)

  """

    image = process_image(image_path)

    return image, label
# Define the batch size. 32 is a good start

BATCH_SIZE = 32





# Function to turn data into batches

def create_data_batches(X,

                        y=None,

                        batch_size=BATCH_SIZE,

                        valid_data=False,

                        test_data=False):

    """

  Creates batches of data out of image (X) and label (y) pairs. Shuffles the data if it's validation data.

  Also accepts test data as input (no labels).

  """

    # If the data is test dataset, we probably don't have labels

    if test_data:

        print("Creating test data batches...")

        data = tf.data.Dataset.from_tensor_slices(

            (tf.constant(X)))  # only filepaths (no labels)

        data_batch = data.map(process_image).batch(BATCH_SIZE)

        return data_batch



    # If the data is a valid dataset, we don't need to shuffle ir

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

        # Shuffling pathnames and labels bafore mapping image processor function is faster than shuffling images

        data = data.shuffle(buffer_size=len(X))



        # Create (image, label) tuples (this also turns the image path into a preprocessed image)

        data_batch = data.map(get_image_label).batch(BATCH_SIZE)



        return data_batch
# Create training and validation data batches

train_data = create_data_batches(X_train, y_train)

val_data = create_data_batches(X_val, y_val, valid_data=True)
# Check out the different attributes of our data batches

train_data.element_spec, val_data.element_spec
# Function for viewing images ina a data batch

def show_25_images(images, labels):

    """

  Displays a plot of a 25 of images and their labels from a data batch.

  """

    # Setup the figure

    plt.figure(figsize=(10, 10))

    # Loop through the 25 * for displaying 25 images:

    for i in range(25):

        ax = plt.subplot(5, 5, i + 1)

        # Display an image

        plt.imshow(images[i])

        # Add the image label as the title

        plt.title(unique_breeds[labels[i].argmax()])

        # Turn the grid lines off

        plt.axis("off")
# Let's visualize our training set

train_images, train_labels = next(train_data.as_numpy_iterator())

show_25_images(train_images, train_labels)
# Now let's visualize our validation set

val_images, val_labels = next(val_data.as_numpy_iterator())

show_25_images(val_images, val_labels)
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
# Load TensorBoard notebook extension

%load_ext tensorboard
# Function to build a TensorBoard callback

def create_tensorboard_callback():

    # Create a log directory for storing TensorBoard logs

    logdir = os.path.join(

        LOGS_PATH,  # make it so the logs get tracked whenever we run an experiment

        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return tf.keras.callbacks.TensorBoard(logdir)
# Create Early Stopping callback

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
# Fit the model to the data

model = train_model()
%tensorboard --logdir $LOGS_PATH
predictions = model.predict(val_data, verbose=1)

predictions
index = 42

print(predictions[index])

print(f"Max value (probability of prediction): {np.max(predictions[index])}")

print(f"Sum: {np.sum(predictions[index])}")

print(f"Max index: {np.argmax(predictions[index])}")

print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")
unique_breeds[113]
# Turn prediction probabilities into their respective label (easier to understand)

def get_pred_label(prediction_probabilities):

    """

  Turn an array of prediction probabilities into a label

  """

    return unique_breeds[np.argmax(prediction_probabilities)]
# Get a predicted label based on an array of prediction probabilities

get_pred_label(predictions[81])
# Function to unbatchify a batch dataset

def unbatchify(data):

    """

  Takes a batched dataset of (image, label) Tensors and return separate arrays

  of images and labels

  """

    images = []

    labels = []

    # Loop trhough unbatched data

    for image, label in data.unbatch().as_numpy_iterator():

        images.append(image)

        labels.append(get_pred_label(label))



    return images, labels
# Unbatchify the validation data

val_images, val_labels = unbatchify(val_data)

val_images[0], val_labels[0]
def plot_pred(prediction_probabilities, labels, images, n=1):

    """

  View the prediction ground truth and image for sample n

  """

    pred_prob, true_label, image = prediction_probabilities[n], labels[

        n], images[n]



    # Get the pred label

    pred_label = get_pred_label(pred_prob)



    # Plot the image & remove ticks

    plt.imshow(image)

    plt.xticks([])

    plt.yticks([])



    # Change the color of the title depending if the prediction is right or wrong

    if pred_label == true_label:

        color = "green"

    else:

        color = "red"



    # Change plot title to be predicted, probability of prediction and truth label

    plt.title(f"{pred_label} {np.max(pred_prob)*100:2.0f}% {true_label}",

              color=color)
plot_pred(prediction_probabilities=predictions,

          labels=val_labels,

          images=val_images)
def plot_pred_conf(prediction_probabilities, labels, n=1):

    """

  Plot the top 10 highest prediction confidences along with the truth label for

  sample n

  """

    pred_prob, true_label = prediction_probabilities[n], labels[n]



    # Get the predicted label

    pred_label = get_pred_label(pred_prob)



    # Find the top 10 prediction confidence indexes

    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]

    # Find the top 10 prediction connfidence values

    top_10_pred_values = pred_prob[top_10_pred_indexes]

    # Find the top 10 prediction labels

    top_10_pred_labels = unique_breeds[top_10_pred_indexes]



    # Setup plot

    top_plot = plt.bar(np.arange(len(top_10_pred_labels)),

                       top_10_pred_values,

                       color="grey")

    plt.xticks(np.arange(len(top_10_pred_labels)),

               labels=top_10_pred_labels,

               rotation="vertical")



    # Change color of true label

    if np.isin(true_label, top_10_pred_labels):

        top_plot[np.argmax(

            top_10_pred_labels == true_label)].set_color("green")

    else:

        pass
plot_pred_conf(prediction_probabilities=predictions, labels=val_labels, n=9)
# Let's check out a few predictions and their different values

i_multiplier = 10

n_rows = 3

n_cols = 2

n_images = n_cols * n_rows

plt.figure(figsize=(10 * n_cols, 5 * n_rows))

for i in range(n_images):

    plt.subplot(n_rows, 2 * n_cols, 2 * i + 1)

    plot_pred(prediction_probabilities=predictions,

              labels=val_labels,

              images=val_images,

              n=i + i_multiplier)

    plt.subplot(n_rows, 2 * n_cols, 2 * i + 2)

    plot_pred_conf(prediction_probabilities=predictions,

                   labels=val_labels,

                   n=i + i_multiplier)

plt.show()
def plot_conf_matrix(prediction_probabilities, labels):

    """

  Plot the confusion matrix of a trained model given its prediction

  probabilities and desired labels

  """

    # First, we get the corresponding labels of the predictions

    pred_labels = [

        get_pred_label(pred_probs) for pred_probs in prediction_probabilities

    ]



    # Check which breeds are present either in true and predicted labels

    breeds_in_true_labels = set(labels)

    breeds_in_pred_labels = set(pred_labels)

    breeds_in_set = [

        breed for breed in unique_breeds

        if breed in breeds_in_pred_labels and breed in breeds_in_true_labels

    ]



    # Computes the confusion matrix

    conf_mat = confusion_matrix(labels, pred_labels, labels=breeds_in_set)



    # Builds the confusion matrix dataframe (for the x and y ticks in the heatmap)

    conf_df = pd.DataFrame(conf_mat,

                           index=breeds_in_set,

                           columns=breeds_in_set)

    conf_df.dropna(inplace=True)



    # Now we plot the confusion matrix

    fig, ax = plt.subplots(figsize=(20, 20))

    conf_plot = sns.heatmap(conf_df, annot=True, cbar=False)



    plt.title("Confusion matrix")

    plt.xlabel("True label")

    plt.ylabel("Predicted label")
plot_conf_matrix(predictions, val_labels)
# Create a function to save a model

def save_model(model, suffix=None):

    """

  Save a given model in a model directory and appends a suffix (string)

  """

    # Create a model directory with current time

    modeldir = os.path.join(MODELS_PATH,

                            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    model_path = modeldir + "_" + suffix + ".h5"  # model save format

    print(f"Saving model to: {model_path}...")

    model.save(model_path)

    return model_path
# Create a function to load a trained model

def load_model(model_path):

    print(f"Loading saved model from: {model_path}...")

    model = tf.keras.models.load_model(

        model_path, custom_objects={"KerasLayer": hub.KerasLayer})

    return model
# Save our model trained on 1000 images

model_path = save_model(model, suffix="1000_images_mobilenetv2_Adam")
# Load a trained model

loaded_1000_image_model = load_model(model_path)
model.evaluate(val_data, )
model.metrics_names
# Create a data batch with the full data set

full_data = create_data_batches(X, y)
full_data
# Create a model for full model

full_model = create_model()
# Create full model callbacks

full_model_tensorboard = create_tensorboard_callback()

# No validation set when training on all the data, so we can't monitor validation accuracy

full_model_early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="accuracy", patience=3)
# Fit the full model to the full data

full_model.fit(x=full_data,

               epochs=NUM_EPOCHS,

               callbacks=[full_model_tensorboard, full_model_early_stopping])
full_model_path = save_model(full_model, suffix="full_image_set_mobilenetv2_Adam")
loaded_full_model = load_model(full_model_path)
# Load test image filenames

test_path = DATA_PATH + "test/"

test_filenames = [test_path + fname for fname in os.listdir(test_path)]

test_filenames[:10]
len(test_filenames)
# Create test data batch

test_data = create_data_batches(test_filenames, test_data=True)
test_data
# Make predictions on test data batch using the loaded full model

test_predictions = loaded_full_model.predict(test_data, verbose=1)
# Save predictions (NumPy arrary) to csv file (for access later)

np.savetxt(OUTPUT_PATH + "preds_array.csv", test_predictions, delimiter=",")
test_predictions = np.loadtxt(OUTPUT_PATH + "preds_array.csv", delimiter=",")
test_predictions
test_predictions.shape
# Create a pandas DataFrame with empty columns

preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))

preds_df
# Append test image ID's to predictions DataFrame

test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]

preds_df["id"] = test_ids

preds_df.head()
# Add the prediction probabilities to each dog breed column

preds_df[list(unique_breeds)] = test_predictions

preds_df.head()
# Save our predictions dataframe to CSV for submission to Kaggle

preds_df.to_csv(OUTPUT_PATH +

                "full_model_predictions_submission_1_mobilenetV2.csv",

                index=False)