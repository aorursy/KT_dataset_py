# Import necessary tools into kaggle

import tensorflow as tf

import tensorflow_hub as hub

print("TF version : ",tf.__version__)

print("TF Hub version : ", hub.__version__)



# Cheak for GPU availability

print("GPU","available (Yes !)" if tf.config.list_physical_devices("GPU") else "Not Available")
# Cheakout the labels of our data

import pandas as pd

labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")

print(labels_csv.describe())

print(labels_csv.head())
labels_csv.head()
labels_csv["breed"].value_counts().plot.bar(figsize=(20,10));
labels_csv["breed"].value_counts().median()
# Let's view an image

from IPython.display import Image

Image("../input/dog-breed-identification/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg")
labels_csv.head()
# Create pathnames from image ID's

filenames = ["../input/dog-breed-identification/train/"+names+".jpg" for names in labels_csv["id"] ]

# Cheak the first 10

filenames[:10]
# Cheak weather number of filenames matches number of actual image files

import os

if len(os.listdir("../input/dog-breed-identification/train/"))==len(filenames):

    print("Filenames match equal ammount of files ! Proceed")

else:

    print("filenames do not match actual ammount of files, cheak the target directory.")
Image(filenames[9275])
import numpy as np

labels = labels_csv["breed"].to_numpy()

# labels = np.array(labels)  # Does same thing as above

labels , len(labels)
# See if number of labels matches the number of filenames

if len(labels) == len(filenames):

    print("Number of labels matches number of filenames !")

else:

    print("Number of labels does not match number of filenames, cheak data directories")
# Find the unique label values

unique_breeds = np.unique(labels)

unique_breeds,len(unique_breeds)
# Trun a single label into array of booleans

print(labels[0])

labels[0] == unique_breeds
# turn every label into a boolean array

boolean_labels = [label == unique_breeds for label in labels]

boolean_labels[:2]
# Example : Turning boolean array into integers

print(labels[0]) # original label

print(np.where(unique_breeds==labels[0])) # index where label occurs

print(boolean_labels[0].argmax()) # index where label occurs in boolean array

print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs
print(labels[2])

print(boolean_labels[2].astype(int))
filenames[:10]
# Setup X and Y variables 

X = filenames 

Y = boolean_labels

len(filenames)
# Set number of images to use for experimenting

NUM_IMAGES = 1000 #@param {type:"slider", min:1000 , max:10000 ,step:100 } works with colab
# Let's split our data into train and validation sets

from sklearn.model_selection import train_test_split



np.random.seed(42)

# Split them into training and validation of total size Num_Images

X_train,X_valid,Y_train,Y_valid = train_test_split(X[:NUM_IMAGES],Y[:NUM_IMAGES],test_size=0.2,random_state=42)



len(X_train),len(Y_train),len(X_valid),len(Y_valid)
X_train[:2] ,Y_train[:2]
# Convert image into NumPy array

from matplotlib.pyplot import imread

image = imread(filenames[42])

image.shape

image
image.max(),image.min()
# Turn image into Tensor

tf.constant(image)
tensor = tf.io.read_file(filenames[26])

tensor
tensor = tf.image.decode_jpeg(tensor ,channels=3)

tensor
tensor = tf.image.convert_image_dtype(tensor, tf.float32)

tensor
# Define image size 

IMG_SIZE = 224



# Create a function for preprocessing images 

def process_image(image_path,img_size=IMG_SIZE):

    """

    Takes an image file path and turns the image into tensors

    """

    

    # Read in a image file 

    image = tf.io.read_file(image_path)

    

    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)

    image = tf.image.decode_jpeg(image ,channels=3)

    

    # Convert the colour channel values from 0-255 to 0-1 values

    image = tf.image.convert_image_dtype(image, tf.float32)

    

    # Resize the image to our desired value 

    image = tf.image .resize(image, size=[IMG_SIZE, IMG_SIZE])

    

    return image
# Create a simple function to return a tuple (image,label)

def get_image_label(image_path,label):

    """

    Takes an image file path name and the associated label,

    processes the image and returns a tuple of (image,label)

    """

    image = process_image(image_path)

    return image, label
(process_image(X[42]),tf.constant(Y[42]))
# Define the batch size, 32 is a good start

BATCH_SIZE = 32



# Create a function to turn into a batches 



def create_data_batches(X, Y=None,batch_size=BATCH_SIZE, valid_data=False , test_data=False):

    """

    Creates batches of data out of image (X) and label (Y) pairs.

    Suffles the data if it is training data but does'nt suffle if it is validation data.

    Also accepts test data as input (no labels)

    """

    # If the data is a test dataset, we probably don't have labels

    if test_data:

        print("Creating test data batches... ")

        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))  # Only filepaths (NO labels)

        data_batch = data.map(process_image).batch(BATCH_SIZE)

        return data_batch

    

    # If the data is a valid dataset , we don't need to suffle it 

    elif valid_data:

        print("Creating validation data batches... ")

        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths

                                                   tf.constant(Y))) # labels 

        data_batch = data.map(get_image_label).batch(BATCH_SIZE)

        return data_batch

    

    # Training dataset

    else: 

        print("Creating training data batches...")

        # Turn filepaths and labels into Tensors

        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths

                                                   tf.constant(Y))) # labels

        # Shffling pathnames and labels before mapping image processor function is faster than suffling images

        data = data.shuffle(buffer_size=len(X))

        

        # Create (image, label) tuples (this also turns the image path into a preprocessed image)

        data = data.map(get_image_label)

        

        # Turn the training data into batches

        data_batch = data.batch(BATCH_SIZE)

        

    return data_batch
# Create training and validation data batches

train_data = create_data_batches(X_train, Y_train)

val_data = create_data_batches(X_valid, Y_valid ,valid_data=True)
# Cheakout the different attributes of our data batches

train_data.element_spec ,val_data.element_spec
import matplotlib.pyplot as plt



# Create a function for viewing images in a data batch

def show_25_images(images,labels):

    fig = plt.figure(figsize=(10,10))

    for i in range(0,25):

        fig.add_subplot(5,5,i+1)

        plt.imshow(images[i])

        plt.title(unique_breeds[train_labels[i].argmax()])

        plt.axis("off")

    plt.show()
train_data
train_images, train_labels = next(train_data.as_numpy_iterator())

len(train_images),len(train_labels)



# Now let's visualize the data in a training batch

show_25_images(train_images, train_labels)
# noe let's visualize our validation set

valid_images, valid_labels= next(val_data.as_numpy_iterator())

show_25_images(valid_images, valid_labels)
IMG_SIZE
# Setup the  input to the model

INPUT_SHAPE = [None, IMG_SIZE,IMG_SIZE,3] # batch,height, Width ,colour channels



# Setup the Output shape of the model

OUTPUT_SHAPE = len(unique_breeds)



# Setup model URL from tensorflow hub

MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
# Create a  function which builds a keras model 

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE,model_url=MODEL_URL):

    print("building model with : ",MODEL_URL)

    

    # Setup the model layers

    model = tf.keras.Sequential([

        hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)

        tf.keras.layers.Dense(units=OUTPUT_SHAPE,

                             activation= "softmax")  # Layer 2 (output layer)

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

    logdir= os.path.join("../working/outputs/logs",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return tf.keras.callbacks.TensorBoard(logdir)
# Create early stopping callback

early_stopping =tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3)



NUM_EPOCHS = 100 #@param {type:"slider" ,min:10 , max:100}
# Cheak to make sure we're still running on the GPU

print("GPU","available (Yes !)" if tf.config.list_physical_devices("GPU") else "Not Available")
# Build a function to train and return a trained model

def train_model():

    """

    Trains a given model and returns the trained version.

    """

    # Create a model

    model = create_model()

    # Create new TensorBoard  sessiion everytime we train a model

    tensorboard = create_tensorboard_callback()

    # Fit the model to the data passing it the callbacks we created

    model.fit(x=train_data,

             epochs=NUM_EPOCHS,

             validation_data= val_data,

             validation_freq= 1,

             callbacks= [tensorboard, early_stopping])

    #Return the fitted model

    return model
# Fit the model to the data

model = train_model()


%tensorboard --logdir working/outputs/logs

!kill 5770
val_data
predictions = model.predict(val_data, verbose= 1)

predictions
predictions.shape
len(Y_valid)
# First prediction

index = 1

print(predictions[index])

print(f"Max value (probablity of prediction): {np.max(predictions[index])}")

print(f"Sum : {np.sum(predictions[index])}")

print(f"Max index : {np.argmax(predictions[index])}")

print(f"Predicted label : {unique_breeds[np.argmax(predictions[index])]}")

# Turn the prediction probablities into their respctive label (easier to understand)

def get_pred_label(prediction_probabilities):

    """

    turns an array of prediction probablities into labels.

    """

    return unique_breeds[np.argmax(prediction_probabilities)]



# Get a predicted label based on an array of prediction probablities 

pred_label = get_pred_label(predictions[81])

pred_label
val_data
# create a function to unbatch a batch dataset



def unbatchify(data):

    """

    Takes a batched dataset of (image, label) Tensors and returns separate arrays of images and labels.

    """

    images = []

    labels = []

    # Loop through unbatched data

    for image,label in data.unbatch().as_numpy_iterator():

        images.append(image)

        labels.append(unique_breeds[np.argmax(label)])

    return images,labels



# Unbatchify the validation data

val_images ,val_labels = unbatchify(val_data)

val_images[0] , val_labels[0]
def plot_pred(prediction_probabilities, labels, images ,n=1):

    """

    View the prediction, ground truth and image of the sample n

    """

    pred_prob, true_label,image =prediction_probabilities[n] ,labels[n],images[n]

    

    # Get the pred label

    pred_label = get_pred_label(pred_prob)

    

    # Plot image & remove ticks

    plt.imshow(image)

    plt.xticks([])

    plt.yticks([])

    # Change the color of the title depending upon the prediction is right or wrong 

    if pred_label == true_label:

        color="green"

    else:

        color="red"

    # Change plot title to be predicted, probablity of prediction and truth label

    plt.title("{} {:2.0f}% {}".format(pred_label,

                                     np.max(pred_prob)*100,

                                     true_label),color=color)
plot_pred(prediction_probabilities= predictions,

         labels= val_labels,

         images= val_images,

         n=77)
def plot_pred_conf(prediction_probabilities , labels , n):

    """

    Plus the top 10 highest prediction confidence along with the truth label for sample n.

    """

    pred_prob, true_label = prediction_probabilities [n], labels[n]

    

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

                      color="grey")

    plt.xticks(np.arange(len(top_10_pred_labels)),

              labels=top_10_pred_labels,

              rotation="vertical")

    

    # Change the colour of true label

    if np.isin(true_label, top_10_pred_labels):

        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")

    else:

        pass
plot_pred_conf(prediction_probabilities=predictions,

              labels=val_labels,

              n=96

              )
# Let's cheak out a few predictions and their different values 

i_multiplier = 20

num_rows= 3

num_cols= 2

num_images = num_rows*num_cols

plt.figure(figsize=(10*num_cols,5*num_rows))

for i in range(num_images):

    plt.subplot(num_rows,2*num_cols,2*i+1)

    plot_pred(prediction_probabilities=predictions,

             labels=val_labels,

             images=val_images,

             n=i+i_multiplier)

    plt.subplot(num_rows ,2*num_cols, 2*i+2)

    plot_pred_conf(prediction_probabilities=predictions,

                  labels=val_labels,

                  n=i+i_multiplier)

plt.tight_layout()

plt.show()
# Create a function to save a model

def save_model(model, suffix=None):

    """

    Saves a given model in a models directory and appends a suffix (string)

    """

    # Create a model directory pathname with current time

    modeldir = os.path.join("../working/models",datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))

    model_path = modeldir + "-" +suffix+".h5" # save format to model

    print(f"Saving model to : {model_path}...")

    model.save(model_path)

    return model_path
# Create a function to load a train model 

def load_model(model_path):

    """

    Loads a save model from a specified path 

    """

    print(f"Loading saved model from:  {model_path}")

    model = tf.keras.models.load_model(model_path,

                                      custom_objects={"KerasLayer":hub.KerasLayer})

    return model
! cd ../working
! mkdir models
# Save our model trained on 1000 images 

save_model_path=save_model(model, suffix="1000-images-mobilenetv2-Adam")
# Load a train model

loaded_1000_image_adam_model = load_model(save_model_path)
# Evaluate the loaded model 

loaded_1000_image_adam_model.evaluate(val_data)
# Evaluate the pre-saved model

model.evaluate(val_data)
len(X) , len(Y)
# Create a data set from a full dataset 

full_data = create_data_batches(X,Y)
full_data
# Create A model for full model

full_model = create_model()
# Create full model callbacks

full_model_tensorboard = create_tensorboard_callback()

# No validation set when training on all the data, so we can't monitor validation accuracy 

full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",patience=3)
# Fit the full model to the full data

full_model.fit(x=full_data,

              epochs=NUM_EPOCHS,

              callbacks=[full_model_tensorboard, full_model_early_stopping])

save_model(full_model,suffix="full-image-model-mobilenetv2-Adam")
import os 

# Load test image filenames

test_path = "../input/dog-breed-identification/test/"

test_filenames = [test_path + fname for fname in os.listdir(test_path)]

test_filenames

ids=[i[:-4] for i in os.listdir(test_path)]

test_filenames
# Create test databatch

test_data=create_data_batches(test_filenames,test_data=True)
test_data
# make predictions on test data using full model

test_predictions = full_model.predict(test_data, verbose = 1)
test_predictions
np.savetxt("../working/outputs/preds_array.csv",test_predictions,delimiter=",")
submission=pd.DataFrame(test_predictions,columns=unique_breeds)

submission.insert(0,"id",ids)

submission
submission.to_csv("../working/outputs/preds_array.csv",index=False)
# Creating a function to apply model on our own images



def identify_breed(filepath):

    temp = create_data_batches([filepath,],test_data=True)

    result = full_model.predict(temp, verbose = 1)

    result = unique_breeds[result.argmax()]

    return result



xi = '../input/dog-breed-identification/test/1672018bbbc549cc43a14d9129197f08.jpg'

print(identify_breed(xi))