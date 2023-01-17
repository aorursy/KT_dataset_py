# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Author: Mbonu Chinedum Endurance 
# Country: Nigeria
# University: Nnamdi Azikiwe University 
# Description: Malaria Prediction From Cell Images Samples 
# Date Created: 2/07/2020 "Buhari Tenor" 
# Date Modified: 2/07/2020 "Buhari Tenor"
# Importing the necessary modules 
import os 
import cv2 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import MaxPooling2D 
# Getting the path to the dataset directory 
# Getting the path to the working directory 
workingDir = "/kaggle/input/cell-images-for-detecting-malaria"
PATH = os.path.sep.join([workingDir, "cell_images"])

# Getting the path ot the training directory 
train_dir = os.path.join(PATH, "cell_images")

# Getting the path to the validation directory 
validation_dir = os.path.join(PATH, "cell_images")
# Getting the path to the directory for the parasitized training cell images and 
# the getting the path to the directory for the uninfected training cell images 
parasitized_train_dir = os.path.join(train_dir, "Parasitized")
uninfected_train_dir = os.path.join(train_dir, "Uninfected")

# Getting the path to the directory for the parasitized validation cell images and 
# the path to the directory for the uninfected validation cell images 
parasitized_val_dir = os.path.join(validation_dir, "Parasitized") 
uninfected_val_dir = os.path.join(validation_dir, "Uninfected")
# Getting the number of images present in the parasitized training directory and the 
# number of images present in the uninfected training directory 
parasitized_images = len(os.listdir(parasitized_train_dir))
uninfected_images = len(os.listdir(uninfected_train_dir))

# Getting the number of images present in the parasitized validation directory and the 
# number of images present in the uninfected validation directory 
parasitized_images_val = len(os.listdir(parasitized_val_dir)) 
uninfected_images_val = len(os.listdir(uninfected_val_dir)) 

# Getting the sum of both the training images and validation images 
total_train = parasitized_images + uninfected_images  
total_val = parasitized_images_val + uninfected_images_val 

# Displaying the results for Training images  
print("Total Training parasitized images: {}".format(parasitized_images)); 
print("Total Training uninfected images: {}".format(uninfected_images)); 
print("__________________________________________________________________________________________________________\n");

# Displaying the results for Validation images  
print("Total Validation parasitized images: {}".format(parasitized_images_val)); 
print("Total Validation uninfected images: {}".format(uninfected_images_val)); 
print("__________________________________________________________________________________________________________\n"); 

# Displaying the total values for the images in both the training and validation directory 
print("Total Train: {}".format(total_train)); 
print("Total Validation: {}".format(total_val)); 
# Setting the batch size, number of epochs, the image height and width parameters 
batch_size = 2000
epochs = 20 
IMG_HEIGHT = 98 
IMG_WIDTH = 98 
# Creating the generator for our training images data and for our validation images data 
train_image_gen = ImageDataGenerator(rescale = 1.0 / 255.0)
validation_image_gen = ImageDataGenerator(rescale = 1.0 / 255.0) 

# Getting the training images from the train directory by using the flow from directory method 
# to load the images with a stated batch size and an image height and width. 
train_data_gen = train_image_gen.flow_from_directory(batch_size = batch_size, 
                                                    directory = train_dir, 
                                                    shuffle = True, 
                                                    target_size = (IMG_HEIGHT, IMG_WIDTH), 
                                                    class_mode = "binary")

# Getting the validation images from the validation directory by using the flow from_from_directory method 
# to load the images, shuffle them, and resize them with an image height and a specified width value. 
validation_data_gen =validation_image_gen.flow_from_directory(batch_size = batch_size, 
                                                             directory = validation_dir, 
                                                             shuffle = True, 
                                                             target_size = (IMG_HEIGHT, IMG_WIDTH), 
                                                             class_mode = "binary")
# Getting the images and labels from the training data generator 
sample_training_images, train_label = next(train_data_gen) 

# Getting the images and labels from the training data generator 
sample_training_images, train_label = next(train_data_gen) 

# Getting the images and labels from the validation data generator 
sample_validation_images, val_label = next(validation_data_gen) 

# Defining a function to plot the images in the form of a grid with 1 row and 15 columns where the 
# Images are placed in each column with their respective labels 
def plotImages(images, batch=None): 
    global train_data_gen 
    fig, axes = plt.subplots(1, 15, figsize=(17, 20))
    # Flatten the axes 
    axes = axes.flatten() 
    # Creating a loop to loop throught the image directory and plot the images 
    for img, ax, labels in zip(images, axes, batch): 
        # Extracting the respective labels or tags for the plotted images 
        for key, value in train_data_gen.class_indices.items(): 
            # Converting the labels into integer values 
            labels = int(labels) 
            # Plotting the images by the label key value for the respective goten integer value 
            if value == labels: 
                ax.set_title(key) 
                ax.imshow(img) 
                ax.axis("off")
                
    # Displaying the plot 
    plt.tight_layout() 
    plt.show() 
    
    
# Plotting 15 random images from the sample training images with its respective labels 
plotImages(sample_training_images[:15], train_label[:15])

# Plotting 15 images from the sample validation images 
plotImages(sample_validation_images[:15], val_label[:15])
# Displaying the shape of the sample training image data 
print("Input Shape: {}".format(sample_training_images.shape)); 

# Displaying the shape of the output training images label 
print("Output Shape: {}".format(train_label.shape)); 

# Displaying the shape of the validation images data 
print("Validation Input Shape: {}".format(sample_validation_images.shape)); 

# Displaying the shape of the sample output validation images label 
print("Validation Output Shape: {}".format(val_label.shape)); 
# Displaying the label class and its respective key value for the class 
labelClass = list(train_data_gen.class_indices.items())

# Showing the values 
print("Label Class: {}".format(labelClass))
# Building the model 
def ModelDefined(dim): 
    # Using the tensorflow Sequential module to create the model then add some specific 
    # Parameters. 
    model = Sequential([
        Conv2D(16, 3, padding="same", activation="relu", input_shape=dim),
        MaxPooling2D(), 
        Conv2D(32, 3, padding="same", activation="relu"), 
        MaxPooling2D(), 
        Conv2D(64, 3, padding="same", activation="relu"), 
        MaxPooling2D(), 
        Flatten(), 
        Dense(512, activation="relu"), 
        Dense(1)
        
    ])
    
    # Compiling the model and setting the optimizer to be adam, and a loss function of binary crossentropy 
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                 metrics=["accuracy"])
    
    # Returning the compiled model to be loaded easily 
    return model 


# Setting the dimensions to be passed into the model function 
dim = (IMG_HEIGHT, IMG_WIDTH, 3); 

# Creating the model 
model = ModelDefined(dim); 

# Displaying the summary of the model 
model.summary(); 
# Training the model on the input data by using the fit_generator function 
H = model.fit_generator(train_data_gen, steps_per_epoch = total_train // batch_size, 
                       epochs = epochs, 
                       validation_data = validation_data_gen, 
                       validation_steps = total_val // batch_size) 


# Saving the model for further uses 
modelName = "MalariaModel.h5" 
model.save_weights(modelName); 
# Setting the type of plot style for the graph 
plt.style.use("ggplot") 

# Getting the accuracy and the validation accuracy 
accuracy = H.history["accuracy"]
validation_accuracy = H.history["val_accuracy"]

# Getting the loss and the validation loss 
loss = H.history["loss"]
validation_loss = H.history["val_loss"] 

# Getting the epochs range 
epochs_range = range(epochs) 

# Plotting the first graph for accuracy 
plt.figure(figsize=(17, 8)); 
plt.plot(epochs_range, accuracy, label = "Training Accuracy"); 
plt.plot(epochs_range, validation_accuracy, label = "Validation Accuracy"); 
plt.xlabel("Epochs"); 
plt.ylabel("Accuracy"); 
plt.legend(loc="lower right"); 
plt.title("Training And Validation Accuracy"); 
plt.show(); 
# Plotting the second graph for loss 
plt.figure(figsize=(17, 8)) 
plt.plot(epochs_range, loss, label = "Training Loss"); 
plt.plot(epochs_range, validation_loss, label = "Validation Loss"); 
plt.xlabel("Epochs"); 
plt.ylabel("Loss"); 
plt.legend(loc="upper right"); 
plt.title("Training And Validation Losses"); 
plt.show() 
# Making predictions 
img = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized"

# Getting the first Four images 
ImgDir = list(os.listdir(img)) 
ImgDir = ImgDir[:3]

# Displaying the first 4 images in the Parasitized folder 
print(ImgDir)
print("_________________________________________________________________________")
print(""); 

# Loading Just a random image from the Parasitized images folder. 
imagePath = os.path.join(img, "C175P136NThinF_IMG_20151127_142326_cell_236.png")

# Displaying the full path to the parasitized image we want to use for prediction. 
print(imagePath)
# Loading the image into memory 
img = cv2.imread(imagePath); 

# Setting the dimensions for the loaded image to be converted into and displaying the shape of the image 
print("Loaded Image Shape: {}".format(img.shape)); 
dim = (IMG_HEIGHT, IMG_WIDTH); 

# Resizing the image 
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA); 
plt.grid(False) 
plt.imshow(img) 
plt.show() 
# Expanding the image dimensions 
image = np.expand_dims(img, axis = 0); 

# Making Final Predictions 
result = model.predict_classes(image)
# Creating a loop to get the actual predicted class 
for key, value in (train_data_gen.class_indices.items()): 
    if value == result: 
        print("The Predicted Class is: {}".format(key))