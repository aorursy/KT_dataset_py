#Import all the necessary packages needed for your problem

import tensorflow as tf

import tensorflow_hub as hub

import pandas as pd

import numpy as np

import os 
#As there are two folder storing their file path in two different variables

Train_female_path="../input/gender-recognition-200k-images-celeba/Dataset/Train/Female"

Train_male_path="../input/gender-recognition-200k-images-celeba/Dataset/Train/Male"

Validation_female_path="../input/gender-recognition-200k-images-celeba/Dataset/Validation/Female"

Validation_male_path="../input/gender-recognition-200k-images-celeba/Dataset/Validation/Male"

Test_female_path="../input/gender-recognition-200k-images-celeba/Dataset/Test/Female"

Test_male_path="../input/gender-recognition-200k-images-celeba/Dataset/Test/Male"
#Creating a list to store all the training male and female filepaths



female_train_files=[]

female_train=list()

male_train=list()

# listdir lists all the files in the given file path and stores it in the list

female_train_files=os.listdir(Train_female_path)

male_train_files=os.listdir(Train_male_path)

# As for now we only get the file name as in (01.jpg) so appending the file path with the filename

for i in range(len(os.listdir(Train_female_path))):

    female_train.append(Train_female_path+"/"+str(female_train_files[i]))

for i in range(len(os.listdir(Train_male_path))):

    male_train.append(Train_male_path+"/"+str(male_train_files[i]))

  
# Checking for the entire size of the training data

len(female_train)+len(male_train)
Train_df=pd.DataFrame()
# Creating a training data frame with female images path and their target value as Female

Train_df=pd.DataFrame({"ID":female_train,"Target":"Female"})
len(Train_df)
Train_df["ID"][0],Train_df["Target"][0]
# Creating Dataframe from male file path and assigning their as Male

Male_df=pd.DataFrame({"ID":male_train,"Target":"Male"})
# Combining the male and female training dataframe

Train_df=Train_df.append(Male_df,ignore_index=False)
len(Train_df)
# Shuffling the entire data frame as their in the female first and male last order

Train_df=Train_df.sample(frac=1)
Train_df.head()
#Fetching all the file names from Training dataframe and storing it as list

All_training_files=[fname for fname in Train_df["ID"]]
All_training_files[:7]
# Converting the target variable to num py

labels=Train_df["Target"].to_numpy()
labels[:10]
# Finding the unique values in the labels as there is only two target that needs to be predicted

true_labels=np.unique(labels)
len(true_labels)
labels[1]==true_labels
boolean_labels=[labels == true_labels for labels in labels]

len(boolean_labels)
print(labels[0])

print(np.where(true_labels[0]==labels[0]))

print(boolean_labels[0].argmax())

print(boolean_labels[0].astype(int))

boolean_labels[0]
Train_df.tail()
# Creating the Feature variable and Target variable

X=All_training_files

y=boolean_labels
# Experimenting with 10k samples 

NUM_IMAGES=10000
female_val_files=[]

female_val=list()

male_val=list()

#Storing  all the files in given file path into two variables

female_val_files=os.listdir(Validation_female_path)

male_val_files=os.listdir(Validation_male_path)

# Appending the filepath with the file name

for i in range(len(os.listdir(Validation_female_path))):

    female_val.append(Validation_female_path+"/"+str(female_val_files[i]))

for i in range(len(os.listdir(Validation_male_path))):

    male_val.append(Validation_male_path+"/"+str(male_val_files[i]))

  
female_val[1]
# Creating a dataframe for male and female

Valid_df=pd.DataFrame({"ID":female_val,"Target":"Female"})

new_val_row=pd.DataFrame({"ID":male_val,"Target":"Male"})

# Combining the both 

Valid_df=Valid_df.append(new_val_row,ignore_index=True)

Valid_df.head()
#Shuffling the data

Valid_df=Valid_df.sample(frac=1)
Valid_df.tail()
# Getting  all the file paths from validation dataframe

All_val_files=[fname for fname in Valid_df["ID"]]
All_val_files[:5]
val_labels=Valid_df["Target"].to_numpy()

val_labels[:10]
val_true_labels=np.unique(val_labels)

len(val_true_labels)
boolean_val_labels=[labels==val_true_labels for labels in val_labels ]

boolean_val_labels[:10]
# Splitting data into training and validation set

X_train,y_train=All_training_files[:NUM_IMAGES],boolean_labels[:NUM_IMAGES]

X_val,y_val=All_val_files[:2000],boolean_val_labels[:2000]
len(X_train)
IMG_SIZE=224

# Creating a function that can preprocess the data

def preprocess_data(image_path,img_size=IMG_SIZE):

    #Reading the image path

    image=tf.io.read_file(image_path)

    #Turning the image into numerical tensors of colour channel

    image=tf.image.decode_jpeg(image,channels=3)

    #Converting the colour channels from 0-255 values to 0-1

    image=tf.image.convert_image_dtype(image,tf.float32)

    #Resize our images into the desired value 224

    image=tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])

    return image
#Function to return a tuple of preprocessed image in form tensors and their respective labels

def get_image_label(image_path,label):

    image=preprocess_data(image_path)

    return image,label
# Function to change all our X and y into data batches

BATCH_SIZE=32

def create_data_batches(X,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):

    # if it is training data then there won't be lables

    if test_data==True:

        print("Create test data batches....")

        data=tf.data.Dataset.from_tensor_slices((tf.context(X)))

        data_batch=data.map(preprocess_data).batch(BATCH_SIZE)

        return data_batch

    # if it is valid data

    elif valid_data==True:

        print("Create validation data batches.....")

        data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))

        data_batch=data.map(get_image_label).batch(BATCH_SIZE)

        return data_batch

    # if it is training data

    else:

        print("Creating training data batches....")

        data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))

        data=data.map(get_image_label)

        data_batch=data.batch(BATCH_SIZE)

        return data_batch

        
len(X_train),len(X_val)
# Create training and validation data batches

train_data=create_data_batches(X_train,y_train)

val_data=create_data_batches(X_val,y_val,valid_data=True)

train_data.element_spec,val_data.element_spec
import matplotlib.pyplot as plt

def show_25_images(images,label):

    # setup a figure 

    plt.figure(figsize=(10,10))

    # loop through 25 images

    for i in range(25):

        # Create subplots (5 rows,5 columns)

        ax=plt.subplot(5,5,i+1)

        # Display an image

        plt.imshow(images[i])

        # Add image label as title

        plt.title(true_labels[label[i].argmax()])

        # Turn the grid lines off

        plt.axis("off")
len(val_data)
# Unbatch the data using as_numpy_iterator

train_images,train_labels=next(train_data.as_numpy_iterator())

# Now lets visualize the images in the training batch

show_25_images(train_images,train_labels)
# Now lets visualize the images in the validation batch

val_images,val_labels=next(val_data.as_numpy_iterator())

show_25_images(val_images,val_labels)




# Setting up a input shape

INPUT_SHAPE=[None,IMG_SIZE,IMG_SIZE,3]



# Settion up a output shape

OUTPUT_SHAPE=len(true_labels)



# Model URL

MODEL_URL="https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
# Building a keras model

def create_model(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,model_url=MODEL_URL):

    model=tf.keras.Sequential([

        hub.KerasLayer(MODEL_URL),

        tf.keras.layers.Dense(units=OUTPUT_SHAPE,

                             activation="softmax")

    ])

    

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),

      optimizer=tf.keras.optimizers.Adam(),

      metrics=["accuracy"]



  )



    # Build the model

    model.build(INPUT_SHAPE)



    return model
model=create_model()

model.summary()
%load_ext tensorboard
# Create a early stopping callback

early_stopping=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3)
NUM_EPOCHS=100
import datetime

# Create a function to build a TensorBoard Callback

def create_tensorboard_callback():

    # Create a log directory for storing TensorBoard logs

    logdir=os.path.join("./kaggle/working/",

                      # Make it so the logs gets tracked whenever we run the expirement

                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return tf.keras.callbacks.TensorBoard(logdir) 
# Build a function to train a model and return a trained model

def train_model():

    # Create a model

    model=create_model()



    # Create a new session everytime we train a model

    tensorboard=create_tensorboard_callback()



    # Fit the model to the data passing it the callbacks we created

    model.fit(x=train_data,

            epochs=NUM_EPOCHS,

            validation_data=val_data,

            validation_freq=1,

            callbacks=[tensorboard,early_stopping])

    return model

  
model=train_model()
len(val_data)
# Make Predictions on the validation data

predictions=model.predict(val_data,verbose=1)

predictions[:10]
predictions.shape
print(predictions[0])

print(f"Max value (probability of prediction): {np.max(predictions[0])}") # the max probability value predicted by the model

print(f"Max index: {np.argmax(predictions[0])}") # the index of where the max value in predictions[0] occurs

print(f"Predicted label: {true_labels[np.argmax(predictions[0])]}")


# Turn prediction probabilities into their respective label (easier to understand)

def get_pred_label(prediction_probabilities):

  """

  Turns an array of prediction probabilities into a label.

  """

  return true_labels[np.argmax(prediction_probabilities)]



# Get a predicted label based on an array of prediction probabilities

pred_label = get_pred_label(predictions[0])

pred_label
len(val_data)
# Create a function to unbatch a batched dataset

def unbatchify(data):

    

    images = []

    labelss = []

    # Loop through unbatched data

    for image, label in data.unbatch().as_numpy_iterator():

        images.append(image)

        labelss.append(true_labels[np.argmax(label)])

    return images, labelss



# Unbatchify the validation data

val_images, val_labels = unbatchify(val_data)

val_images[0], val_labels[0]
len(val_images)
def plot_pred(prediction_probabilities, labels, images, n=0):

    

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



    plt.title("{} {:2.0f}% ({})".format(pred_label,

                                      np.max(pred_prob)*100,

                                      true_label),

                                      color=color)
len(predictions)
len(val_labels)
plot_pred(prediction_probabilities=predictions,

          labels=val_labels,

          images=val_images)
model.evaluate(val_data)

len(X)
X=All_training_files[:90000]

y=boolean_labels[:90000]
X_val,y_val=All_val_files,boolean_val_labels
# Creating full training and validation data batches

Full_training_data=create_data_batches(X,y)

Full_validation_data=create_data_batches(X_val,y_val,valid_data=True)
len(Full_training_data)
full_model=create_model()
# Create full model callbacks



# TensorBoard callback

full_model_tensorboard = create_tensorboard_callback()



# Early stopping callback

full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",

                                                             patience=3)
NUM_EPOCHS
full_model.fit(x=Full_training_data,

            epochs=NUM_EPOCHS,

            validation_data=Full_validation_data,

            validation_freq=1,

            callbacks=[early_stopping])

    
# Storing all the test file paths 

female_test_files=[]

female_test=list()

male_test=list()

female_test_files=os.listdir(Test_female_path)

male_test_files=os.listdir(Test_male_path)

for i in range(len(os.listdir(Test_female_path))):

    female_test.append(Test_female_path+"/"+str(female_test_files[i]))

for i in range(len(os.listdir(Test_male_path))):

    male_test.append(Test_male_path+"/"+str(male_test_files[i]))

  
#Creating a test Dataframe

Test_df=pd.DataFrame()

Test_df=pd.DataFrame({"ID":female_test,"Target":"Female"})

new_test_row=pd.DataFrame({"ID":male_test,"Target":"Male"})

Test_df=Test_df.append(new_test_row,ignore_index=False)

#Shuffling the data frame

Test_df=Test_df.sample(frac=1)
Test_df.head()
len(Test_df)
X_test,y_test=Test_df["ID"][:10000],Test_df["Target"][:10000]
len(X_test)
test_data=create_data_batches(X_test,y_test)
# Making Predictions on the entire data set

test_predictions=full_model.predict(test_data,verbose=1)
test_predictions.shape
test_images,test_labels=unbatchify(test_data)
plot_pred(prediction_probabilities=test_predictions,

          labels=test_labels,

          images=test_images,n=200)
def show_test_25_images(images,label,predictions):

    # setup a figure 

    plt.figure(figsize=(10,10))

    # loop through 25 images

    for i in range(25):

        # Create subplots (5 rows,5 columns)

        ax=plt.subplot(5,5,i+1)

        # Display an image

        plot_test_pred(prediction_probabilities=predictions, labels=label, images=images, n=i)

        

        # Turn the grid lines off

        plt.axis("off")
plot_pred(prediction_probabilities=test_predictions, labels=test_labels, images=test_images, n=0)
def plot_test_pred(prediction_probabilities, labels, images, n=0):

    

    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

  

    # Get the pred label

    pred_label = get_pred_label(pred_prob)

  

  # Plot image & remove ticks

    plt.imshow(image)

    plt.xticks([])

    plt.yticks([])



  # Change the color of the title depending on if the prediction is right or wrong

    

    color = "green"



    plt.title("{} {:2.0f}%".format(pred_label,

                                      np.max(pred_prob)*100),

                                      color=color)
show_test_25_images(images=test_images,label=test_labels,predictions=test_predictions)