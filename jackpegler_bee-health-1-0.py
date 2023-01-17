#####################################
# Libraries
#####################################
# Common libs
import pandas as pd
import numpy as np
import missingno as msno
import sys
import os
import random

# Image processing
import imageio
import skimage.io
import skimage.transform

# Charts
import matplotlib.pyplot as plt
import seaborn as sns


# ML

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from keras.models import Sequential
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPool2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


#####################################
# Settings
#####################################

# Set random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(42)

# Global variables
img_folder='../input/bee_imgs/bee_imgs/'

# plotting
plt.style.use('ggplot')
### READ IN BEE DATA AND FORMAT SOME COLUMNS
bee_data = pd.read_csv("../input/bee_data.csv", 
            parse_dates={'datetime': ['date', 'time']}, 
            dtype={'subspecies':'category', 'health':'category','caste':'category'})

bee_data.head()
f, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

bee_data.subspecies.value_counts().plot(kind='bar',ax=ax[0, 0])
ax[0,0].set_ylabel('Count')
ax[0,0].set_title('Subspecies')

bee_data.location.value_counts().plot(kind='bar', ax=ax[0, 1])
ax[0,1].set_title('Location')
ax[0,1].set_ylabel('Count')

bee_data.caste.value_counts().plot(kind='bar', ax=ax[1, 0])
ax[1,0].set_title('Caste')
ax[1,0].set_ylabel('Count')

bee_data.health.value_counts().plot(kind='bar', ax=ax[1,1])
ax[1,1].set_title('Health')
ax[1,1].set_ylabel('Count')

f.subplots_adjust(hspace=0.7)
f.tight_layout()
plt.show()
### RECORDS WE HAVE IN THE CSV FILE
print("Number of records in CSV: {}".format(bee_data.shape[0]))

### RECORDS WITH EXISTING PHOTOS AVAILABLE
img_exists = bee_data['file'].apply(lambda f: os.path.exists(img_folder + f))
bee_data = bee_data[img_exists]
print("Number of records with a photo available: {}".format(bee_data.shape[0]))

### PLOT MATRIX TO SEE IF HAVE ANY NaN RECORDS
msno.matrix(bee_data)
### FUNCTION FOR PLOTTING IMAGES

def plot_image_grid(data, W, H, title="ADD A PLOT TITLE"):
    
    #### VIEW IMAGES IN A GRID
    W_grid = W
    L_grid = H

    fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,10))

    axes = axes.ravel() #flatten the 5x2 matrix into 10 array

    for i in np.arange(0, W_grid * L_grid): # create evenly spaces variable

        # select a random number
        index = random.choice(data.index)

        # read and display an image with the selected index
        axes[i].imshow(skimage.io.imread(img_folder + data['file'][index]))
        axes[i].set_title(data['health'][index], fontsize = 8)
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
healthy = bee_data[bee_data['health'] == 'healthy']

plot_image_grid(data = healthy, W=5, H=2, title="HEALTHY BEES")
### first lets get the categories of "unhealthy"
ailments = bee_data['health'].cat.categories
ailments = [a for a in ailments if a != 'healthy']

print('Non healthy bees fall into these categories: ')
print(ailments)
### NOW LETS SAMPLE ACROSS EACH CATEGORY AND PLOT EXAMPLE IMAGES
non_healthy = pd.DataFrame()
for sickness in ailments:
    non_healthy = non_healthy.append(bee_data[bee_data['health'] == sickness].sample(2))
plot_image_grid(data = non_healthy, W=5, H=2, title="SICK BEES")
image_properties = pd.DataFrame()

for file in bee_data['file']:  
    h,w,c = np.array(skimage.io.imread(img_folder + file)).shape
    image_properties = image_properties.append([[h,w,c]])
image_properties.describe()
### SET THESE HERE SO EASY TO CHANGE LATER IF WE WANT TO
image_height = 100
image_width = 100
image_channels = 3
def split_and_balance(dataset, balance_col):
    """
    1. Split into our different data sets
    2. Balance the training set for balance_col
    """
    ### 70% data for train
    data_train, data_dev = train_test_split(dataset, test_size=0.3, random_state=42)
    ### 15% each for validation and testing
    data_dev, data_test = train_test_split(data_dev, test_size=0.5, random_state=42)
    
    print ("number of training examples = " + str(data_train.shape[0]))
    print ("number of dev examples = " + str(data_dev.shape[0]))
    print ("number of test examples = " + str(data_test.shape[0]))
    
    
    #### BALANCING - LETS NOT DO THIS ON FIRST RUN
    #### ADD IT IN AS EXCERSIZE TO IMPROVE MODEL LATE
    
    #### ALSO WOULD BE GREAT TO PLOT THE BALANE BEFORE AND AFTER
    
    return data_train, data_dev, data_test
### LOADING IMAGES
def load_images(file):
    """
    Given a file name it will load the images in the correct size.
    """
    image = skimage.io.imread(img_folder + file)
    image = skimage.transform.resize(image, (image_width, image_height), mode='reflect')
    return image[:,:,:image_channels]
def data_prep(data_train, data_dev, data_test, target_col, normalize=True):
    """
    1. Loads images into the train, dev, test sets
    2. OHE the target col for train, dev and test sets
    3. Normalize images
    """
    num_cat = data_train[target_col].cat.categories
    
    ### TRAINING DATA
    X_train = np.stack(data_train['file'].apply(load_images))
    y_train = pd.get_dummies(data_train[target_col], drop_first=False)
    
    ### DEV DATA
    X_dev = np.stack(data_dev['file'].apply(load_images))
    y_dev = pd.get_dummies(data_dev[target_col], drop_first=False)
    
    ### TEST DATA
    X_test = np.stack(data_test['file'].apply(load_images))
    y_test = pd.get_dummies(data_test[target_col], drop_first=False)
    
    ### NORMALIZE THE FEATURES (ASSUME 255)
    if normalize:
        X_train = X_train / 255
        X_dev = X_dev / 255
        X_test = X_test / 255
        
    #### NOT DOING GENERATOR TO TRANSFORM  / AUGMENT THE DATA
    #### ADD THIS IN LATER ITERATION
    
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of dev examples = " + str(X_dev.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    
    print ("\nnumber of clategories for classifier = " + str(num_cat.shape[0]))
    print(num_cat)
    
    print ("\nX_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    
    print ("\nX_dev shape: " + str(X_dev.shape))
    print ("\nY_dev shape: " + str(y_dev.shape))
    
    print ("\nX_test shape: " + str(X_test.shape))
    print ("\nY_test shape: " + str(y_test.shape))
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test
data_train, data_dev, data_test = split_and_balance(bee_data, 'health')
data_train.dtypes
data_train['health'].cat.categories
X_train, X_dev, X_test, y_train, y_dev, y_test = data_prep(data_train, data_dev, data_test, 'health', normalize=False)
def BeesModel():

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    model1=Sequential()
    model1.add(Conv2D(6, kernel_size=3, input_shape=(image_width, image_height, 3), activation='relu', padding='same'))
    model1.add(MaxPool2D(2))
    model1.add(Conv2D(12, kernel_size=3, activation='relu', padding='same'))
    model1.add(Flatten())
    model1.add(Dense(y_train.columns.size, activation='softmax'))

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    return model1
### INIITATE THE MODEL
beesModel = BeesModel()
### COMPILE THE MODEL
beesModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
### TRAIN THE MODEL
run_history_1 = beesModel.fit(x = X_train, y = y_train, validation_data=(X_dev, y_dev), epochs = 10,batch_size = 8, verbose=1)
# list all data in history
print(run_history_1.history.keys())

# summarize history for accuracy
plt.plot(run_history_1.history['acc'])
plt.plot(run_history_1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(run_history_1.history['loss'])
plt.plot(run_history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
### function for plotting scores

def how_did_it_do(ml_model, X_dev, y_dev, use_model_dot_score=True, cf_matrix=True):
    ## predict from dev set
    y_pred = ml_model.predict(X_dev)
    y_pred = np.argmax(y_pred, axis=1)
    y_dev = np.argmax(y_dev.values, axis=1)
    print("performance on X_dev:")
    
    if use_model_dot_score:
        # Accuracy
        print("\nAccuracy:")
        acc = round(ml_model.score(X_dev, y_dev), 3)
        print(acc)
    else:
        print("\nAccuracy score:")
        acc = round(accuracy_score(y_dev, y_pred), 3)
        print(acc)    

    # of predicted +ve, how many correct
    print("Precision score:")
    prec = round(precision_score(y_dev, y_pred, average='macro'), 3)
    print(prec)


    # of all actual +ve how many did we get
    print("Recall score:")
    rec = round(recall_score(y_dev, y_pred, average='macro'), 3)
    print(rec)

    # f1 combines
    print("Global F1 score:")
    f1 = round(f1_score(y_dev, y_pred, average='macro'), 3)
    print(f1)
    
    ### plot confusion matrix if needed
    if cf_matrix:
        cm = confusion_matrix(y_dev, y_pred.round())
        df_cm = pd.DataFrame(cm, index = (0,1,2,3,4,5), columns=(0,1,2,3,4,5))
        plt.figure(figsize = (10,7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot = True, fmt='g')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
how_did_it_do(beesModel, X_test, y_test, use_model_dot_score=False, cf_matrix=True)
data_dev['health'].value_counts()
data_dev['health'].cat.categories
y_pred = beesModel.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
target_names=y_test.columns
y_test = np.argmax(y_test.values, axis=1)
print(metrics.classification_report(y_test, y_pred, target_names=target_names))




#### VIEW IMAGES IN A GRID
W_grid = 5
L_grid = 2

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,10))

axes = axes.ravel() #flatten the 5x2 matrix into 10 array

for i in np.arange(0, W_grid * L_grid): # create evenly spaces variable

    # select a random number
    #index = random.choice(data.index)

    # read and display an image with the selected index
    axes[i].imshow(X_test[i])
    title = str(np.argmax(y_test.values, axis=1)[i]) + " :" + str(y_pred[i])
    axes[i].set_title(title , fontsize = 8)
    axes[i].axis('off')

plt.suptitle("EXAMPLES FROM TEST SET")
plt.tight_layout()
X_test[0:5].shape
