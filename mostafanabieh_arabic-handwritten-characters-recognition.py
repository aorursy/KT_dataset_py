# Import main libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import libraries needed for reading image and processing it
import csv
from PIL import Image
from scipy.ndimage import rotate

# Pretty display for notebooks
%matplotlib inline
# Training letters images and labels files
letters_training_images_file_path = "../input/ahcd1/csvTrainImages 13440x1024.csv"
letters_training_labels_file_path = "../input/ahcd1/csvTrainLabel 13440x1.csv"
# Testing letters images and labels files
letters_testing_images_file_path = "../input/ahcd1/csvTestImages 3360x1024.csv"
letters_testing_labels_file_path = "../input/ahcd1/csvTestLabel 3360x1.csv"

# Loading dataset into dataframes
training_letters_images = pd.read_csv(letters_training_images_file_path, header=None)
training_letters_labels = pd.read_csv(letters_training_labels_file_path, header=None)
testing_letters_images = pd.read_csv(letters_testing_images_file_path, header=None)
testing_letters_labels = pd.read_csv(letters_testing_labels_file_path, header=None)

# print statistics about the dataset
print("There are %d training arabic letter images of 32x32 pixels." %training_letters_images.shape[0])
print("There are %d testing arabic letter images of 32x32 pixels." %testing_letters_images.shape[0])
training_letters_images.head()
def convert_values_to_image(image_values, display=False):
    image_array = np.asarray(image_values)
    image_array = image_array.reshape(32,32).astype('uint8')
    # The original dataset is reflected so we will flip it then rotate for a better view only.
    image_array = np.flip(image_array, 0)
    image_array = rotate(image_array, -90)
    new_image = Image.fromarray(image_array)
    if display == True:
        new_image.show()
    return new_image
convert_values_to_image(training_letters_images.loc[0], True)
training_letters_images_scaled = training_letters_images.values.astype('float32')/255
training_letters_labels = training_letters_labels.values.astype('int32')
testing_letters_images_scaled = testing_letters_images.values.astype('float32')/255
testing_letters_labels = testing_letters_labels.values.astype('int32')
print("Training images of letters after scaling")
print(training_letters_images_scaled.shape)
training_letters_images_scaled[0:5]
from keras.utils import to_categorical

# one hot encoding
# number of classes = 28 (arabic alphabet classes)
number_of_classes = 28

training_letters_labels_encoded = to_categorical(training_letters_labels-1, num_classes=number_of_classes)
testing_letters_labels_encoded = to_categorical(testing_letters_labels-1, num_classes=number_of_classes)
print(training_letters_labels_encoded)
# reshape input letter images to 32x32x1
training_letters_images_scaled = training_letters_images_scaled.reshape([-1, 32, 32, 1])
testing_letters_images_scaled = testing_letters_images_scaled.reshape([-1, 32, 32, 1])

print(training_letters_images_scaled.shape, training_letters_labels_encoded.shape, testing_letters_images_scaled.shape, testing_letters_labels_encoded.shape)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense

def create_model(optimizer='adam', kernel_initializer='he_normal', activation='relu'):
    # create model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', input_shape=(32, 32, 1), kernel_initializer=kernel_initializer, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    #Fully connected final layer
    model.add(Dense(28, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    return model
model = create_model()
model.summary()
import pydot
from keras.utils import plot_model

plot_model(model, to_file="model.png", show_shapes=True)
from IPython.display import Image as IPythonImage
display(IPythonImage('model.png'))
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# define the grid search parameters
optimizer = ['RMSprop', 'Adam', 'Adagrad', 'Nadam']
kernel_initializer = ['normal', 'uniform']
activation = ['relu', 'linear', 'tanh']

param_grid = dict(optimizer=optimizer, kernel_initializer=kernel_initializer, activation=activation)

# count number of different parameters values combinations
parameters_number = 1
for x in param_grid:
    parameters_number = parameters_number * len(param_grid[x]) 
print("Number of different parameter combinations = {}".format(parameters_number))
epochs = 5
batch_size = 20 # 20 divides the training data samples

model = create_model(optimizer='Adam', kernel_initializer='uniform', activation='relu')

from keras.callbacks import ModelCheckpoint  

# using checkpoints to save model weights to be used later instead of training again on the same epochs.
checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
history = model.fit(training_letters_images_scaled, training_letters_labels_encoded, 
                    validation_data=(testing_letters_images_scaled, testing_letters_labels_encoded),
                    epochs=15, batch_size=20, verbose=1, callbacks=[checkpointer])
import matplotlib.pyplot as plt

def plot_loss_accuracy(history):
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'],'r',linewidth=3.0)
    plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16) 
plot_loss_accuracy(history)
model.load_weights('weights.hdf5')
# Final evaluation of the model
metrics = model.evaluate(testing_letters_images_scaled, testing_letters_labels_encoded, verbose=1)
print("Test Accuracy: {}".format(metrics[1]))
print("Test Loss: {}".format(metrics[0]))
epochs = 30
batch_size = 20

checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

history = model.fit(training_letters_images_scaled, training_letters_labels_encoded, 
                    validation_data=(testing_letters_images_scaled, testing_letters_labels_encoded),
                    epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[checkpointer])
          
model.load_weights('weights.hdf5')
plot_loss_accuracy(history)
plot_loss_accuracy(history)
# Final evaluation of the model
metrics = model.evaluate(testing_letters_images_scaled, testing_letters_labels_encoded, verbose=1)
print("Test Accuracy: {}".format(metrics[1]))
print("Test Loss: {}".format(metrics[0]))
from keras.models import model_from_yaml
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# compile the loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
def get_predicted_classes(model, data, labels=None):
    image_predictions = model.predict(data)
    predicted_classes = np.argmax(image_predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    return predicted_classes, true_classes, image_predictions
from sklearn.metrics import classification_report

def get_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
!pip install Pillow

y_pred, y_true, image_predictions = get_predicted_classes(model, testing_letters_images_scaled, testing_letters_labels_encoded)
get_classification_report(y_true, y_pred)
import cv2 as cv2
from matplotlib import pyplot as plt
import pytesseract

img = cv2.imread("../input/ahcd1/Test Images 3360x32x32/test/id_1037_label_15.png",0)
#cv.rectangle(img,(29,2496),(604,2992),(255,0,0),5)
plt.imshow(img)
img = cv2.imread("../input/ahcd1/Test Images 3360x32x32/test/id_1051_label_22.png",0)
img = cv2.resize(img,(32,32))
image = np.expand_dims(img, axis=0)


img = np.reshape(image,[1,32,32,1])

classes = model.predict_classes(img)
classes
import cv2 as cv2
from matplotlib import pyplot as plt
import pytesseract

img = cv2.imread("../input/ahcd1/Test Images 3360x32x32/test/id_1037_label_15.png",1)
#cv.rectangle(img,(29,2496),(604,2992),(255,0,0),5)
plt.imshow(img)
errors = (y_pred - y_true != 0)


Y_pred_classes_errors = y_pred[errors]
Y_pred_errors = image_predictions[errors]
Y_true_errors = y_true[errors]
X_val_errors = testing_letters_images_scaled[errors]


def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            
            image_array = img_errors[error]
            image_array = np.flip(image_array, 0)
            image_array = rotate(image_array, -90)
            
            
            ax[row,col].imshow((image_array).reshape((32,32)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted letters
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
fig = plt.figure(0, figsize=(18,18))
indices = np.random.randint(0, testing_letters_labels.shape[0], size=49)
y_pred = np.argmax(model.predict(training_letters_images_scaled), axis=1)

for i, idx in enumerate(indices):
    plt.subplot(7,7,i+1)
        
    image_array = training_letters_images_scaled[idx][:,:,0]
    image_array = np.flip(image_array, 0)
    image_array = rotate(image_array, -90)
       
    plt.imshow(image_array, cmap='gray')
    plt.title("Pred: {} - Label: {}".format(y_pred[idx], (training_letters_labels[idx] -1)))
    plt.xticks([])
    plt.yticks([])
plt.show()

