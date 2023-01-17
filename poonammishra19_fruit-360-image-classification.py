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
import matplotlib.pyplot as plt
import cv2 
from keras.utils import np_utils
from sklearn.datasets import load_files       
import tensorflow as tf 
from tensorflow import keras 
from glob import glob 
import os
print(os.listdir("../input/fruits360v14/"))

# Any results you write to the current directory are saved as output
#Loading the train images into the root directory for training the model
root_dir = '../input/fruits360v14/fruits-360-v-14/Training'
#We will be selecting around 30 images for visualization purposes
rows = 14
cols = 6
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(12, 20))
fig.suptitle('Images selected at random from multiple classes', fontsize=20)
sorted_food_dirs = sorted(os.listdir(root_dir))
for i in range(rows):
    for j in range(cols):
        try:
            food_dir = sorted_food_dirs[i*cols + j]
        except:
            break
        all_files = os.listdir(os.path.join(root_dir, food_dir))
        rand_img = np.random.choice(all_files)
        img = plt.imread(os.path.join(root_dir, food_dir, rand_img))
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax[i][j].text(0, -20, food_dir, size=10, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
        
#Visualizsing the results
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#Now we will load the actual train and test datasets and define a function for the same
#Defining the function
def load_dataset(path):
    data = load_files(path)
    fruit_files = np.array(data['filenames'])
    fruit_targets = np_utils.to_categorical(np.array(data['target']), 60)
    return fruit_files, fruit_targets

# Setting uo train and test data into variables from input dataset
train_files, train_targets = load_dataset('../input/fruits360v14/fruits-360-v-14/Training/')
test_files, test_targets = load_dataset('../input/fruits360v14/fruits-360-v-14/Validation/')


# loading a directory with the names of the fruits
fruit_names = [item[9:] for item in sorted(glob("Training/*"))]

# Basic description of the dataset
print('Total images of fruits in the dataset are %s.\n' % len(np.hstack([train_files, test_files])))
print('No. of training images are %d .' % len(train_files))
print('No of test images are %d .'% len(test_files))
#Inclusion of images using glob
import glob

#initializing empty lists fot training image and training label
training_fruit_img = []
training_label = []

for dir_path in glob.glob("../input/fruits360v14/fruits-360-v-14/Training/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64,64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_fruit_img.append(img)
        training_label.append(img_label)
training_fruit_img = np.array(training_fruit_img)
training_label = np.array(training_label)
len(np.unique(training_label))
#initializing empty lists fot test image and test label
test_fruit_img = []
test_label = []
for dir_path in glob.glob("../input/fruits360v14/fruits-360-v-14/Validation/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64,64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_fruit_img.append(img)
        test_label.append(img_label)
test_fruit_img = np.array(test_fruit_img)
test_label = np.array(test_label)
len(np.unique(test_label))
#Labeling images uniquely with individual ids
label_to_id = {v : k for k, v in enumerate(np.unique(training_label))}
id_to_label = {v : k for k, v in label_to_id.items()}
training_label_id = np.array([label_to_id[i] for i in training_label])
test_label_id = np.array([label_to_id[i] for i in test_label])
test_label_id
#Normalizing the images
training_fruit_img, test_fruit_img = training_fruit_img / 255.0, test_fruit_img / 255.0 
#Visualising a sample random image after normalising
plt.imshow(training_fruit_img[12])
#Importing cv2 library - This is a library of Python bindings designed to solve computer vision problems
#All the OpenCV array structures are converted to and from Numpy arrays
#This also makes it easier to integrate with other libraries that use Numpy such as SciPy and Matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt

#imread method loads an image from the specified file
img = cv2.imread(train_files[0])
color = ('b','g','r')
for i,col in enumerate(color):
#To deal with an image which  consist of intensity distribution of pixels where pixel value varies
#We use cv2's inbuild function calcHist to to plot the histogram.
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
#Instanitating the model
model = keras.Sequential()
#Conv layer 1
model.add(keras.layers.Conv2D(16, (3, 3), input_shape = (64,64, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

#Conv layer 2
model.add(keras.layers.Conv2D(32, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

#Conv layer 3
model.add(keras.layers.Conv2D(32, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

#Conv layer 4
model.add(keras.layers.Conv2D(64, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Flatten())

#Hidden layer
model.add(keras.layers.Dense(256, activation = "relu"))

#Output layer
model.add(keras.layers.Dense(75, activation = "softmax"))
#Compiling the model
model.compile(loss = "sparse_categorical_crossentropy", optimizer = keras.optimizers.Adamax(), metrics = ['accuracy'])
tensorboard = keras.callbacks.TensorBoard(log_dir = "./Graph", histogram_freq = 0, write_graph = True, write_images = True)
model.summary()
#Fitting the model
#Hpyer parameters
BATCH_SIZE = 128
EPOCHS = 10
History = model.fit(training_fruit_img, training_label_id, batch_size = BATCH_SIZE, epochs = EPOCHS, callbacks = [tensorboard])
#Training accuracy
loss, accuracy = model.evaluate(training_fruit_img,training_label_id)
print("\n\nLoss:", loss)
print("Accuracy:", accuracy)
#model.save("model.h5")
#Test accuracy
loss, accuracy = model.evaluate(test_fruit_img, test_label_id)
print("\n\nLoss:", loss)
print("Accuracy:", accuracy)
model.save("model.fruit")
#Plotting the results
N = 10
plt.figure()
plt.plot(np.arange(0, N), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), History.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
predictions = model.predict(test_fruit_img)
plt.figure(figsize = (20, 20))
for i in range(30):
    plt.subplot(9,5, i + 1)
    plt.xlabel("{}".format(id_to_label[np.argmax(predictions[i])]))
    plt.imshow(test_fruit_img[i])
