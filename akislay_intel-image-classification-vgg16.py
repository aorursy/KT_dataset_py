# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%reset -f
## 1. Call libraries
import numpy as np
# 1.1 Classes for creating models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense

# 1.2 Class for accessing pre-built models
from tensorflow.keras import applications

# 1.3 Class for generating infinite images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1.4 Miscelleneous
import matplotlib.pyplot as plt
import time, os
# Traind Data Path
train_data_dir = "..//input//intel-image-classification//seg_train//seg_train"
# Validation Dara Path
validation_data_dir = "..//input//intel-image-classification//seg_test//seg_test"


# 2.2 dimensions of the images during image generation
img_width, img_height = 150,150
# 2.3 Batch Size Selection
nb_train_samples, nb_validation_samples = 14000, 3000
# 2.4 Epochs Batch Size
batch_size = 50   
# 2.5 File to which transformed bottleneck features for train data wil be stored
bf_filename = '..//working//bottleneck_features_train.npy'

# 2.6 File to which transformed bottleneck features for validation data wil be stored
val_filename = '..//working//bottleneck_features_validation.npy'

## 3. Data augmentation

# 3.1 Instanstiate an image data generator: Needd to feed into the model
#     Only normalization & nothing else like flipping, rotation etc;

datagen_train = ImageDataGenerator(rescale=1. / 255)

# 3.2 Configure datagen_train further
#     Datagenerator is configured twice. 
#    Ist configuration is about image manipulation features
#    IInd configuration is regarding data source, data classes, batchsize  etc

generator_tr = datagen_train.flow_from_directory(
              directory = train_data_dir,             # Path to target train directory.
              target_size=(img_width, img_height),    # Dimensions to which all images will be resized.
              batch_size=batch_size,                  # At a time so many images will be output
              class_mode=None,                        # Return NO labels along with image data
              shuffle=False                           # Default shuffle = True
              )
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
img = load_img(train_data_dir + "/buildings/1001.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()
# 3.3. Generator for validation data.
#      Initialize ImageDataGenerator object once more
#      shuffle = False => Sort data in alphanumeric order
datagen_val = ImageDataGenerator(rescale=1. / 255)
generator_val = datagen_val.flow_from_directory(
                                          validation_data_dir,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False   # Default shuffle = True

                                          )
# 4. Buld VGG16 network model with 'imagenet' weights
model = applications.VGG16(
                       include_top=False,
                       weights='imagenet',
                       input_shape=(img_width, img_height,3)
                      )
model.summary()
# 4.1 Feed images through VGG16 model in batches (steps) and make 'predictions'.

start = time.time()
bottleneck_features_train = model.predict_generator(
                                                    generator = generator_tr,
                                                    steps = nb_train_samples // batch_size,
                                                    verbose = 1
                                                    )
end = time.time()
print("Time taken: ",(end - start)/60, "minutes")
# 4.2   Similarly, make predictions for validation data and extract features

start = time.time()
bottleneck_features_validation = model.predict_generator(
                                                         generator = generator_val,
                                                         steps = nb_validation_samples // batch_size,
                                                         verbose = 1
                                                         )

end = time.time()
print("Time taken: ",(end - start)/60, "minutes")
# 5.1 Save the train features

if os.path.exists(bf_filename):
    os.system('rm ' + bf_filename)

# 5.2 Next save the train-features
np.save(open(bf_filename, 'wb'), bottleneck_features_train)


# 5.3 Save validation features from model
if os.path.exists(val_filename):
    os.system('rm ' + val_filename)

np.save(open(val_filename, 'wb'), bottleneck_features_validation)
# 5.4 Save the Training label details in file

label_filename = '..//working//bottleneck_features_train_label.npy'
if os.path.exists(label_filename):
    os.system('rm ' + label_filename)

np.save(open(label_filename, 'wb'), generator_tr.labels)

# 5.5 Save the Validation label details in file

val_label_filename = '..//working//bottleneck_features_validation_label.npy'
if os.path.exists(val_label_filename):
    os.system('rm ' + val_label_filename)

np.save(open(val_label_filename, 'wb'), generator_val.labels)
os.listdir()
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Softmax
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras import applications
import matplotlib.pyplot as plt
import time, os
# 2. Hyperparameters/Constants
# 2.1 Dimensions of the images.
img_width, img_height = 150,150  # 150, 150
nb_train_samples = 14000
nb_validation_samples = 3000
epochs = 50
batch_size = 50
num_classes = 6
# 2.2 Bottleneck features for train data
bf_filename = '..//working//bottleneck_features_train.npy'
# Validation-bottleneck features filename
val_filename = '..//working//bottleneck_features_validation.npy'
# Training Label filename
label_filename = '..//working//bottleneck_features_train_label.npy'

# Validation Label filename
val_label_filename = '..//working//bottleneck_features_validation_label.npy'

# 2.3 File to which FC model weights could be stored
top_model_weights_path = "..//working//bottleneck_fc_model.h5"
# Load first train features
train_data_features = np.load(open(bf_filename,'rb'))

train_data_features.shape

# Load Train label
train_data_label = np.load(open(label_filename,'rb'))
train_data_label.shape
# 3. Load Validation label
val_data_label = np.load(open(val_label_filename,'rb'))
val_data_label.shape
train_labels =train_data_label[:14000]
train_labels.shape
# One hot encode training the labels
train_labels_cat = to_categorical(train_labels , )
train_labels_cat.shape
# One hot encode the Validation labels
val_labels_cat = to_categorical(val_data_label , )
val_labels_cat.shape
# Load Validation features
validation_data_features = np.load(open(val_filename,'rb'))
validation_data_features.shape 
# Adding Model to VGG16
model = Sequential()
model.add(Flatten(input_shape=train_data_features.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
# For Categorical Classification
model.compile(
              optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
epochs=50
train_data_features.shape

train_labels_cat.shape


validation_data_features.shape

val_labels_cat.shape
# Fit model and make predictions on validation dataset
start = time.time()
history = model.fit(train_data_features, train_labels_cat,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data_features, val_labels_cat),
                    verbose =1
                   )
end = time.time()
print("Time taken: ",(end - start)/60, "minutes")
history.history.keys()   
len(history.history['accuracy'])
len(history.history['val_accuracy'])
model.save_weights(top_model_weights_path)
def plot_learning_curve():
    val_acc = history.history['val_accuracy']
    tr_acc=history.history['accuracy']
    epochs = range(1, len(val_acc) +1)
    plt.plot(epochs,val_acc, 'b', label = "Validation accu")
    plt.plot(epochs, tr_acc, 'r', label = "Training accu")
    plt.title("Training and validation accuracy")
    plt.xlabel("epochs-->")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
plot_learning_curve()