import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
path = "../input/covid19-radiography-database/COVID-19 Radiography Database"
labels = ["COVID-19", "NORMAL", "Viral Pneumonia"]
# Randomly view 5 images in each category

fig, axs = plt.subplots(len(labels), 5, figsize = (15, 15))

class_len = {}
for i, c in enumerate(labels):
    class_path = os.path.join(path, c)
    all_images = os.listdir(class_path)
    sample_images = random.sample(all_images, 5)
    class_len[c] = len(all_images)
    
    for j, image in enumerate(sample_images):
        img_path = os.path.join(class_path, image)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axs[i, j].imshow(img)
        axs[i, j].set(xlabel = c, xticks = [], yticks = [])

fig.tight_layout()
# Make a pie-chart to visualize the percentage contribution of each category.
fig, ax = plt.subplots()
ax.pie(
    class_len.values(),
    labels = class_len.keys(),
    autopct = "%1.1f%%"
)
fig.show()
# The dataset is imbalance so we will have to take care of that later.
# We do not have separate folders for training and validation. 
# We need to read training and validation images from the same folder such that:
# 1. There is no data leak i.e. Training images should not appear as validation images.                 
# 2. We must be able to apply augmentation to training images but not validation images.  
# We shall adopt the following strategy:
# 1. Use the same validation_split in ImageDataGenerator for training and validation.
# 2. Use the same seed when using flow_from_directory for training and validation. 
# To veify the correctness of this approach, you can print filenames from each generator and check for overlap.

# Another problem is that along with the 3 image data folders, there are files we are not making use of.
# To be sure that images are read from the correct folders, we can specify the directory and the labels.

# Note that we use simple augmentation to avoid producing unsuitable images.

datagen_train = ImageDataGenerator(
    rescale = 1./255, 
    validation_split = 0.2,
    rotation_range = 5,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    zoom_range = 0.01
)

datagen_val = ImageDataGenerator(
    rescale = 1./255, 
    validation_split = 0.2 
)    

train_generator = datagen_train.flow_from_directory(
    directory = path,
    classes = labels,
    seed = 42,
    batch_size = 32, 
    shuffle = True,
    subset = 'training'
)

val_generator = datagen_val.flow_from_directory(
    directory = path,
    classes = labels,
    seed = 42,
    batch_size = 32, 
    shuffle = True,
    subset = 'validation'
)
# To veify the correctness of this approach (empty set is expected)
set(val_generator.filenames).intersection(set(train_generator.filenames))
# Check out labeling
val_generator.class_indices
basemodel = InceptionV3(
    include_top = False, 
    weights = 'imagenet', 
    input_tensor = Input((256, 256, 3)),
)
basemodel.trainable = True
basemodel.summary()
# Add classification head to the model
headmodel = basemodel.output
headmodel = GlobalAveragePooling2D()(headmodel)
headmodel = Flatten()(headmodel) 
headmodel = Dense(256, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(3, activation = "softmax")(headmodel) # 3 classes

model = Model(inputs = basemodel.input, outputs = headmodel)
model.summary()
# Compile the model

# Given that COVID-19 spreads very quickly, it is important that we identify as many cases as possible.
# We do not care a lot about Flase Positives (precision), because it may be okay to declare normal people as being COVID-19 positive.
# However, we really really care about False Negatives (recall), because it is NOT okay to declare COVID-19 positive people as being normal!

MyList = ["accuracy"]
MyList += [Recall(class_id = i) for i in range(len(labels))] 
MyList += [Precision(class_id = i) for i in range(len(labels))]

model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = MyList
)

# Using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(
    monitor = "recall",
    patience = 20
)

# save the best model with lower loss
checkpointer = ModelCheckpoint(
    filepath = "weights.hdf5", 
    save_best_only = True
)
# Previously we found that there was class imbalance. 
# We shall use class weights to tackle this before moving to training.

total_wt = sum(class_len.values())

weights = {
    0: 0.5 * (1 - class_len[labels[0]]/total_wt),
    1: 0.5 * (1 - class_len[labels[1]]/total_wt),
    2: 0.5 * (1 - class_len[labels[2]]/total_wt)
}
weights
# Finally, fit the neural network model to the data.

history = model.fit_generator(
    train_generator,
    class_weight = weights,
    validation_data = val_generator,
    steps_per_epoch = 32,
    epochs = 100, 
    callbacks = [earlystopping, checkpointer]
)
# Plotting training and validation loss per epoch

train_loss = history.history["loss"]
valid_loss = history.history["val_loss"]

epochs = range(len(train_loss)) 

plt.plot(epochs, train_loss)
plt.plot(epochs, valid_loss)
plt.legend(["Training Loss", "Validation Loss"])
plt.title("Training and Validation Loss")
# Plotting training and validation accuracy per epoch

train_acc = history.history["accuracy"]
valid_acc = history.history["val_accuracy"]

epochs = range(len(train_acc)) 

plt.plot(epochs, train_acc)
plt.plot(epochs, valid_acc)
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.title("Training and Validation Accuracy")
# Plotting training and validation recall per epoch

fig, axs = plt.subplots(1, 3, figsize = (15, 5))

train_rec_0 = history.history["recall"]
valid_rec_0 = history.history["val_recall"]
train_rec_1 = history.history["recall_1"]
valid_rec_1 = history.history["val_recall_1"]
train_rec_2 = history.history["recall_2"]
valid_rec_2 = history.history["val_recall_2"]

epochs = range(len(train_rec_0)) 

axs[0].plot(epochs, train_rec_0)
axs[0].plot(epochs, valid_rec_0)
axs[0].legend(["Training Recall (Class 0)", "Validation Recall (Class 0)"])
axs[0].set_title("Training and Validation Recall for class 0")

axs[1].plot(epochs, train_rec_1)
axs[1].plot(epochs, valid_rec_1)
axs[1].legend(["Training Recall (Class 1)", "Validation Recall (Class 1)"])
axs[1].set_title("Training and Validation Recall for class 1")

axs[2].plot(epochs, train_rec_2)
axs[2].plot(epochs, valid_rec_2)
axs[2].legend(["Training Recall (Class 2)", "Validation Recall (Class 2)"])
axs[2].set_title("Training and Validation Recall for class 2")

fig.tight_layout()
# Plotting training and validation precision per epoch

fig, axs = plt.subplots(1, 3, figsize = (15, 5))

train_pre_0 = history.history["precision"]
valid_pre_0 = history.history["val_precision"]
train_pre_1 = history.history["precision_1"]
valid_pre_1 = history.history["val_precision_1"]
train_pre_2 = history.history["precision_2"]
valid_pre_2 = history.history["val_precision_2"]

epochs = range(len(train_pre_0)) 

axs[0].plot(epochs, train_pre_0)
axs[0].plot(epochs, valid_pre_0)
axs[0].legend(["Training Precision (Class 0)", "Validation Precision (Class 0)"])
axs[0].set_title("Training and Validation Precision for class 0")

axs[1].plot(epochs, train_pre_1)
axs[1].plot(epochs, valid_pre_1)
axs[1].legend(["Training Precision (Class 1)", "Validation Precision (Class 1)"])
axs[1].set_title("Training and Validation Precision for class 1")

axs[2].plot(epochs, train_pre_2)
axs[2].plot(epochs, valid_pre_2)
axs[2].legend(["Training Precision (Class 2)", "Validation Precision (Class 2)"])
axs[2].set_title("Training and Validation Precision for class 2")

fig.tight_layout()
# Confusion Matrix 

# Since we do not have a lot of data, we did not split into training-validation-testing.
# Instead we split into training-validation.
# Strictly speaking, we should verify performance against new images from testing dataset.
# However, we shall use images in validation dataset for testing. 

# There is one problem. Previously, we set shuffle = True in our generator.
# This makes it difficult to obtain predictions and their corresponding ground truth labels.
# Thus, we shall call the generator again, but this time set shuffle = False.

val_generator = datagen_val.flow_from_directory(
    directory = path,
    classes = labels,
    seed = 42,
    batch_size = 32, 
    shuffle = False,
    subset = 'validation'
)

# Obtain predictions
pred = model.predict_generator(val_generator) # Gives class probabilities
pred = np.round(pred) # Gives one-hot encoded classes
pred = np.argmax(pred, axis = 1) # Gives class labels

# Obtain actual labels
actual = val_generator.classes
    
# Now plot matrix
cm = confusion_matrix(actual, pred, labels = [0, 1, 2])
sns.heatmap(
    cm, 
    cmap="Blues",
    annot = True, 
    fmt = "d"
)
plt.show()
# Classification Report
print(classification_report(actual, pred))
# These results are not too great but not too bad either. 
# In this notebook, we saw how we can identify COVID-19 from Chest X-Rays.
# Cheers. Happy Learning!