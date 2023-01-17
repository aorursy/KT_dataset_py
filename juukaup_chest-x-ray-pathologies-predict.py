

import numpy as np 

import pandas as pd 

import matplotlib.pylab as plt

import seaborn as sns

import cv2

import sklearn

from sklearn import model_selection

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc



import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Sequential, Model

from keras.optimizers import Adam

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D

from keras.layers import MaxPool2D, MaxPooling2D, GlobalAveragePooling2D, SeparableConv2D

from keras.applications.vgg16 import VGG16

import tensorflow as tf



from glob import glob

import os

%matplotlib inline

# Set seeds

np.random.seed(999)

tf.random.set_seed(1999)



# Load entry data

BBox_List = pd.read_csv("../input/data/BBox_List_2017.csv")

Data_Entry = pd.read_csv("../input/data/Data_Entry_2017.csv")



print(os.listdir("../input/data/"))

print()

print("Image indices in data entry: {}".format(len(Data_Entry["Image Index"])))



Data_Entry = Data_Entry.set_index("Image Index")

Data_Entry.head()
# Map the image paths onto xray_data

# Credit: https://www.kaggle.com/kmader/train-simple-xray-cnn



image_paths = {os.path.basename(x): x for x in glob('../input/data/images*/images/*.png')}

Data_Entry['path'] = Data_Entry.index.map(image_paths.get)
# Let's examine the distribution of findings

unique_labels = Data_Entry["Finding Labels"].unique()

print("Number of unique labels: {}".format(len(unique_labels)))



label_counts = Data_Entry["Finding Labels"].value_counts()

#print(label_counts)



# Bar plot the most frequent finding labels

fig = plt.figure(figsize=(10,6))

bp = sns.barplot(label_counts.index[:15], label_counts.values[:15])



# Rotate labels and suppress output

_ = bp.set_xticklabels(label_counts.index, rotation=90)
# The idea is to try to classify only non-related conditions from the 

# images ignoring the probable secondary conditions



# Let's filter the data so that only images with one of the chosen conditions 

# is present in each image in the used dataset



conditions = ["No Finding","Cardiomegaly", "Emphysema", "Fibrosis", "Nodule", "Pneumonia"]



def filter_conditions(df, conditions, x):

    filtered = df[df["Finding Labels"].str.contains(x)]

    

    other_conditions = list(conditions)

    other_conditions.remove(x)

    

    for cond in other_conditions:

        remove = filtered[filtered["Finding Labels"].str.contains(cond, regex=True)]

        filtered = filtered.drop(remove.index)

    return filtered



data_filtered = pd.DataFrame()

for x in conditions:

    data_filtered = data_filtered.append(filter_conditions(Data_Entry, conditions, x))



# Since a great majority of the data has "No Finding" label, let's balance the dataset by sampling out  

# a large fraction of the class



no_finding_frac = data_filtered[data_filtered["Finding Labels"]=="No Finding"].sample(frac=0.9, replace=False).index

data_filtered = data_filtered.drop(no_finding_frac)

print("Size of dataset: {}".format(len(data_filtered)))

data_filtered.head(10)

# Adding target labels for the examined conditions

data_dummies = data_filtered["Finding Labels"].str.get_dummies("|")



#data_filtered["target"] = data_dummies[conditions].values.tolist()

#for cond in conditions:

#    data_filtered[cond] = data_filtered["Finding Labels"].map(lambda labels : cond in labels )

#data_filtered["target"] = data_filtered["target"].map(np.array)



data_filtered["target"] = data_dummies[conditions].idxmax(axis=1)

print(data_filtered["target"].value_counts())

data_filtered.sample(10)
# Plot data distribution

fig = plt.figure(figsize=(8,5))

sns.barplot(conditions, data_dummies[conditions].sum())
# Show a few sample images of conditions



def plot_samples(condition, n_samples, ax):

    img = data_filtered[data_filtered["target"]==condition].sample(n_samples, random_state=1111).index

    image_paths = list(data_filtered.loc[img, "path"])

    for i,axis in enumerate(ax):

        image = cv2.imread(image_paths[i])

        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        axis.axis('off')

        axis.set_title(condition)



n = len(conditions)

fig, ax = plt.subplots(n, 5, figsize=(16,20))



for j,cond in enumerate(conditions):

    plot_samples(cond, 5, ax[j,:])
train_data, test_data = train_test_split(data_filtered, test_size=0.25, random_state=3030)

test_data, validation_data = train_test_split(test_data, test_size=0.5, random_state=2020)

train_data.head()

print("number of samples in training set: {}".format(len(train_data)))

print("number of samples in validation set: {}".format(len(validation_data)))

print("number of samples in test set: {}".format(len(test_data)))
# Creating generator for data



train_data_gen = ImageDataGenerator(

        rescale=1.0/255.0,

        shear_range=0.2,

        zoom_range=0.2,

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        horizontal_flip=True)



test_data_gen = ImageDataGenerator(rescale=1.0/255.0)

batch_size = 64



# Use the path variable as filename since it already contains the full path

train_generator = train_data_gen.flow_from_dataframe(

        train_data,

        x_col="path",

        y_col = "target",

        class_mode = "categorical",

        seed = 1,

        color_mode = 'rgb',

        batch_size = batch_size,

        target_size = (224,224)

        )



validation_generator = test_data_gen.flow_from_dataframe(

        validation_data,

        x_col="path",

        y_col = "target",

        class_mode = "categorical",

        color_mode = 'rgb',

        target_size = (224,224),

        batch_size = 256

        )



test_generator = test_data_gen.flow_from_dataframe(

        test_data,

        x_col="path",

        y_col = "target",

        class_mode = "categorical",

        color_mode = 'rgb',

        target_size = (224,224),

        batch_size = 64,

        shuffle=False

        )
# Plotting a sample image from the training generator



train_x, train_y = next(train_generator)

print(train_x[0].shape)

fig = plt.figure()

plt.imshow(cv2.cvtColor(train_x[0], cv2.COLOR_BGR2RGB))

print(os.listdir("../input"))

base_model = VGG16(include_top=False, weights=None, input_shape=(224,224,3))

print(base_model.summary())
# Adding some layers to the pretrained vgg16



base_model.load_weights("../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

base_model.trainable = True

x = base_model.output

x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='block6_conv1')(x)

x = BatchNormalization()(x)

x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='block6_conv2')(x)

x = BatchNormalization()(x)

x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='block6_conv3')(x)



#x = GlobalAveragePooling2D()(x)

x = MaxPooling2D((2,2), name='block6_pool1')(x)

x = Flatten(name='flatten')(x)

x = Dense(1024, activation='relu', name='dense_1')(x)

x = Dropout(0.5, name='dropout_1')(x)

x = Dense(512, activation='relu', name='dense_2')(x)

x = Dropout(0.2, name='dropout_2')(x)

prediction = Dense(6, activation='softmax', name='output_1')(x)



model = Model(inputs=base_model.input, outputs=prediction)



# For fixing the pretrained parameters

#print(len(model.layers))

#for layer in model.layers[:10]:

#    layer.trainable = False

    

# Playing around with different learing rate and decay

lr = 0.0001

n_epochs = 40



opt = Adam(lr=lr, decay=1e-5)



model.compile(loss='categorical_crossentropy', 

                  optimizer=opt, 

                  metrics=['accuracy'])



print(model.summary())
early_stop = EarlyStopping(patience=10)

checkpoint = ModelCheckpoint(filepath='current_best_model', save_best_only=True)



# Define the number of training steps

n_train_steps = train_data.shape[0]//batch_size

n_validation_steps = validation_data.shape[0]//batch_size



print("Number of training steps: {} ".format(n_train_steps))

history = model.fit_generator(train_generator, epochs=n_epochs, steps_per_epoch=n_train_steps,

                              validation_data=validation_generator,

                              validation_steps = n_validation_steps,

                              callbacks=[early_stop, checkpoint])


fig, axes = plt.subplots(2,1, figsize=(10,12))

measures = ["accuracy", "loss"]

for i,ax in enumerate(axes):

    ax.plot(history.history[measures[i]])

    ax.plot(history.history['val_' + measures[i]])

    ax.set_title('model ' + measures[i])

    ax.set_ylabel(measures[i])

    ax.set_xlabel('epoch')

    ax.legend(['train', 'validation'], loc='upper left')
model.load_weights("../input/chest-x-ray-pathologies-predict/current_best_model")
predicted = model.predict(test_generator,steps=len(test_generator), verbose = True)
test_generator.reset()

targets = test_data["target"].unique()

targets.sort()

print(test_data["target"].value_counts())

print(targets)

print()



print(predicted[0:5])

print()



# Predicted classes as integers and One hot encoded

pred_Y_as_int = np.argmax(predicted, axis=1)

n_classes = np.max(pred_Y_as_int)+1

pred_Y = np.eye(n_classes)[pred_Y_as_int]

print(pred_Y[0:5])

print()



# Same for true labels

test_Y_as_int = test_generator.labels

n_test = np.max(test_Y_as_int)+1

test_Y = np.eye(n_test)[test_Y_as_int]

print(test_Y[0:5])

test_data.head()

score = accuracy_score(test_Y, pred_Y)

print("Accuracy score: {}".format(score))



fig, ax = plt.subplots(1,1, figsize = (9, 9))

for (idx, target) in enumerate(targets):

    fpr, tpr, thresholds = roc_curve(test_Y[:,idx], predicted[:,idx])

    ax.plot(fpr, tpr, label = "{0:s} (AUC:{1:.2f})".format(target, auc(fpr, tpr)))



ax.legend()

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')
from mlxtend.plotting import plot_confusion_matrix

plot_confusion_matrix(confusion_matrix(test_Y_as_int, pred_Y_as_int), figsize=(10,10))

plt.xticks(range(len(targets)), targets, fontsize=10, rotation=45)

plt.yticks(range(len(targets)), targets, fontsize=10)