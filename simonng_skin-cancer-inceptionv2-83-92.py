# See our code and result: https://github.com/SimonNgj/skinCancer
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

np.random.seed(123)

import itertools

from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight



import keras

from keras import applications

from keras import backend as K

from keras import regularizers

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Model

from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, AveragePooling2D

from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau

from keras.layers import Input, BatchNormalization, Activation, Dense

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from keras.applications.inception_resnet_v2 import InceptionResNetV2
# Load images

base_skin_dir = os.path.join('./skinCancer')



# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary



imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}



# This dictionary is useful for displaying more human-friendly labels later on



lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}
# Load notations

skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))



# Creating New Columns for better readability



skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)

skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 

skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
# By disease

fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))

skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
# By location of images

skin_df['localization'].value_counts().plot(kind='bar')
# By age

skin_df['age'].hist(bins=40)
# By sex

skin_df['sex'].value_counts().plot(kind='bar')
# dimensions of our images.

img_width, img_height = 224, 224



# Crop and resize images

skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((img_height,img_width))))
# Show examples

n_samples = 5

fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))

for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(['cell_type']).groupby('cell_type')):

    n_axs[0].set_title(type_name)

    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):

        c_ax.imshow(c_row['image'])

        c_ax.axis('off')

#fig.savefig('category_samples.png', dpi=300)
# Take label

features=skin_df.drop(columns=['cell_type_idx'],axis=1)

target=skin_df['cell_type_idx']



# Checking the image size distribution

skin_df['image'].map(lambda x: x.shape).value_counts()
# Split data into train/test set

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.2,random_state=34)



# Normalize data

x_train = np.asarray(x_train_o['image'].tolist())

x_test = np.asarray(x_test_o['image'].tolist())
# One-hot encoding on the labels

y_train = to_categorical(y_train_o, num_classes = 7)

y_test = to_categorical(y_test_o, num_classes = 7)



# Split train set into train/validation set

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.25, random_state=12)
# Reshape image in 3 dimensions (height, width , canal = 3)

x_train = x_train.reshape(x_train.shape[0], *(img_height, img_width, 3))

x_validate = x_validate.reshape(x_validate.shape[0], *(img_height, img_width, 3))

x_test = x_test.reshape(x_test.shape[0], *(img_height, img_width, 3))
# With data augmentation to prevent overfitting 

datagen = ImageDataGenerator(rotation_range=10)  # randomly rotate images in the range (degrees, 0 to 180)



datagen.fit(x_train)
input_shape = (img_height, img_width, 3)

num_classes = 7
def build_model():

    # load pre-trained model graph, don't add final layer

    model = InceptionResNetV2(include_top=False, input_shape = (img_height,img_width,3), weights='imagenet')

    new_output =GlobalAveragePooling2D()(model.output)

    new_output = Dropout(0.25)(new_output)

    new_output=Dense(256,activation='relu')(new_output)

    new_output = Dense(num_classes, kernel_regularizer=regularizers.l2(0.001), activation='softmax')(new_output)

    model = Model(model.inputs, new_output)

    return model
model=build_model()

model.compile(metrics=['accuracy'],loss='categorical_crossentropy', optimizer='Adam')

callbacks=[ ModelCheckpoint('model_4e.h5',monitor='val_loss',save_best_only=True), 

           ReduceLROnPlateau(monitor='val_loss',patience=5)]
epochs = 20

batch_size = 16

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=callbacks)
# Predict test set

accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)

accuracy = model.evaluate(x_test, y_test, verbose=1)

print(model.metrics_names)

print("Validation: ", accuracy_v)

print("Test: ", accuracy)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(x_validate)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_validate,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(7)) 
label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)

plt.bar(np.arange(7),label_frac_error)

plt.xlabel('True Label')

plt.ylabel('Fraction classified incorrectly')
# Function to plot model's validation loss and validation accuracy

met = 'acc'

def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history[met])+1),model_history.history[met])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history[met])+1),len(model_history.history[met])/10)

    axs[0].legend(['train', met], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

    

plot_model_history(history)
p1 = model.predict(x_test[52:53])

print("Prediction: ", np.argmax(p1))

print("Actual: ", np.argmax(y_test[52:53]))
abc = x_test[0:1]

abc.shape
abc1= abc[0]

abc1.shape
np.shape(x_train)
np.shape(x_tes)
name = "ISIC_0024432"

test_image = "./"+name+".jpg"

original = Image.open(test_image)

original.show()
width, height = original.size   # Get dimensions

left = (width-height)/2

right = left+height

top = 0

bottom = height

cropped_example = original.crop((left, top, right, bottom)).resize((100,100))



cropped_example.show()

cropped_example.save('./'+name+'_crop.bmp')
resize_example = original.resize((100,75))



resize_example.show()

resize_example.save('./'+name+'_resize.bmp')
a1 = np.array(cropped_example)

a1.shape

a2 = a1.reshape((1,100,100,3))
# Predict

p2 = model.predict(a2)

print("Prediction: ", np.argmax(p2))