%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
#1. Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')

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
skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

# Creating New Columns for better readability

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
# Now lets see the sample of tile_df to look on newly made columns
skin_df.head()
skin_df.isnull().sum()
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
skin_df.isnull().sum()
print(skin_df.dtypes)
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
skin_df['dx_type'].value_counts().plot(kind='bar')
skin_df['localization'].value_counts().plot(kind='bar')
skin_df['age'].hist(bins=40)

skin_df['sex'].value_counts().plot(kind='bar')
sns.scatterplot('age','cell_type_idx',data=skin_df)
sns.factorplot('sex','cell_type_idx',data=skin_df)
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
skin_df.head()
def remove_dups(df, subset=None, index=False):
    """
    Drop all for EXTRA occurences. 
    Arguments:
        ``subset`` - column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by default use all of the columns
        ``index`` - True or False; default False.
            Whether you want to duplication to be judged solely on the index.
    """
    if index: dedup_tf = df.index.duplicated(subset=subset, keep='first') #returns an array with T/F for EXTRA occurences of dates
    else: dedup_tf = df.duplicated(subset=subset, keep='first') #returns an array with T/F for EXTRA occurences of dates
    dedup_indx = np.where(dedup_tf == False) #record non-duplicate indices
    return df.iloc[dedup_indx] #slice df by indicies
skin_df_dd = remove_dups(skin_df, subset='lesion_id').copy()
#check for duplicate image_ids
skin_df_dd['lesion_id'].value_counts().head()
#order by diagnosis
skin_df_dd.sort_values(by=['dx'], inplace=True)
#look at the existing abbreviations and choose two
set(skin_df_dd['dx'])
#grab 20 images from 2 classes for the training set
akiec = skin_df_dd.iloc[np.where(skin_df_dd['dx']=='akiec')[0][0:20], :] 
bcc = skin_df_dd.iloc[np.where(skin_df_dd['dx']=='bcc')[0][0:20], :] 
#concatonate them
train = pd.concat([akiec, bcc])
train.head()
#grab 200 images from 2 classes for the test set
akiec_test = skin_df_dd.iloc[np.where(skin_df_dd['dx']=='akiec')[0][20:220], :] 
bcc_test = skin_df_dd.iloc[np.where(skin_df_dd['dx']=='bcc')[0][20:220], :] 
#concatonate them
test = pd.concat([akiec_test, bcc_test])
test.head()
print(train.shape)
print(test.shape)
n_samples = 5
fig, m_axs = plt.subplots(2, n_samples, figsize = (4*n_samples, 3*2))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         train.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)
# Checking the image size distribution
train['image'].map(lambda x: x.shape).value_counts()
#split x and y columns
x_train = train.drop(columns=['cell_type_idx'],axis=1)
y_train = train['cell_type_idx']

x_test = test.drop(columns=['cell_type_idx'],axis=1)
y_test = test['cell_type_idx']
x_train_arr = np.asarray(x_train['image'].tolist())
x_test_arr = np.asarray(x_test['image'].tolist())
x_train_arr.shape
# Perform one-hot encoding on the labels
y_train = to_categorical(y_train, num_classes = 2)
y_test = to_categorical(y_test, num_classes = 2)
y_train[:5,:]
y_train[-6:-1,:]
# With data augmentation to prevent overfitting 

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train_arr)
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input_shape = (75, 100, 3)
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["acc"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# Fit the model
epochs = 50 
batch_size = 10
history = model.fit_generator(datagen.flow(x_train_arr, y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (x_test_arr, y_test,),
                              verbose = 0, 
                              steps_per_epoch=x_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction])
plot_model_history(history)
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))
# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False
#compile model
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#fit model
epochs = 50 
batch_size = 10

history = my_new_model.fit_generator(datagen.flow(x_train_arr, y_train, batch_size=batch_size),
                           verbose = 0,
                           epochs = epochs,
                           steps_per_epoch=x_train.shape[0] // batch_size,
                           validation_data = (x_test_arr, y_test)
                          )
plot_model_history(history)