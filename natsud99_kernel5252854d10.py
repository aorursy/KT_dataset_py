import numpy as np
import pandas as pd
import os

%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from glob import glob
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import itertools
skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')

# Merging the images from the 2 folders
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(skin_dir, '*', '*.jpg'))}
len(imageid_path_dict)# Gives the total number of images in the 2 folders
skin_df=pd.read_csv(os.path.join(skin_dir, 'HAM10000_metadata.csv'))
skin_df.head()
plt.style.use('bmh')
fig, ax1 = plt.subplots(1, 1, figsize = (8, 5))
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
plt.title('No. of images for diff. diagnosis types')
# This shows us that the data is skewed. 
# There are way more images for nv. 
# So we need to resample/upsample the data to handle this.
skin_df['cell_type_idx'] = pd.Categorical(skin_df['dx']).codes
skin_df.head()
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((128,128))))
skin_df.head()
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=123)
print(x_train_o.shape, x_test_o.shape, y_train_o.shape, y_test_o.shape)
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
# One-hot encoding on the labels
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)
# Reshaping into channels first
x_train = x_train.reshape(x_train.shape[0], *(128, 128, 3))
x_test = x_test.reshape(x_test.shape[0], *(128, 128, 3))
x_valid = x_valid.reshape(x_valid.shape[0], *(128, 128, 3))
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
for layer in base_model.layers[:-50]:
    layer.trainable=False
len(base_model.layers)
model=Sequential()
model.add(base_model)
    
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform', 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))

model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform', 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.00142), loss='categorical_crossentropy', metrics=['accuracy'])
datagen= ImageDataGenerator(rotation_range= 60,
                           zoom_range= 0.2,
                           width_shift_range= 0.1,
                           height_shift_range= 0.1,
                           horizontal_flip= True,
                           shear_range=0.2)
datagen.fit(x_train)
filepath='model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, 
                                   patience=10, verbose=1, mode='auto', 
                                   epsilon=0.0001, cooldown=5, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10)

callbacks_list = [checkpoint, early, reduce_lr]
%%time

epochs= 6
batch_size= 32
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size= batch_size),
                              epochs= epochs, validation_data= (x_valid, y_valid),
                              verbose= 1, steps_per_epoch= x_train.shape[0]//batch_size, 
                              callbacks= callbacks_list)
model.save('skc_model.h5')
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print("Test set accuracy= %f ; loss= %f" % (acc, loss))
val_loss, val_acc = model.evaluate(x_valid, y_valid, verbose=1)
print("Validation set accuracy= %f ; val_loss= %f" % (val_acc, val_loss))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

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
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_valid)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_valid,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7)) 
