import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cv2
import os
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model,Sequential, load_model
import gc
expression_types = ['Disappointed', 'interested', 'neutral']
data_path = '../input/fercustomdataset-3classes/FER_Custom_Dataset'
data_dir = os.path.join(data_path)
data = []
for expression_type, sp in enumerate(expression_types):
    for file in os.listdir(os.path.join(data_dir, sp)):
        data.append(['{}/{}'.format(sp, file), expression_type, sp])
        
data = pd.DataFrame(data, columns=['File', 'Expression_Id','Expression_Type'])

print(data.Expression_Id.value_counts())
data.head()
SEED = 0
data = data.sample(frac=1, random_state=SEED) 
data.index = np.arange(len(data)) # Reset indices
data.head()
# Display images for different expressions
def plot_expression(expression_type, rows, cols,dir_type,df):
    fig, ax = plt.subplots(rows, cols, figsize=(5, 5))
    expression_files = df['File'][df['Expression_Type'] == expression_type].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(dir_type, expression_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1
plot_expression('interested', 2, 2, data_dir, data)
IMAGE_SIZE = 128

def read_image(filepath,data_dir):
    return cv2.imread(os.path.join(data_dir, filepath),0) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
X = np.zeros((data.shape[0], IMAGE_SIZE, IMAGE_SIZE))
for i, file in tqdm(enumerate(data['File'].values)):
    image = read_image(file,data_dir)
    if image is not None:
        X[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X = X / 255.
print('Train Shape: {}'.format(X.shape))
Y = data['Expression_Id'].values
Y_categorical = to_categorical(Y)
print(Y_categorical.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_categorical, test_size=0.02, random_state = SEED)
del X
gc.collect()
X_train = np.reshape(X_train,(X_train.shape[0], IMAGE_SIZE, IMAGE_SIZE,1))
X_test = np.reshape(X_test,(X_test.shape[0], IMAGE_SIZE, IMAGE_SIZE,1))
print(X_train.shape)
print(X_test.shape)
EPOCHS = 50
SIZE = 128
N_ch = 1
BATCH_SIZE = 64
def build_densenet():
    densenet = DenseNet201(weights='../input/densenet202weightsfile/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

    input = tf.keras.Input(shape=(SIZE, SIZE,N_ch))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(3,activation = 'softmax', name='root')(x)
 

    # model
    model = Model(input,output)
    
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalCrossentropy(), 'accuracy'])
    model.summary()
    
    return model
model = build_densenet()
annealer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-4)
checkpoint = tf.keras.callbacks.ModelCheckpoint('facial_expression_densenet201v2.h5', verbose=1, save_best_only=True)

# Generates batches of image data with data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 20, # Degree range for random rotations
                        width_shift_range = 0.1,                               # Range for random horizontal shifts
                        height_shift_range = 0.1,                              # Range for random vertical shifts
                        zoom_range = 0.1,                                      # Range for random zoom
                        horizontal_flip = True,                                # Randomly flip inputs horizontally
                        vertical_flip = False)                                 # Randomly flip inputs vertically
datagen.fit(X_train)
# Fits the model on batches with real-time data augmentation
hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
               epochs=EPOCHS,
               verbose=1,
               callbacks=[annealer, checkpoint],
               validation_data=(X_test, Y_test))

