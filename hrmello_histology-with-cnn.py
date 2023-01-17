import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob #deal with paths
import os #deal with paths
import cv2 #deal with images
from skimage.io import imread #read images from files
import matplotlib.pyplot as plt #make plots
import seaborn as sb #pretty plots :P 
%matplotlib inline 
base_tile_dir = '../input/kather_texture_2016_image_tiles_5000/'
df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir, '*', '*', '*.tif'))}) 
df['cell_type'] = df['path'].map(lambda x: os.path.basename(os.path.dirname(x)).split('_')[1])
df['cell_type_idx'] = df['path'].map(lambda x: os.path.basename(os.path.dirname(x)).split('_')[0])
df['image_name'] = df['path'].map(lambda x: os.path.basename(x).split('_Row')[0])
df['image_row'] = df['path'].map(lambda x: os.path.basename(x).split('_Row')[1].split('_')[1])
df['image_col'] = df['path'].map(lambda x: os.path.basename(x).split('_Row')[1].split('_')[3].split('.')[0])
df['image'] = df['path'].map(imread)
df.sample(5)
from random import randint

def get_first_cell_images(df):
    #get unique cell types
    unique_cell_type = df.cell_type.unique()
    cell_images = []
    for cell in unique_cell_type:
        #get the first row containing an image of cell_type == cell
        first_img_idx= np.where(df.cell_type == cell) 
        #list containing one image of each type
        cell_images.append((df[df.cell_type == cell].loc[first_img_idx[0][0],'image'], cell))
    return cell_images

images = get_first_cell_images(df)

#create the subplots
fig, m_axs = plt.subplots(1, len(images), figsize = (20, 2))
#show the images and label them
for ii, c_ax in enumerate(m_axs):
    c_ax.imshow(images[ii][0])
    c_ax.set_title(images[ii][1])

from keras.preprocessing import image                  
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True     
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(150,150))
    # convert PIL.Image.Image type to 3D tensor with shape (150, 150, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 150, 150, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

images_path = list(df.path)

images_tensors = paths_to_tensor(images_path)

images_tensors.shape
df.cell_type.unique()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils


train_fraction = 0.8

encoder = LabelBinarizer()
y = encoder.fit_transform(df.cell_type)
x = images_tensors

train_tensors, test_tensors, train_targets, test_targets =\
    train_test_split(x, y, train_size = train_fraction, random_state = 42)

val_size = int(0.5*len(test_tensors))

val_tensors = test_tensors[:val_size]
val_targets = test_targets[:val_size]
test_tensors = test_tensors[val_size:]
test_targets = test_targets[val_size:]
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from tensorflow import set_random_seed

set_random_seed(42)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 8)
checkpointer = ModelCheckpoint(filepath='weights.hdf5', 
                               verbose=1, save_best_only=True)
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (150, 150, 3)))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3))
model.add(GlobalMaxPooling2D())
model.add(Dense(8, activation = 'softmax'))

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50
model.fit(train_tensors, train_targets, 
          validation_data=(val_tensors, val_targets),
          epochs=epochs, batch_size=20, verbose=1, callbacks = [early_stopping, checkpointer])
model.load_weights('weights.hdf5')

cell_predictions =  [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

test_accuracy = 100*np.sum(np.array(cell_predictions)==np.argmax(test_targets, axis=1))/len(cell_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)