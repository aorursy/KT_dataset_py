# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import zipfile

import tensorflow as tf
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np 
sns.set()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
df.tail()

sns.countplot(df['sex'])
sns.countplot(y = df['anatom_site_general_challenge'], hue =df['sex'])
sns.distplot(df['age_approx'], bins=20, kde=False, rug=True);
sns.countplot(y = df['diagnosis'], hue =df['sex'])
sns.countplot(y = df['anatom_site_general_challenge'], hue =df['sex'])
sns.countplot(x = df['benign_malignant'],  hue =df['sex'] )
sns.countplot(x = df['target'])
a = np.sum(df['target'].values)
# count 1s 
print ('number of one in target are:', a)
print ('% of one in target are:', (a/(len(df)))*100, '%')
# count 0s 
print ('number of zeros in target are:', len(df)-a)
print ('% of zeros in target are:', ((len(df)-a)/len(df))*100, '%')



# for malignant image 
Path_train="/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
train_dir=os.listdir(Path_train)
import cv2
for i in range(len(df)):
    
    if df['benign_malignant'].values[i] == 'malignant' :
        
        plt.figure(figsize = [10,19])
        #print(df.iloc[i])
        a = df['image_name'].values[i]

        raw_image = cv2.imread(Path_train+a+'.jpg')
        #print(raw_image)
        plt.imshow(raw_image)
        #plt.colorbar()
        plt.title('Raw Image (malignant)')
        print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel")
        print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
        print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")
        break 
# for malignant image 
Path_train="/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
train_dir=os.listdir(Path_train)
import cv2
for i in range(len(df)):
    
    if df['benign_malignant'].values[i] == 'benign' :
        
        plt.figure(figsize = [10,19])
        #print(df.iloc[i])
        a = df['image_name'].values[i]

        raw_image = cv2.imread(Path_train+a+'.jpg')
        #print(raw_image)
        plt.imshow(raw_image)
        #plt.colorbar()
        plt.title('Raw Image (benign)')
        print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel")
        print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
        print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")
        break 
# Directory with our training horse pictures
train_dir = os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/')
train_names = os.listdir(train_dir)
print(train_names[:10])

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 2
ncols = 4

# Index for iterating over images
pic_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 16
next_pix = [os.path.join(train_dir, fname) 
                for fname in train_names[pic_index-8:pic_index]]


for i, img_path in enumerate(next_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


'''class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get(tf.keras.metrics.AUC())>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()  '''

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
tf.keras.utils.plot_model(model,show_layer_names=True,show_shapes=True)
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=[tf.keras.metrics.AUC()])
# All images will be rescaled by 1./255
from tensorflow.keras.preprocessing.image import ImageDataGenerator


Image_path='/kaggle/input/siim-isic-melanoma-classification/jpeg/'
# dtype string because its reads in string format
train_csv=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv',dtype=str)
test_csv=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv',dtype=str)
train_augmenter=ImageDataGenerator(
    rescale=1./255, 
    #rotation range and fill mode only
    samplewise_center=True, 
    samplewise_std_normalization=True, 
    horizontal_flip = True, 
    vertical_flip = True, 
    height_shift_range= 0.05, 
    width_shift_range=0.1, 
    rotation_range=45, 
    shear_range = 0.1,
    fill_mode = 'nearest',
    zoom_range=0.10,
    #preprocessing_function=function_name,
    )

test_augmenter=ImageDataGenerator(
    rescale=1./255
    )
def jpg_tag(image_name):
    return image_name+'.jpg'

train_csv['image_name']=train_csv['image_name'].apply(jpg_tag)
test_csv['image_name']=test_csv['image_name'].apply(jpg_tag)

from keras.utils.data_utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator

class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""
    def __init__(self, x, y, datagen, batch_size=32):
        self.datagen = datagen
        self.batch_size = batch_size
        self._shape = x.shape        
        datagen.fit(x)
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)

    def __len__(self):
        return self._shape[0] // self.batch_size

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()
batch_size=16
IMG_size=224
train_generator=train_augmenter.flow_from_dataframe(
dataframe=train_csv,
directory=Image_path+'train',
#save_to_dir='augmented',
#save_prefix='_aug'
#save_format='jpg'
x_col='image_name',
y_col='target',
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode='binary',
target_size=(300, 300),
)



test_generator=test_augmenter.flow_from_dataframe(
dataframe=test_csv,
directory=Image_path+'test',
x_col='image_name',
batch_size=batch_size, #preffered 1
shuffle=False,
class_mode=None,
target_size=(300, 300)
)
history = model.fit(
      train_generator,
      steps_per_epoch=10,
        
      epochs=5,
      verbose=1)
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
img_files = [os.path.join(train_dir, f) for f in train_names]
img_path = random.choice(img_files)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL imagea
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
submission = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submission.head()
name = submission['image_name'].values 
name 
Path_test="/kaggle/input/siim-isic-melanoma-classification/jpeg/test/"
test_dir=os.listdir(Path_test)
ad = Path_test+name[0]+'.jpg'
img = mpimg.imread(ad)
plt.imshow(img)
array = []
for i in range(len(submission)):
    from keras.preprocessing import image
    ad = Path_test+name[i]+'.jpg'
    img = image.load_img(ad, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    '''print(classes[0])'''
    array.append(classes[0])
print(array)
for k in range(len(array)):
    if array==1:
        print('yes')
print ('finish')
er = pd.DataFrame(array)
er.columns = ['target']
er
submission = submission.drop(columns='target')
submission 
submission['target'] =  er
submission.to_csv('submission.csv') 




