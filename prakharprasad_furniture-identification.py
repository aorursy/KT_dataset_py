# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

###### Uncomment the code below if required #########
#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# usual imports #
import os
import numpy as np
import pandas as pd

# visualization imports #
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread
%matplotlib inline

# consistent plots #
from pylab import rcParams
rcParams['figure.figsize']= 12,5
rcParams['xtick.labelsize']= 12
rcParams['ytick.labelsize']= 12
rcParams['axes.labelsize']= 12

# ignore unwanted warnings #
import warnings
warnings.filterwarnings(action='ignore',message='^internal gelsd')
# designate directory to save the images #
ROOT_DIR = '/kaggle/input/day-3-kaggle-competition'

DATA_PATH = os.path.join(ROOT_DIR , 'data_comp/data_comp')
TRAIN_PATH = os.path.join(DATA_PATH,'train')
TEST_PATH = os.path.join(DATA_PATH + '/' + 'test')
# check the files or directories in the training path #
os.listdir(TRAIN_PATH)
rand = np.random.randint(len(os.listdir(TRAIN_PATH)))
furniture_title = os.listdir(TRAIN_PATH)[rand]
furniture_path = os.path.join(TRAIN_PATH,furniture_title)
furniture_images  = os.listdir(furniture_path)
n_rows = 2
n_cols = 4

for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows,n_cols,index+1)
        sample_image_path = os.path.join(furniture_path + '/',furniture_images[index])
        furniture = imread(sample_image_path)
        plt.imshow(furniture,cmap='binary',interpolation='nearest')
        plt.axis('off')
        plt.title(furniture_title,fontsize=10)  
num_images = 0
for folder in os.listdir(TRAIN_PATH):
    num_images = num_images + len(os.listdir(os.path.join(TRAIN_PATH + '/' + folder)))    
print ('Total number of images in the train dir = {}'.format(num_images))
# check the dimension of each training image and calculate the mean shape #
dim1 = []
dim2 = []

for folder in os.listdir(TRAIN_PATH):
    for image_filename in os.listdir(TRAIN_PATH + '/' + folder):
        img = imread(os.path.join(TRAIN_PATH,folder,image_filename))
        #print(os.path.join(TRAIN_PATH,folder,image_filename))
        d1,d2 = img.shape[0],img.shape[1]
        dim1.append(d1)
        dim2.append(d2)
print (np.mean(dim1),np.mean(dim2))

IMAGE_SHAPE = (int(np.mean(dim1)),int(np.mean(dim2)),3)

# image shape with the color channel to be later fed into the model #
IMAGE_SHAPE
# import the image data generator 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# generate images using the data generator --> check help(ImageDataGenerator) #
image_gen = ImageDataGenerator(rotation_range=90,
                               width_shift_range=0.10, 
                               height_shift_range=0.10,
                               rescale=1./255,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest',
                               vertical_flip=False,
                               validation_split=0.3)  
# visualize one of the original image of a furniture #
furniture_orig = imread(sample_image_path)
plt.imshow(furniture_orig)
plt.axis('off')
plt.title('Original Image');
# visualize one the randomly generated image by the image generator of the same fruit #
plt.imshow(image_gen.random_transform(furniture_orig))
plt.axis('off')
plt.title('Image Generated using Data Generator');
image_gen.flow_from_directory(TRAIN_PATH)
# import the libraries #
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from tensorflow import keras
# clear the session #
keras.backend.clear_session()
np.random.seed(42)
# create a sequential model #
model = Sequential()

# convolutional and max pool layer #
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

# flatten the layer before feeding into the dense layer #
model.add(Flatten())

# dense layer together with dropout to prevent overfitting #
model.add(Dense(units=128,activation='relu',kernel_initializer='he_normal'))
model.add(Dense(units=64,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(units=32,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))

# there are 5 classes, hence 5 neurons in the final layer #
model.add(Dense(units=5,activation='softmax'))

# compile the model #
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# check the model summary # 
model.summary()
#model.layers
# import early stopping and model checkpoint #
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
BATCH_SIZE = 16
IMAGE_SHAPE[:2]
train_image_gen = image_gen.flow_from_directory(TRAIN_PATH,target_size=IMAGE_SHAPE[:2],
                                               color_mode='rgb',batch_size=BATCH_SIZE,
                                               class_mode='categorical',seed=1,subset='training')
validation_image_gen = image_gen.flow_from_directory(TRAIN_PATH,target_size=IMAGE_SHAPE[:2],
                                               color_mode='rgb',batch_size=BATCH_SIZE,
                                               class_mode='categorical', shuffle=False,subset='validation',
                                               seed=1)
# check the class indices #
train_image_gen.class_indices
# fit the model and train with early stop enabled #
epoch = 30
history=model.fit(train_image_gen,
                  validation_data = validation_image_gen,
                  epochs = epoch,callbacks=[early_stop])
# create dataframe of the loss and accuracy of the train and validation data #
df_loss = pd.DataFrame(model.history.history)
df_loss.head()
df_loss[['loss','accuracy','val_loss','val_accuracy']].plot()
plt.xlabel('epochs')
plt.ylabel('loss')
model.evaluate(validation_image_gen)
test_image_gen = ImageDataGenerator(rescale=1./255)   

os.makedirs('/kaggle/test',exist_ok=True)
from distutils.dir_util import copy_tree
TEST_PATH
src = TEST_PATH
dest = '/kaggle/test/test'
copy_tree(src,dest)
#test_path = '/kaggle/test'
test_path = '/kaggle/test'
test_generator = test_image_gen.flow_from_directory(directory=test_path,
                                                 target_size=IMAGE_SHAPE[:2],
                                                 color_mode='rgb',
                                                 batch_size=BATCH_SIZE,
                                                 class_mode=None,
                                                  shuffle=False)
pred = model.predict(test_generator,steps=len(test_generator),verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_image_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
# get filenames 
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results
results.head()
def file_name(st):
    x = st.split('/')
    y = x[1].split('.')
    return y[0]
results['image'] = results['Filename'].apply(file_name)
results.head()
results.drop('Filename',axis=1,inplace=True)
results.head(5)
def pred(st):
    if st=='chair':
        return 1
    elif st == 'swivelchair':
        return 3
    elif st == 'bed':
        return 0
    elif st == 'table':
        return 4
    else:
        return 2
        
    
    
results['target'] = results['Predictions'].apply(pred)
results.head(100)
results.drop('Predictions',axis=1,inplace=True)
results.head(10)
results.to_csv('furn30_submission_2.csv',index=False)
os.listdir('/kaggle/working')
results.head()
type(results['image'][0])
