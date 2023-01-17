import numpy as np 
import tensorflow as tf
import cv2
import os 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import save
def download(path):
  '''
  Input: The path for the dataset
  
  Output: y_train ; It is a pandas dataframe consisting of the y labels.
          x_train ; It is a tensor with image data.
  '''
  data = pd.DataFrame({0:['n99999999_001.JPEG'], 
                       1:[5],
                       2:[0],
                       3:[0]}) 
  image = None
  for (root,dirs,files) in os.walk(path): 
    if(root != path):
      if(dirs == []):
        for file in files:
          print(file)
          if(image != None):
            a = root+'/' + file
            image_raw = tf.io.read_file(a)
            new = tf.image.decode_image(image_raw)
            if(tf.shape(new)[2] == 1):
              new = tf.image.grayscale_to_rgb(new, name=None)
            new = tf.reshape(new,[1,64,64,3])
            image = tf.concat([image,new],0)
    
          
          else:
            a = root+'/' + file
            image_raw = tf.io.read_file(a)
            image = tf.image.decode_image(image_raw)
            if(tf.shape(image)[2] == 1):
              image = tf.image.grayscale_to_rgb(image, name=None)
            image = tf.reshape(image,[1,64,64,3])
     
     
      else:
        for dir in files:
          a = root + '/' + dir 
          data = data.append(pd.read_csv(a,delimiter="\t",header = None),ignore_index = True)
          
  y_train = data 
  x_train = image
  return y_train,x_train
     
         


def Y_dataframe(Y_values):

 '''
Input : It is a pandas dataframe that countains the lables and bounding boxes of each image in the file.

Output: Y and Y_keys 
        Y_ keys is a  dataframe that has all the labels.
        Y is a  dataframe that holds all the one hot codes and its respective labels. 
 '''
 
 Y_values[0] = Y_values[0].str[:9]
 Y_values.drop(Y_values.index[:1], inplace=True)

 # Key for the dataset 
 image_types = Y_values[0].unique()
 Y_train = pd.DataFrame(image_types, columns=['Image_Types'])
 labelencoder = LabelEncoder()
 Y_train['Image_Types_labels'] = labelencoder.fit_transform(Y_train['Image_Types'])
 Y_keys = Y_train
 
 image_types = np.array(Y_values[0])
 Y = pd.DataFrame(image_types, columns=['Image_Types'])
 labelencoder = LabelEncoder()
 Y['Image_Types_labels'] = labelencoder.fit_transform(Y['Image_Types'])
 
    
 enc = OneHotEncoder(handle_unknown='ignore')
 enc_df = pd.DataFrame(enc.fit_transform(Y[['Image_Types_labels']]).toarray())
 Y = Y.join(enc_df)
 
 return Y,Y_keys
# Seperate Code for validation y_values 

def dow_val():
 Y_values = pd.read_csv('../input/image-detect/val/val_annotations.txt',delimiter="\t",header = None)
# Y_values = Y_values.sort_values([0])

 # Key for the dataset 
 image_types = Y_values[1].unique()
 Y_train = pd.DataFrame(image_types, columns=['Image_Types'])
 labelencoder = LabelEncoder()
 Y_train['Image_Types_labels'] = labelencoder.fit_transform(Y_train['Image_Types'])
 Y_keys = Y_train

 image_types = np.array(Y_values[1])

 Y = pd.DataFrame(image_types, columns=['Image_Types'])
 labelencoder = LabelEncoder()
    
 Y['Image_Types_labels'] = labelencoder.fit_transform(Y['Image_Types']) 
 enc = OneHotEncoder(handle_unknown='ignore')
 enc_df = pd.DataFrame(enc.fit_transform(Y[['Image_Types_labels']]).toarray())
 Y = Y.join(enc_df)
 
 return Y,Y_keys
 

def hot_encode(X_train):
 arra = X_train[X_train.columns[2:]]
 arra = np.array(arra)
 return arra 
# First downloading data from drive 
y_train,X_train = download('../input/image-detect/train')
y_val,X_val = download('../input/image-detect/val')

# Assembling the Y values in the right order
y_train,y_train_keys = Y_dataframe(y_train)
y_val,y_val_keys =  dow_val()

y_val,X_val = download('../input/image-detect/val')
for (root,dirs,files) in os.walk('../input/image-detect/val/images'):
    print(files)
import glob
for filepath in glob.iglob(r'../input/image-detect/val/images/*.JPEG'):
    print(filepath)
Y_train = hot_encode(y_train)
Y_val = hot_encode(y_val)
X_train = X_train/255
X_val = X_val/255

save('X_val_unsortted.npy',X_val)
save('Y_val_unsortted',Y_val)



print(X_train.dtype)

tf.dtypes.cast(X_train, tf.float16)
print(X_train.dtype)
import numpy as np 
import tensorflow as tf
from numpy import load
import pandas as pd
import gc
#Downloading X_train,X_val,Y_train,Y_val
X_train = load('../input/pre-trained-data/X_train.npy')
X_val = load('../input/last-data/X_train(3).npy')/255
Y_train = load('../input/pre-trained-data/Y_train.npy')
Y_val = load('../input/validation/Y_val_unsortted.npy')

#Dowloading Key values dataframe 
Y_train_keys = pd.read_csv('../input/pre-trained-data/y_train_keys.txt',delimiter="\t")
Y_val_keys =pd.read_csv('../input/pre-trained-data/y_val_keys.txt',delimiter="\t")
inputs = tf.keras.layers.Input(shape = (64, 64,3))

layer1 = tf.keras.layers.Conv2D(42,(3,3),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',padding = 'same')(inputs)
MaxPool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1),padding = 'same')(layer1)


layer2 = tf.keras.layers.Conv2D(42,(3,3),activation = 'relu',use_bias = 1,kernel_regularizer= 'l2',padding = 'same')(MaxPool1)
MaxPool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1),padding = 'same')(layer2)


#Inception Layer 1
Inception_layer1_con_1 = tf.keras.layers.Conv2D(42,(3,3),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(MaxPool2)
Inception_layer1_con_2 = tf.keras.layers.Conv2D(42,(5,5),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(MaxPool2)
MaxPool3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1),padding = 'same')(MaxPool2)
Concatenate1 = tf.keras.layers.Concatenate(axis=-1)([Inception_layer1_con_1,Inception_layer1_con_2,MaxPool3])

#Inception Layer 2 
Inception_layer2 = tf.keras.layers.Conv2D(64,(1,1),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(Concatenate1)

Inception_layer2_con_1 = tf.keras.layers.Conv2D(42,(3,3),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(Inception_layer2)
Inception_layer2_con_2 = tf.keras.layers.Conv2D(42,(5,5),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(Inception_layer2)
MaxPool4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1),padding = 'same')(Inception_layer2)
Concatenate2 = tf.keras.layers.Concatenate(axis=-1)([Inception_layer2_con_1,Inception_layer2_con_2,MaxPool4])

Concatenate3 =  tf.keras.layers.Concatenate(axis=-1)([Concatenate1, Concatenate2])
layer3 = tf.keras.layers.Conv2D(64,(1,1),activation = 'relu',use_bias = 1,kernel_regularizer = 'l2',strides = (1,1),padding ='same')(Concatenate3)

#Fully Connected Layers
Flatten =  tf.keras.layers.Flatten()(layer3)
hidden_1 = tf.keras.layers.Dense(20, activation = 'relu')(Flatten)
dropout = tf.keras.layers.Dropout(rate = 0.1)(hidden_1)
outputs = tf.keras.layers.Dense(200, activation = tf.keras.activations.softmax)(dropout)


model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.summary()
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1./255,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)

checkpoint_filepath = './checkpoint.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose = 1)

model.load_weights('./checkpoint')
model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.02), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
print('done')
his = model.fit(datagen.flow(X_train, Y_train, batch_size=500),steps_per_epoch=len(X_train) /500, epochs=800,validation_data=(X_val,Y_val),shuffle = True,callbacks=[model_checkpoint_callback])