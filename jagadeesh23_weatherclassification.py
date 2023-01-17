import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random as rd
import shutil
from google.colab import drive
drive.mount('/content/gdrive')
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Kaggle"
%cd /content/gdrive/My Drive/Kaggle
!kaggle datasets download -d jagadeesh23/weather-classification
!ls
!unzip \*.zip  && rm *.zip
!mv Data WPI
%cd 'Data'
if os.path.isdir('train') is False:  
  os.mkdir('train')
if os.path.isdir('valid') is False:  
  os.mkdir('valid')
if os.path.isdir('test') is False:  
  os.mkdir('test')

for i in range(0, 5):
    lists=os.listdir(str(i))
    if os.path.isdir('valid/'+str(i)) is False:
      os.mkdir('valid/'+str(i))
    if os.path.isdir('train/'+str(i)) is False:  
      os.mkdir('train/'+str(i))
    if os.path.isdir('test/'+str(i)) is False:
      os.mkdir('test/'+str(i))

    rd.shuffle(lists) 

    tle=int(0.8*len(lists))
    vle= tle + int(0.15*len(lists))
    tele = vle + int(0.05*len(lists))

    c=0
    for j in lists:
        if c<=tle:
          shutil.copy(str(i)+'/'+j,"train/"+str(i)+"/")
          c=c+1
        elif c>tle & c<=vle:
          shutil.copy(str(i)+'/'+j,"valid/"+str(i)+"/")
          c=c+1
        else:
          shutil.copy(str(i)+'/'+j,"test/"+str(i)+"/")
          c=c+1    

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=.2,
    )
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen= tf.keras.preprocessing.image.ImageDataGenerator()  
os.listdir()
train_generator = train_datagen.flow_from_directory('Data/train',batch_size=32,target_size=(100, 100),
                class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(
        'Data/valid', target_size=(100, 100), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('Data/test',batch_size=32,target_size=(100, 100),class_mode='categorical')
def res_identity(x, filters): 
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters

  #first block 
  x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  x =  tf.keras.layers.BatchNormalization()(x)
  x =  tf.keras.layers.Activation('relu')(x)

  #second block # bottleneck (but size kept same with padding)
  x =  tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  x =  tf.keras.layers.BatchNormalization()(x)
  x =  tf.keras.layers.Activation('relu')(x)

  # third block activation used after adding the input
  x =  tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  x =  tf.keras.layers.BatchNormalization()(x)
  x =  tf.keras.layers.Activation('relu')(x)

  # add the input 
  x =  tf.keras.layers.Add()([x, x_skip])
  x =  tf.keras.layers.Activation('relu')(x)

  return x
def res_conv(x, s, filters):
  '''
  here the input size changes''' 
  x_skip = x
  f1, f2 = filters

  # first block
  x =  tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x =  tf.keras.layers.BatchNormalization()(x)
  x =  tf.keras.layers.Activation('relu')(x)

  # second block
  x =  tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  x =  tf.keras.layers.BatchNormalization()(x)
  x =  tf.keras.layers.Activation('relu')(x)

  #third block
  x =  tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  x =  tf.keras.layers.BatchNormalization()(x)

  # shortcut 
  x_skip =  tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
  x_skip =  tf.keras.layers.BatchNormalization()(x_skip)

  # add 
  x =  tf.keras.layers.Add()([x, x_skip])
  x =  tf.keras.layers.Activation('relu')(x)

  return x

input_im = tf.keras.layers.Input(shape=(100,100,3))
x =  tf.keras.layers.ZeroPadding2D(padding=(3, 3))(input_im)

# 1st stage
# here we perform maxpooling, see the figure above

x = tf.keras.layers.Conv2D(64,kernel_size=(7, 7), strides=(2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

#2nd stage 
# frm here on only conv block and identity block, no pooling

x = res_conv(x, s=1, filters=(64, 256))
x = res_identity(x, filters=(64, 256))

# 3rd stage

x = res_conv(x, s=2, filters=(128, 512))
x = res_identity(x, filters=(128, 512))
x = res_identity(x, filters=(128, 512))

# 4th stage

x = res_conv(x, s=2, filters=(256, 1024))
x = res_identity(x, filters=(256, 1024))
x = res_identity(x, filters=(256, 1024))
x = res_identity(x, filters=(256, 1024))

# ends with average pooling and dense connection

x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(5, activation='softmax', kernel_initializer='he_normal')(x) #multi-class

# define the model 

model = tf.keras.models.Model(inputs=input_im, outputs=x, name='Resnet50')
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
checkpoint_path = "Checkpoints/training1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 2 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=2)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['AUC'])
history =model.fit(train_generator,epochs=100,validation_data=valid_generator,callbacks=[cp_callback,reduce_lr],verbose=2,steps_per_epoch=3702 // 32)
model.save('Models/my_model1.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,10)
plt.legend(['Train', 'Test'])
plt.show()
plt.savefig('loss.png')
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend(['Train','Test'])
plt.show()
plt.savefig('auc.png')
