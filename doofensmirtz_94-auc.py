import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
%matplotlib inline
#display full output of cell not only last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# 2. Load data 

PATH = "../input/chest-xray-pneumonia/chest_xray"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "val")
test_dir = os.path.join(PATH, "test")

train_dir
validation_dir
test_dir
train_norm_dir = os.path.join(train_dir, "NORMAL")  #train directory containing normal(non pneumonia images) 
train_pne_dir = os.path.join(train_dir, "PNEUMONIA") #train directory containing pneumonia images

val_norm_dir = os.path.join(validation_dir, "NORMAL") # validation directory
val_pne_dir = os.path.join(validation_dir, "PNEUMONIA")

test_norm_dir = os.path.join(test_dir, "NORMAL") # validation directory
test_pne_dir = os.path.join(test_dir, "PNEUMONIA")

total_train = len(os.listdir(train_norm_dir)) + len(os.listdir(train_pne_dir)) #total no of training images
total_val = len(os.listdir(val_norm_dir)) + len(os.listdir(val_pne_dir)) #total no of validation images
total_test = len(os.listdir(test_norm_dir)) + len(os.listdir(test_pne_dir)) #total no of validation images

#total no of training and validation images
total_train
total_val
total_test
len(os.listdir(train_norm_dir))
len(os.listdir(train_pne_dir))

sample_img = os.path.join(train_norm_dir, "IM-0149-0001.jpeg")
show_img = load_img(sample_img)
plt.imshow(show_img)
plt.show;
# 3. preprocessing data

#setting some variables we will be requiring in training
batch_size=32
epochs = 20
img_width = 150
img_height = 150
train_img_generator = ImageDataGenerator(rescale= 1.0/255,
                                        shear_range=0.2,
                                        zoom_range=0.2,)
val_img_generator = ImageDataGenerator(rescale = 1.0/255)
test_img_generator = ImageDataGenerator(rescale = 1.0/255)
train_data_gen = train_img_generator.flow_from_directory(batch_size = batch_size,
                                                        directory = train_dir,
                                                        shuffle=True,
                                                        target_size = (img_height,img_width),
                                                        class_mode = 'binary')

val_data_gen = val_img_generator.flow_from_directory(batch_size = batch_size,
                                                    directory = validation_dir,
                                                    shuffle = True,
                                                    target_size = (img_height,img_width),
                                                    class_mode = 'binary')
test_data_gen = test_img_generator.flow_from_directory(batch_size = batch_size,
                                                    directory = test_dir,
                                                    shuffle = True,
                                                    target_size = (img_height,img_width),
                                                    class_mode = 'binary')
sample_train_imgs , label=  next(train_data_gen) 
plt.imshow(sample_train_imgs[1])
plt.show();

#training input size (batch_size , width, height, channels)

sample_train_imgs.shape
len(label)
with tf.device("/GPU:0"):
    model = tf.keras.models.Sequential([
                                     tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150,150,3),padding='same'),
                                     tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',padding='same'),
                                     tf.keras.layers.MaxPooling2D((2,2)),
                                     
                                     tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',padding='same'),
                                     tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',padding='same'),
                                     tf.keras.layers.MaxPooling2D((2,2)),
                                     
                                     tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',padding='same'),
                                     tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',padding='same'),
                                     tf.keras.layers.MaxPooling2D((2,2)),

                                     tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',padding='same'),
                                     tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',padding='same'),
                                     tf.keras.layers.MaxPooling2D((2,2)),

                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(512, activation='relu'),
                                     tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer= tf.keras.optimizers.Adam(lr = 1e-5),
              loss='binary_crossentropy',
              metrics=['acc','AUC']
               )

model.summary()
%mkdir "./cpkt"
CPKT = "./cpkt/"
callback_1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

callback_2 = tf.keras.callbacks.ModelCheckpoint(
    CPKT, monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch', options=None
)
with tf.device("/GPU:0"):
    history = model.fit_generator(train_data_gen,
                             epochs= epochs,
                             validation_data = val_data_gen,                        
                             callbacks = [callback_1, callback_2])
auc = history.history['auc']
val_auc = history.history['val_auc']

plt.figure(figsize=(5, 5))
plt.plot(auc, label='Training AUC')
plt.plot(val_auc, label='Validation AUC')
plt.legend(loc='lower right')
plt.ylabel('AUC')
plt.ylim([min(plt.ylim()),1])
plt.title('AUC')

plt.show();
result = model.evaluate(test_data_gen)
print("loss, accuracy, AUC:", result)
model.save_weights('first_train.h5')