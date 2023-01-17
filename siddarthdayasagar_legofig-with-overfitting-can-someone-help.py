import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
import cv2

os.listdir('/kaggle/input/lego-minifigures-classification/')

index=pd.read_csv('../input/lego-minifigures-classification/index.csv')
index.head()
metadata=pd.read_csv('../input/lego-minifigures-classification/metadata.csv')
metadata.head()
df = pd.merge(index, metadata[['class_id', 'minifigure_name']], on='class_id')
df.head()

#CHECKING IF THE IS ANY MISSING VALUES
df.isnull().sum()
hero_name=df['minifigure_name'].unique()
hero_name
count=df['minifigure_name'].value_counts()
count

plt.figure(figsize=(12,10))
sns.barplot(x=hero_name, y=count,palette='rocket')

plt.xticks(rotation= 90)
plt.xlabel('SUPERHERO')
plt.ylabel('Count')
plt.title('Dataset Analysis')
plt.show()
HERO=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(512,512,3)),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(0.2),
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(0.2),
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(0.2),
                                
                                tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(512,activation='relu'),
                                tf.keras.layers.Dense(31,activation='softmax')])
HERO.summary()
#Splitting into train and validation

training = df[df["train-valid"] == 'train']
validation = df[df["train-valid"] == 'valid']

trainD = np.zeros((training.shape[0], 512, 512, 3))

for i in range(training.shape[0]):
    
    image = cv2.imread('../input/lego-minifigures-classification/' + training["path"].values[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512,512))
    

    trainD[i] = image / 255.0

trainL = np.array(training["class_id"])-1


validD = np.zeros((validation.shape[0], 512, 512, 3))

for i in range(validation.shape[0]):
    
    image = cv2.imread('../input/lego-minifigures-classification/' + validation["path"].values[i])
    
    #Converting BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Resizing image to (512 x 512)
    image = cv2.resize(image, (512,512))
    
    #Normalizing pixel values to [0,1]
    validD[i] = image / 255.0

validL = np.array(validation["class_id"])-1

class mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epochs,logs={}):
        if(logs.get('accuracy')>1.0):
            self.model.stop_training=True
def create_model(input_shape):
    # initialize the base model as VGG16 model with input shape as (512,512,3)
    base_model = tf.keras.applications.MobileNetV2(input_shape = input_shape,
                       include_top = False,
                       weights = 'imagenet')

    # we do not have to train all of the layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.Dense(31, activation = 'softmax')(x)
    
    return tf.keras.models.Model(base_model.input,x)
model = create_model((512,512,3))
callbacks=mycallbacks()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
             loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
history=model.fit(trainD,trainL,epochs=200,validation_data=(validD, validL),callbacks=[callbacks],shuffle=True,batch_size=5)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
