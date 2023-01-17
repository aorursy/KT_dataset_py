#This model was trained on google colab so the cell outputs are not visible here.


#Go to https://colab.research.google.com/drive/1qGw8XtxRPir7stp-awqHBbd2XEpzC3AY?usp=sharing



import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from google.colab import files
uploaded = files.upload()
optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')
import zipfile
import io
zf = zipfile.ZipFile(io.BytesIO(uploaded['Train.zip']), "r")
zf.extractall()
train_dir='/content/Train/Train'
val_dir='/content/Train/val'
print(len(os.listdir(train_dir)))
print(len(os.listdir(val_dir)))
from tensorflow.keras.applications.inception_v3 import InceptionV3
model=InceptionV3(include_top=False,weights='imagenet',input_shape=(224,224,3),classes=45)
for layer in model.layers[0:7]:
  layer.trainable = False
for layer in model.layers[8:]:
  layer.trainable=True
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.1,
                                   shear_range=0.1,
                                   horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(train_dir,
                                                  class_mode='categorical',
                                                  target_size=(224,224),
                                                  batch_size=25,
                                                  color_mode='rgb',
                                                  shuffle = True,
                                                  seed=42)
val_generator=val_datagen.flow_from_directory(val_dir,
                                              class_mode='categorical',
                                              target_size=(224,224),
                                              batch_size=10,
                                              color_mode='rgb',
                                              shuffle = True,
                                              seed=42
                                              )
steps_train=train_generator.n//train_generator.batch_size
steps_val=val_generator.n//val_generator.batch_size
x=Flatten()(model.output)
x=Dense(2048,activation='relu')(x)
x=Dropout(0.5)(x)
pred=Dense(45,activation='softmax')(x)
model_tr = Model(model.input,pred)

model_tr.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import Callback
class myCallback(Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_accuracy') > 0.93 ):    
          self.model.stop_training = True
callbacks = myCallback()
history=model_tr.fit_generator(
    train_generator,
    steps_per_epoch=steps_train,
    validation_data=val_generator,
    validation_steps=steps_val,
    epochs=50,
    callbacks=[callbacks]
)

from google.colab import drive
drive.mount('/content/drive')
test_dir='/content/drive/My Drive/car_classification/Test/Test1'
from tensorflow.keras.preprocessing.image import load_img,img_to_array
image=load_img(os.path.join(test_dir,'image'+str(20)+'.jpg'),
                   grayscale=False,
                   color_mode="rgb",
                   target_size=(224,224))
image
input_arr = img_to_array(image)
input_arr = np.array([input_arr],np.float32)/255
np.argmax(model_tr.predict(input_arr))
pred1=[]
for i in range(0,450):
  image=load_img(os.path.join(test_dir,'image'+str(i+1)+'.jpg'),
                   grayscale=False,
                   color_mode="rgb",
                   target_size=(224,224))
  input_arr = img_to_array(image)
  input_arr = np.array([input_arr],np.float32)/255
  pred1.append(np.argmax(model_tr.predict(input_arr)))
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions1 = [labels[k] for k in pred1]
for i in range(0,450):
  print(predictions1[i],i+1)
df=pd.read_csv('/content/drive/My Drive/car_classification/sample_submission.csv')
df.head()
df.drop('predictions',axis=1)
df['predictions']=pred1
df.head()
df.to_csv("results_new1.csv",index=False)
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
model_tr.save('Inception_model.h5')
