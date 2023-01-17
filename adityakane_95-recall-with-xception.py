! pip install -q kaggle
from google.colab import files

files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia  

zip_ref   = zipfile.ZipFile('/content/chest-xray-pneumonia.zip', 'r')
zip_ref.extractall('/tmp')

#!rm -r '/content/drive/My Drive/pneumonia/chest_xray/chest_xray/'
import os
import zipfile
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
normal_dir=os.path.join('../input/chest-xray-pneumonia/chest_xray/train/NORMAL')
pneumonia_dir=os.path.join('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/')

pic_index = 4

next_normal_dir=[os.path.join(normal_dir,fname) for fname in os.listdir(normal_dir)[pic_index-2:pic_index]]
next_pneumonia_dir=[os.path.join(pneumonia_dir,fname) for fname in os.listdir(pneumonia_dir)[pic_index-2:pic_index]]


for i, img_path in enumerate(next_normal_dir+next_pneumonia_dir):
  #print(img_path)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
train_dir=os.path.join('../input/chest-xray-pneumonia/chest_xray/train/')
validation_dir=os.path.join('../input/chest-xray-pneumonia/chest_xray/test/')

train_datagen=ImageDataGenerator(rescale=1.0/255.0,
                                 rotation_range=40,
                                 width_shift_range=0.2,            
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
validation_datagen=ImageDataGenerator(rescale=1.0/255.0,)

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),class_mode='binary')
validation_generator=validation_datagen.flow_from_directory(validation_dir,target_size=(150,150),class_mode='binary')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.96):
            print("\nReached 90.0% accuracy so cancelling training!")
            self.model.stop_training = True
callbk=myCallback()


model=keras.Sequential()  
xception=tf.keras.applications.Xception(input_shape=(150,150,3),include_top=False)
xception.trainable=False
model.add(xception)
model.add(keras.layers.Conv2D(64,(1,1),activation='relu'))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(8,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.Recall()],
                optimizer='sgd')
model.summary()




import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
SVG(model_to_dot(model).create(prog='dot', format='svg'))
Utils.plot_model(model,show_shapes=True)
with tf.device('/device:GPU:0'):
    import datetime
    a = datetime.datetime.now()

    history = model.fit_generator(train_generator,validation_data=validation_generator,
                                    epochs=55,verbose = 1,callbacks=[callbk])
    b = datetime.datetime.now()
    print(b-a)
 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss (categorical)')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



test_dir=os.path.join('../input/chest-xray-pneumonia/chest_xray/test/')
test_datagen=ImageDataGenerator(rescale=1.0/255.0,)
test_generator=test_datagen.flow_from_directory(test_dir,target_size=(150,150),class_mode='binary')


model.evaluate(x=test_generator, verbose=0,return_dict=True)
