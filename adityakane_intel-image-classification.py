
import os
import random
import zipfile
import random
from shutil import copyfile
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
!rm -r '/content/drive/My Drive/data/intel'
zip_ref   = zipfile.ZipFile('/content/intel-image-classification.zip', 'r')
zip_ref.extractall('/content/drive/My Drive/intel/')

building_dir=os.path.join('../input/intel-image-classification/seg_train/seg_train/buildings/')
forest_dir=os.path.join('../input/intel-image-classification/seg_train/seg_train/forest/')
sea_dir=os.path.join('../input/intel-image-classification/seg_train/seg_train/sea/')
glacier_dir=os.path.join('../input/intel-image-classification/seg_train/seg_train/glacier/')
mountain_dir=os.path.join('../input/intel-image-classification/seg_train/seg_train/mountain/')
street_dir=os.path.join('../input/intel-image-classification/seg_train/seg_train/street/')



pic_index = 4

next_building_dir=[os.path.join(building_dir,fname) for fname in os.listdir(building_dir)[pic_index-2:pic_index]]
next_forest_dir=[os.path.join(forest_dir,fname) for fname in os.listdir(forest_dir)[pic_index-2:pic_index]]
next_sea_dir=[os.path.join(sea_dir,fname) for fname in os.listdir(sea_dir)[pic_index-2:pic_index]]
next_glacier_dir=[os.path.join(glacier_dir,fname) for fname in os.listdir(glacier_dir)[pic_index-2:pic_index]]
next_mountain_dir=[os.path.join(mountain_dir,fname) for fname in os.listdir(mountain_dir)[pic_index-2:pic_index]]
next_street_dir=[os.path.join(street_dir,fname) for fname in os.listdir(street_dir)[pic_index-2:pic_index]]

for i, img_path in enumerate(next_building_dir+next_forest_dir+next_sea_dir+next_glacier_dir+next_mountain_dir+next_street_dir):
  #print(img_path)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
train_dir=os.path.join('../input/intel-image-classification/seg_train/seg_train/')
validation_dir=os.path.join('../input/intel-image-classification/seg_test/seg_test/')

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=50,
                                 width_shift_range=0.3,
                                 height_shift_range=0.3,
                                 shear_range=0.3,
                                 zoom_range=0.3,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),class_mode='sparse')
validation_generator=validation_datagen.flow_from_directory(validation_dir,target_size=(150,150),class_mode='sparse')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.9):
            print("\nReached 90.0% accuracy so cancelling training!")
            self.model.stop_training = True
callbk=myCallback()
resnet=tf.keras.applications.InceptionResNetV2(input_shape=(150,150,3),include_top=False)
resnet.trainable=False #freeze downloaded network
model=keras.Sequential()
model.add(resnet)
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(576,activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(192,activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(48,activation='relu'))
model.add(keras.layers.Dense(6,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],
              optimizer='adam')
model.summary()
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
SVG(model_to_dot(model).create(prog='dot', format='svg'))
Utils.plot_model(model,to_file='./inception_resnet_batchnorm.png',show_shapes=True)
import datetime
a = datetime.datetime.now()

history = model.fit_generator(train_generator,validation_data=validation_generator,
                              epochs=40,verbose = 1,callbacks=[callbk])
b = datetime.datetime.now()
print(b-a)
model.save('./intel_inception_resnet_adam_batchnorm.h5')
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

fig, axs = plt.subplots(5, 5,figsize=(15,15))
images=os.listdir('../input/intel-image-classification/seg_pred/seg_pred/')

label_codes={ 0:'buildings', 1:'forest', 2:'glacier',3:'mountain', 4:'sea', 5:'street'}
to_print=list(random.sample(images,25))
def GetKey(val):
   for key, value in label_codes.items():
      if val == value:
         return key
      return "None of these"
w=50
h=50
#fig=plt.figure(figsize=(40, 40))
columns = 5
rows = 5
for i in range(0, columns*rows ):
    img = image.load_img('../input/intel-image-classification/seg_pred/seg_pred/'+str(to_print[i-1]), target_size=(150, 150))
    #fig.add_subplot(rows, columns, i+1)
    axs[i//5, i%5].imshow(img)
    pred_img = image.img_to_array(img)
    pred_img = np.expand_dims(pred_img, axis=0)
    tf.image.resize(pred_img, [150,150])
    pred_img/=255
    axs[i//5, i%5].set_title('Prediction is: '+label_codes[np.argmax(model.predict(pred_img))],y=0, pad=-15)
    axs[i//5, i%5].axis('off')
    #axs[i//5, i%5].set(xlabel='Prediction is: '+label_codes[np.argmax(model.predict(pred_img))])

#plt.show()
        
