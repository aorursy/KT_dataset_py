!unzip -q   /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip -d .

!unzip -q  /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip -d .
import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.image import load_img,img_to_array

import numpy as np

import os

import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

import time

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import shutil

from keras.preprocessing.image import ImageDataGenerator



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print("GPU is {}".format(tf.config.list_physical_devices('GPU')))

print("tensorflow version {}".format(tf.__version__))



print(os.listdir("./"))



!nvidia-smi



keras.backend.clear_session()


def show_cats_and_dogs(show="",width=150,height=150, images_path ='./train/'):

  cols = 25

  limit = 100

  index = 0

  images = list()

  vertical_images=[]

 

  for path in os.listdir(images_path):

    if show != "" and  (show in path)==False:

          continue

    index=index+1

    if index%limit==0:

        break

    #keras.preprocessing.image

    image = load_img(images_path+path, target_size=(width,height))

    image= img_to_array(image) #to numpy

    image_height, image_width, image_channel = image.shape

    horizontal_side = np.ones((image_height, 5,  image_channel), dtype=np.float32)*255

    

    images.append(image)

    images.append(horizontal_side)



    if index%cols==0:

      horizontal_image = np.hstack((images))

      image_height, image_width, image_channel = horizontal_image.shape

      vertical_side = np.ones((5, image_width,  image_channel), dtype=np.float32)*255

      vertical_images.append(horizontal_image)

      vertical_images.append(vertical_side)

      images=list()

  gallery=np.vstack((vertical_images)) 

  plt.figure(figsize=(12,12))

  plt.xticks([])

  plt.yticks([])

  title={"":"cães & gatos",

          "cat": "gatos",

          "dog": "cães"}

  plt.title("{} imagens de {} [ path {} ] .".format(limit, title[show],images_path))

  plt.imshow(gallery.astype(np.uint8))
# raw Dataset

print("O dataset possui {} imagens de gatos e cães para classificação.".format(len(os.listdir("./train"))))

print("O dataset de teste possui {}.".format(len(os.listdir("./test"))))

show_cats_and_dogs(show='cat')
show_cats_and_dogs(show='dog')
show_cats_and_dogs(show='')
show_cats_and_dogs(images_path='./test/')
image_width,image_height = 150,150#299,299

labels =['dog','cat']

for d in labels:

  dir_path = './train/' + d

  if not os.path.exists(dir_path):

    print('{} criado.'.format(dir_path))

    os.mkdir(dir_path)

  else:

    print('{} já existe.'.format(dir_path))





train_path ="./train/"

for  file in  os.listdir(train_path):

  category = file.split(".")[0]

  if '.jpg' in file:

    if 'dog'in category: 

      shutil.copyfile(train_path+file,'./train/dog/'+ file)

    elif 'cat'in category:  

      shutil.copyfile(train_path+file,'./train/cat/'+ file)

print("Total de cães:\t{}".format(sum([len(files) for r, d, files in os.walk('./train/dog/')])))

print("Total de gatos:\t{}".format(sum([len(files) for r, d, files in os.walk('./train/cat/')])))
keras.backend.clear_session()

batch_size=64

validation_split=0.3

val_size = 7500

dataset_size = 17500 

train_data_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=validation_split)



train_datagenerator = train_data_generator.flow_from_directory(train_path,

                                                     target_size=(image_width,image_height ),

                                                     class_mode="categorical",

                                                     batch_size=batch_size,

                                                     shuffle=True,

                                                     subset='training')



val_datagenerator = train_data_generator.flow_from_directory(train_path,

                                                     target_size=(image_width,image_height),

                                                     class_mode="categorical",

                                                     shuffle=True,

                                                     batch_size=batch_size,

                                                     subset='validation')

inception_v3_model = keras.applications.InceptionV3(include_top=False, weights='imagenet',input_shape=(image_width,image_height,3))
keras.backend.clear_session()

x = inception_v3_model.output

avg_pool2d = keras.layers.GlobalAveragePooling2D()(x)

dense = keras.layers.Dense(512, activation= keras.activations.relu)(avg_pool2d)

output = keras.layers.Dense(2,activation=keras.activations.softmax)(dense)

model = keras.Model(inputs=inception_v3_model.input, outputs=output,name = "transfer_inception_v3")

freeze= np.round((len(model.layers)-len(model.layers)*0.3),0).astype('int') 

for layer in model.layers[:freeze]:

    layer.trainable =False

for layer in model.layers[freeze:]:

    layer.trainable=True

model.summary()
tf.keras.utils.plot_model( model)
plateau_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=2, factor=.5, min_lr=.00001)



start = time.time()



model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), 

              optimizer=keras.optimizers.RMSprop(lr=0.0005, decay = 1e-6, momentum = 0.9),

              metrics=['accuracy'])



history = model.fit(train_datagenerator,

                  steps_per_epoch=(dataset_size//batch_size),

                  epochs= 5, 

                  verbose=1,

                  validation_data=val_datagenerator,

                  validation_steps=(val_size//batch_size),

                  callbacks=[plateau_callback]

)                                              

                                                                                          



print("training: ",time.time()-start)
print("Train Accuracy:{:.3f}".format(history.history['accuracy'][-1]))

print("Val Accuracy:{:.3f}".format(history.history['val_accuracy'][-1]))

print('')

print("Train Loss:{:.3f}".format(history.history['loss'][-1]))

print("Val Loss:{:.3f}".format(history.history['val_loss'][-1]))
score = model.evaluate_generator(val_datagenerator,verbose=1)

print('Val loss: ', score[0])

print('Val accuracy', score[1])
epochs = list(range(1,len(history.history['accuracy'])+1))

epochs

plt.plot(epochs, history.history['accuracy'],epochs,history.history['val_accuracy'])

plt.legend(('Training','Validation'))

plt.show()
epochs = list(range(1,len(history.history['loss'])+1))

epochs

plt.plot(epochs, history.history['loss'],epochs,history.history['val_loss'])

plt.legend(('Training','Validation'))

plt.show()
test_path ="./test/"

if not os.path.exists("./test"):

  os.mkdir("./test")

  print('./test criado.')



dir_path = "./test/data"

if not os.path.exists(dir_path):

  print('{} criado.'.format(dir_path))

  os.mkdir(dir_path)

else:

  print('{} já existe.'.format(dir_path))

for file in os.listdir(test_path):

    if '.jpg' in file:

        shutil.copyfile(test_path+file,dir_path+'/'+file)



print("Total de gatos:\t{}".format(sum([len(files) for r, d, files in os.walk(dir_path+'/')])))



test_path = dir_path+'/'

test_data_generator = ImageDataGenerator(rescale=1./255)



test_generator = test_data_generator.flow_from_directory(directory ='./test',

                                                         target_size=(image_width,image_height),

                                                     batch_size=batch_size,

                                                     class_mode=None,

                                                     shuffle=False)
predict = model.predict(test_generator,verbose=1)
index = 56

path= test_generator.filenames[index]

plt.figure(figsize=(4, 4))

img=load_img('./test/'+path, target_size=(image_width,image_height))

plt.imshow(img)

if (predict[index,1]) >= 1.:

    label='Dog'

else:

    label='Cat'

plt.title("Class: {}".format(label))

plt.show()
submission = pd.DataFrame({

    'id':pd.Series(test_generator.filenames),

    'label':pd.Series(predict[:,1])

    })

submission['id'] = submission.id.str.extract('(\d+)')

submission['id']=pd.to_numeric(submission['id']).astype('int')

submission['label']=pd.to_numeric(submission['label']).astype('int')

submission.to_csv("submission_v9.csv",index=False)
submission.head(10)
shutil.rmtree("./test")

shutil.rmtree("./train")