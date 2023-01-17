from os import listdir

import tensorflow as tf

import keras

import cv2

import matplotlib.pyplot as plt

import numpy as np

from keras.models import load_model

import keras.layers as L
plt.figure(figsize=(9,6))

for id, img in enumerate(listdir('../input/cat-and-dog/training_set/training_set/cats/')[4:7]):

    img = cv2.imread('../input/cat-and-dog/training_set/training_set/cats/'+img)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    plt.subplot(2,3,id+1)

    plt.imshow(img)

for id, img in enumerate(listdir('../input/cat-and-dog/training_set/training_set/dogs/')[5:8]):

    img = cv2.imread('../input/cat-and-dog/training_set/training_set/dogs/'+img)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    plt.subplot(2,3,id+4)

    plt.imshow(img)
len_train_set = len(listdir('../input/cat-and-dog/training_set/training_set/dogs/'))+len(listdir('../input/cat-and-dog/training_set/training_set/cats/'))

len_test_set = len(listdir('../input/cat-and-dog/test_set/test_set/dogs/'))+len(listdir('../input/cat-and-dog/test_set/test_set/cats/'))
print(f'length of test set:{len_test_set}')

print(f'length of training set:{len_train_set}')
IMG_SIZE = 250
# Initialising the CNN

classifier = keras.models.Sequential()



# Step 1 - Convolution

classifier.add(L.Convolution2D(32, 3, 3, input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'relu'))



# Step 2 - Pooling

classifier.add(L.MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional block

classifier.add(L.Convolution2D(32, 3, 3, activation = 'relu'))

classifier.add(L.MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(L.Flatten())



# Step 4 - Full connection

classifier.add(L.Dense(128, activation = 'relu'))

classifier.add(L.Dense(1, activation = 'sigmoid'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../input/cat-and-dog/training_set/training_set',

                                                 target_size = (IMG_SIZE, IMG_SIZE),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory('../input/cat-and-dog/test_set/test_set',

                                            target_size = (IMG_SIZE, IMG_SIZE),

                                            batch_size = 32,

                                            class_mode = 'binary')

chckpt_path = 'model.CNN.hdf5'



chckpt  = keras.callbacks.ModelCheckpoint(chckpt_path, save_best_only=True,monitor = 'val_accuracy',mode='max')

classifier.fit_generator(training_set,

                         steps_per_epoch = len_train_set/32,

                         epochs = 10,

                         validation_data = test_set,

                         validation_steps = len_test_set/32,

                         callbacks=[chckpt]

                         )
model = load_model('model.CNN.hdf5')
model.summary()
def image_reader(path='../input/cat-and-dog/training_set/training_set/'):

    images= np.array(listdir(path+'dogs/')[1:]+listdir(path+'cats/'))

    np.random.shuffle(images)

    labels = {'dog':1,'cat':0}

    for image in images:

        if(image.startswith('dog') or image.startswith('cat')):

            label = image.split('.')[0]

            img_path = path+ label+'s'+'/'+image



            img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

            img = keras.applications.vgg16.preprocess_input(img)

            yield img,labels[label]
def batch_generator(items,batch_size):

    a=[]

    i=0

    for item in items:

        a.append(item)

        i+=1



        if i%batch_size==0:

            yield a

            a=[]

    if len(a) is not 0:

        yield a
def data_generator(batch_size,path='../input/cat-and-dog/training_set/training_set/'):

    while True:

        for bat in batch_generator(image_reader(path),batch_size):

            batch_images = []

            batch_labels = []

            for im,im_label in bat:

                batch_images.append(im)

                batch_labels.append(im_label)

            batch_images = np.stack(batch_images,axis=0)

            batch_labels =  np.stack(batch_labels,axis=0)

            yield batch_images,batch_labels
train_path = '../input/cat-and-dog/training_set/training_set/'

test_path = '../input/cat-and-dog/test_set/test_set/'
def VGG():

    model = keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(250,250,3))

    new_output = keras.layers.GlobalAveragePooling2D()(model.output)



    new_output = keras.layers.Dense(256,activation='relu')(new_output)

    new_output = keras.layers.Dense(64,activation='relu')(new_output)

    new_output = keras.layers.Dense(1,activation='sigmoid')(new_output)



    model = keras.engine.training.Model(model.inputs,new_output)

    return model
model = VGG()
tf.keras.utils.plot_model(model)
for layer in model.layers:

    layer.trainable=True

for layer in model.layers[:-8]:

    layer.trainable=False
model.summary()
model.compile(optimizer=keras.optimizers.Adamax(lr=1e-2),loss='binary_crossentropy',metrics=["accuracy"])
ch_path = 'vgg16.best.hdf5'

chckpt = keras.callbacks.ModelCheckpoint(ch_path,save_best_only=True,monitor='val_accuracy',mode='max')


model.fit_generator(data_generator(32,train_path),

          steps_per_epoch=len_train_set/32,

          epochs=10,

          validation_data=data_generator(32,test_path),

          validation_steps=len_test_set/32,

          callbacks=[chckpt]

          )


def image_reader(path='../input/cat-and-dog/training_set/training_set/'):

    images= np.array(listdir(path+'dogs/')[1:]+listdir(path+'cats/'))

    np.random.shuffle(images)

    labels = {'dog':1,'cat':0}

    for image in images:

        if(image.startswith('dog') or image.startswith('cat')):

            label = image.split('.')[0]

            img_path = path+ label+'s'+'/'+image

    

            img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

            img = keras.applications.inception_v3.preprocess_input(img)

            yield img,labels[label]

    
def Inception():

    model=keras.applications.InceptionV3(include_top=False,weights='imagenet',input_shape=(250,250,3))

    

    new_output=keras.layers.GlobalAveragePooling2D()(model.output)

    

    new_output = keras.layers.Dense(4,activation="relu")(new_output)

    new_output=keras.layers.Dense(1,activation='sigmoid')(new_output)

    model=keras.engine.training.Model(model.inputs,new_output)

    

    return model
model = Inception()
for layer in model.layers:

    layer.trainable=True

    

    if isinstance(layer,keras.layers.BatchNormalization):

        layer.momentum=0.9

for layer in model.layers[:-50]:

    if not isinstance(layer,keras.layers.BatchNormalization):

        layer.trainable=False
print(f'Number of layers: {len(model.layers)}\nNumber of Parameters: {model.count_params()}')
model.compile(loss='binary_crossentropy',

             optimizer=keras.optimizers.Adamax(lr=1e-2),

             metrics=['accuracy'])
ch_path = 'model.best.hdf5'

chckpt = keras.callbacks.ModelCheckpoint(ch_path, save_best_only=True, monitor='val_accuracy', mode='max')


model.fit_generator(data_generator(32,train_path),

          steps_per_epoch=len_train_set/32,

          epochs=10,

          validation_data=data_generator(32,test_path),

          validation_steps=len_test_set/32,

          callbacks=[chckpt]

          )
inception = load_model('model.best.hdf5')
def util_predict(prob):

    if(prob>0.5):

        return f'dog : {prob[0][0]}'

    else:

        return f'cat : {1-prob[0][0]}'
!wget -c "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg" -O 'cat0.jpg'

!wget -c "https://d3544la1u8djza.cloudfront.net/APHI/Blog/2016/10_October/persians/Persian+Cat+Facts+History+Personality+and+Care+_+ASPCA+Pet+Health+Insurance+_+white+Persian+cat+resting+on+a+brown+sofa-min.jpg" -O 'cat1.jpg'

!wget -c "https://img.webmd.com/dtmcms/live/webmd/consumer_assets/site_images/article_thumbnails/other/cat_relaxing_on_patio_other/1800x1200_cat_relaxing_on_patio_other.jpg" -O 'cat2.jpg'
cats = ['cat0.jpg','cat1.jpg','cat2.jpg']

plt.figure(figsize=(9,5))

for id,cat in enumerate(cats):

    img = cv2.cvtColor(cv2.imread(cat),cv2.COLOR_BGR2RGB)

    plt.subplot(1,3,id+1)

    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

    plt.imshow(img)

    img = keras.applications.inception_v3.preprocess_input(img)

    plt.title(util_predict(inception.predict(np.stack([img],axis=0))))
!wget -c "https://post.medicalnewstoday.com/wp-content/uploads/sites/3/2020/02/322868_1100-1100x628.jpg" -O 'dog0.jpg'

!wget -c "https://i.insider.com/5484d9d1eab8ea3017b17e29?width=1100&format=jpeg&auto=webp" -O 'dog1.jpg'

!wget -c "https://scx2.b-cdn.net/gfx/news/hires/2018/2-dog.jpg" -O 'dog2.jpg'
dogs = ['dog0.jpg','dog1.jpg','dog2.jpg']

plt.figure(figsize=(9,5))

for id,dog in enumerate(dogs):

    img = cv2.cvtColor(cv2.imread(dog),cv2.COLOR_BGR2RGB)

    plt.subplot(1,3,id+1)

    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

    plt.imshow(img)

    img = keras.applications.inception_v3.preprocess_input(img)

    plt.title(util_predict(inception.predict(np.stack([img],axis=0))))