# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/chest_xray/chest_xray/"))
print(os.listdir("../input/chest_xray/chest_xray/val"))
len(os.listdir("../input/chest_xray/chest_xray/val/NORMAL")), len(os.listdir("../input/chest_xray/chest_xray/val/PNEUMONIA"))
from PIL import Image
img = Image.open("../input/chest_xray/chest_xray/val/NORMAL/"+os.listdir("../input/chest_xray/chest_xray/val/NORMAL")[0])



img
img = Image.open("../input/chest_xray/chest_xray/val/PNEUMONIA/"+os.listdir("../input/chest_xray/chest_xray/val/PNEUMONIA")[0])

img
base_dir = "../input/chest_xray/chest_xray/"



train_dir = base_dir+"/train"

val_dir = base_dir+"/val"

test_dir = base_dir+"/test"

os.listdir(val_dir)




#len(os.listdir(train_dir)), len(os.listdir(val_dir)), len(os.listdir(test_dir))

os.listdir(train_dir)



tr_norm, tr_pneu = os.listdir(train_dir+'/NORMAL'), os.listdir(train_dir+'/PNEUMONIA')

val_norm, val_pneu = os.listdir(val_dir+'/NORMAL'), os.listdir(val_dir+'/PNEUMONIA')

test_norm, test_pneu = os.listdir(test_dir+'/NORMAL'), os.listdir(test_dir+'/PNEUMONIA')



len(tr_norm), len(tr_pneu), len(val_norm), len(val_pneu), len(test_norm), len(test_pneu)
from keras.preprocessing.image import ImageDataGenerator
# val_datagen is no good



# train_datagen = ImageDataGenerator(rescale=1./255)

# val_datagen = ImageDataGenerator(rescale=1./255)



# train_generator = train_datagen.flow_from_directory(

#     train_dir,

#     target_size=(150,150),

#     batch_size=20,

#     class_mode='binary')



# val_generator = val_datagen.flow_from_directory(

#     val_dir,

#     target_size=(150,150),

#     batch_size=5,

#     class_mode='binary')



batch_size = 20

img_height, img_width = 150,150

train_data_dir = train_dir



# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

train_datagen = ImageDataGenerator(rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.2) # set validation split



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary',

    subset='training') # set as training data



validation_generator = train_datagen.flow_from_directory(

    train_data_dir, # same directory as training data

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary',

    subset='validation') # set as validation data



for data_batch, labels_batch in train_generator:

    print('data shape', data_batch.shape)

    print('data label', labels_batch)

    break
from keras import layers

from keras import models

from keras.layers import normalization

def make_model():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu',

                           input_shape=(150, 150, 3)))

    # middle layers

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    # output layer

    model.add(layers.Flatten())

    model.add(layers.Dropout(.5))

    model.add(layers.Dense(16, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    

    return model



def make_model_2():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu',

                           input_shape=(150, 150, 3)))

    # middle layers

    #model.add(normalization.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    #model.add(normalization.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))



    model.add(layers.Conv2D(128, (3,3), activation='relu'))

    #model.add(normalization.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(128, (3,3), activation='relu'))

    #model.add(layers.Conv2D(128, (3,3), activation='relu'))

    #model.add(normalization.BatchNormalization())

    model.add(layers.MaxPooling2D((2,2)))

    

    # output layers

    model.add(layers.Flatten())

    model.add(layers.Dropout(.2))

    model.add(layers.Dense(128, activation='relu'))

    #model.add(layers.Dropout(.2))

    #model.add(layers.Dense(32, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    

    return model

    

#model = make_model()

model = make_model_2()
from keras import optimizers

from keras.callbacks import ModelCheckpoint
model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['acc'])
model.summary()
# checkpoint best models

checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', 

                             verbose=1, monitor='val_loss', 

                             save_best_only=True, mode='auto')
# train



#history = model.fit_generator(train_generator,

#                             steps_per_epoch=100,

#                             epochs=30,

#                             validation_data=val_generator,

#                             validation_steps=50)



history = model.fit_generator(train_generator,

                             steps_per_epoch=100,

                             epochs=30,

                             validation_data=validation_generator,

                             validation_steps=50,

                             callbacks=[checkpoint])

!ls -lh
model.save('convx2_102_337.h5')
!ls -lhtr
import matplotlib.pyplot as plt
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'bo', label='Training Acc')

plt.plot(epochs, val_acc, 'b', label='Validation Acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
# predictions
xs = os.listdir(test_dir+'/NORMAL')



len(xs)
xs = os.listdir(test_dir+'/PNEUMONIA')



len(xs)
# generator
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(test_dir,

                                                 target_size=(150,150),

                                                 batch_size=20,

                                                 class_mode='binary')



samp = test_generator.next()



samp[0].shape, samp[1].shape, samp[1]
preds = model.predict_generator(test_generator, steps=5)
for p in preds[10:]:

    print(f'%2.2f'%p)
# resize images
# copy images into seperate dirs

!mkdir test_ok

!cp ../input/chest_xray/chest_xray/test/NORMAL/*.jpeg test_ok



!mkdir test_bad

!cp ../input/chest_xray/chest_xray/test/PNEUMONIA/*.jpeg test_bad



len(os.listdir('test_ok')), len(os.listdir('test_bad'))
# reassign using new paths

test_ok, test_bad = 'test_ok/', 'test_bad/'
# resize image



for f in os.listdir(test_ok):

    im = Image.open(test_ok+'/'+f)

    im =im.resize((150,150))

    # print('im', im.size)

    im.save(test_ok+'/'+f)
# resize bad images

path = test_bad

for f in os.listdir(path):

    #print('file',path,f)

    im = Image.open(path+'/'+f)

    im =im.resize((150,150))

    #print('im', im.size)

    im.save(path+'/'+f)



len(os.listdir(test_bad))
from keras.preprocessing import image
test_oks = []

path = test_ok



for f in os.listdir(path):

    # print('load file', path+f)

    

    im = image.load_img(path+f, target_size=(150,150))

    # print('im.size', im.size)

    

    ar = image.img_to_array(im)

    # print('ar.shape', ar.shape)

    ar = ar.reshape((1,) + ar.shape)

    # print('ar.shape', ar.shape)

    

    pr = model.predict(ar)

    # print(f,pr)

    test_oks.append((path+f, pr))



    #break



labels = [res[1] for res in test_oks]

thresh = .5



correct = [pr[0] for pr in test_oks if pr[1][0][0]<=thresh]

print('correct', len(correct),'/',len(test_oks), 'acc', len(correct)/len(test_oks))

    



test_oks[:10]
test_bads = []

path = test_bad

for f in os.listdir(path):

    # print('load file', test_ok+f)

    

    im = image.load_img(path+f, target_size=(150,150))

    # print('im.size', im.size)

    

    ar = image.img_to_array(im)

    # print('ar.shape', ar.shape)

    ar = ar.reshape((1,) + ar.shape)

    # print('ar.shape', ar.shape)

    

    pr = model.predict(ar)

    # print(f,pr)

    test_bads.append((path+f, pr))



    # break



test_bads[:5]



labels = [res[1] for res in test_oks]

thresh = .5



correct = [pr[0] for pr in test_bads if pr[1][0][0]>thresh]

print('correct', len(correct),'of',len(test_bads), 'acc', len(correct)/len(test_bads))

    



test_bads[:5]
def pred_img(path, img_f):

    im = image.load_img(path+img_f, target_size=(150,150))

    ar = image.img_to_array(im)

    ar = ar.reshape((1,) + ar.shape)

    pr = model.predict(ar)

    

    return (path+'/'+img_f, pr)
# write image and probability to file



#from IPython.core.display import display, HTML

#display(HTML('<h1>Hello, world!</h1>'))
# base64 images into html file



#import base64



#with open("yourfile.ext", "rb") as image_file:

#    encoded_string = base64.b64encode(image_file.read())
!rm -rf test_ok

!rm -rf test_bad