from PIL import Image, ImageDraw

from collections import defaultdict

import os

import numpy as np 

import pickle

import tensorflow

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import optimizers

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

from tensorflow.keras import layers

from tensorflow.keras import Input

from tensorflow.keras.models import Model

import matplotlib.pyplot as plt



def save(path,obj, name ):

    with open(path+ name + '.pkl', 'wb') as f:

        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def load(path ,name ):

    with open(path + name + '.pkl', 'rb') as f:

        return pickle.load(f)
def create_drawing(xcoor,ycoor,draw):

        points= []

        list=[]

        for i in range(len(xcoor)):

            try:

                 list.append((xcoor[i],ycoor[i]))

            except Exception as e:

                print(i)

        draw.line(list,fill="black", width=3) 
#Prepare Matrix of size 27*27 and create PKL files from ndjson

def PrepareData(folderpath):

    files = glob(folderpath)

    filename= "sample"

    k=0;

    for file in files:

        with nlj.open(file) as src:

            print(file)

            imagedata = defaultdict(list)

            for i, line in enumerate(src):

                array = np.asarray(line['drawing'])

                matrix = np.array([[255 for i in range(27)] for i in range(27)])

                matrix = matrix.astype('float64') 

                if array.ndim == 2:

                        im = Image.new('L', (255, 255),color="white")

                        draw = ImageDraw.Draw(im)

                        for row in range(array.shape[0]):

                            create_drawing(array[row,0], array[row,1],draw)

                        resized = im.resize((27,27), Image.ADAPTIVE)

                        matrix = np.array(resized)

                imagedata[line['word']].append(matrix)

                filename = line['word']

        save('..\\batch3\\',imagedata[line['word']],filename)

        

#PrepareData("..\\ndjson\\*.ndjson")

#prepare consolidated data

def ConsolidateData(folderpath):

    train = np.zeros(1, dtype = int) 

    train_labels = np.zeros(1, dtype = int)

    k=0

    with os.scandir(folderpath) as entries:

        for entry in entries:

            name = entry.name.split('.')[0]

            data =  load_obj(folderpath,name)

            data_img = np.array(data)

            length =len(data)

            input_data = data_img.reshape(length, 27*27)

            counttrain = round(length * 0.35)

            countvalidate = round(length * .05)

            counttest = round(length * 0.05)

            if train.shape[0] == 1:

                train = input_data[0:counttrain,:]

                train_labels = np.array([k for i in range(counttrain)])

                validate = input_data[counttrain: counttrain + countvalidate,:]

                validate_labels = np.array([k for i in range(countvalidate)])

                test =  input_data[counttrain + countvalidate: counttest + counttrain + countvalidate ,:]

                counttest = test.shape[0]

                test_labels = np.array([k for i in range(counttest)])

            else:

                train = np.concatenate((train, input_data[0:counttrain,:]))

                train_labels = np.concatenate((train_labels, np.array([k for i in range(counttrain)])))

                validate = np.concatenate((validate,input_data[counttrain: counttrain + countvalidate,:]))

                validate_labels = np.concatenate((validate_labels,np.array([k for i in range(countvalidate)])))

                testdata = input_data[counttrain + countvalidate: counttest + counttrain + countvalidate,:]

                test = np.concatenate((test, testdata))

                test_labels = np.concatenate((test_labels, np.array([k for i in range(testdata.shape[0])])))

            k = k + 1    

            print(name)

folderpath ='../batch3/'

#ConsolidateData(folderpath)            

path = '../batch3/'

# save(path,train,'train_images')

# save(path,train_labels,'train_labels')

# save(path,validate,'validate_images')

# save(path,validate_labels,'validate_labels')

# save(path,test,'test_images')

# save(path,test_labels,'test_labels')
path = '../input/images/'



train_images = load(path,'train_images')

train_labels =  load(path,'train_labels')



validate_images = load(path,'validate_images')

validate_labels = load(path,'validate_labels')



test_images = load(path,'test_images')

test_labels = load(path,'test_labels')
path = '../input/images/'



train_images = load(path,'train_images')

train_labels =  load(path,'train_labels')



validate_images = load(path,'validate_images')

validate_labels = load(path,'validate_labels')



test_images = load(path,'test_images')

test_labels = load(path,'test_labels')



# import random

# indexes = np.array([],dtype = int)

# for x in range(500000):

#     k = random.randint(1,531699)

#     indexes = np.concatenate((indexes,[k]))



# # print(indexes)

# train_images = np.delete(train_images, indexes,axis=0)

# train_labels = np.delete(train_labels, indexes)

# print(train_images.shape)

# print(train_labels.shape)



# print(np.mean(train_images, axis=0))

################################################################

indexes = np.array([],dtype = int)

for i in range(len(train_images)):

  result = all(train_images[i][0] == elem  for elem in train_images[i])

  if result == True:

       indexes = np.concatenate((indexes,[i]))

      #print(result,i)

train_images = np.delete(train_images, indexes,axis=0)

train_labels = np.delete(train_labels, indexes)

#################################################################

indexes = np.array([],dtype = int)

for i in range(len(validate_images)):

  result = all(validate_images[i][0] == elem  for elem in validate_images[i])

  if result == True:

       indexes = np.concatenate((indexes,[i]))

      #print(result,i)

validate_images = np.delete(validate_images, indexes,axis=0)

validate_labels = np.delete(validate_labels, indexes)

##################################################################    

indexes = np.array([],dtype = int)

for i in range(len(test_images)):

  result = all(test_images[i][0] == elem  for elem in test_images[i])

  if result == True:

       indexes = np.concatenate((indexes,[i]))

      #print(result,i)

test_images = np.delete(test_images, indexes,axis=0)

test_labels = np.delete(test_labels, indexes)



print(train_images.shape)

print(train_labels.shape)

print(test_images.shape)

print(test_labels.shape)

train_images = train_images.reshape(len(train_images),27,27,1)

test_images = test_images.reshape(len(test_images),27,27,1)

validate_images = validate_images.reshape(len(validate_images),27,27,1)



#all(item[2] == 0 

# for item in train_images:

#     print(item)

# for i in range(0, 8):

#     if train_images[7]

#     print(train_images[7])

#     display(image.array_to_img(train_images[7]))



# datagen = ImageDataGenerator(

#                             rotation_range=40,

#                             width_shift_range=0.2,

#                             height_shift_range=0.2,

#                             shear_range=0.2,

#                             zoom_range=0.2,

#                             horizontal_flip=True,

#                             fill_mode='nearest')



# count = 0

# for batch_image,batch_label in datagen.flow(train_images,train_labels, batch_size=10000):

#     if count == 50:

#         break;

#     count = count + 1

#     print(batch_image.shape)

#     for i in range(0, 100):

#         print("processing" , i)

#         img = batch_image[i]

#         lab = batch_label[i]

#         data = np.array([img])

#         label = np.array([lab])

#         train_labels = np.concatenate((train_labels,label))

#         train_images = np.concatenate((train_images,data))

#         #display(image.array_to_img(batch_image[i]))

#         #print(lab)





train_images = (train_images - np.mean(train_images, axis=0)) / np.std(train_images, axis=0)

validate_images = (validate_images - np.mean(validate_images, axis=0)) / np.std(validate_images, axis=0)

test_images = (test_images - np.mean(test_images, axis=0)) / np.std(test_images, axis=0)

# # validation_generator = test_datagen.flow(train_images, batch_size=1)



train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)

validate_labels = to_categorical(validate_labels)





tensorflow.keras.backend.clear_session()

input_tensor = Input(shape=(27, 27,1))

branch_a = layers.Conv2D(60, (3, 3), activation='relu',input_shape=(27, 27,1))(input_tensor) #25*25

branch_a = layers.BatchNormalization()(branch_a)

branch_a = layers.MaxPooling2D((2, 2))(branch_a)  #12*12

branch_a = layers.Conv2D(80, (3,3), activation='relu')(branch_a) #10*10

# branch_a = layers.BatchNormalization()(branch_a)

branch_a = layers.MaxPooling2D((4, 4))(branch_a)  #12*12

branch_a = layers.Flatten()(branch_a)

branch_a = layers.Dense(500, activation='relu')(branch_a)



branch_b = layers.Conv2D(20, (3, 27), activation='relu',input_shape=(27, 27,1))(input_tensor)

#branch_b = layers.BatchNormalization()(branch_b)

branch_b = layers.MaxPooling2D((3, 1))(branch_b)

#branch_b = layers.Conv2D(30, (2,2), activation='relu',strides=1)(branch_b)

# branch_b = layers.MaxPooling2D((2, 2))(branch_b)

branch_b = layers.Flatten()(branch_b)

branch_b = layers.Dense(350, activation='relu')(branch_b)



branch_c = layers.Conv2D(40, (27, 3), activation='relu',input_shape=(27, 27,1))(input_tensor) #

#branch_c = layers.BatchNormalization()(branch_c)

branch_c = layers.MaxPooling2D((1, 3))(branch_c)

branch_c = layers.Flatten()(branch_c)

branch_c = layers.Dense(500, activation='relu')(branch_c)



# branch_d = layers.Conv2D(30, (9, 9), activation='relu',input_shape=(27, 27,1))(input_tensor) #19*19

# branch_d = layers.BatchNormalization()(branch_d)

# # branch_d = layers.MaxPooling2D((2, 2))(branch_d)  #12*12

# # branch_d = layers.Conv2D(30, (3,3), activation='relu')(branch_d) #10*10

# # branch_d = layers.BatchNormalization()(branch_d)

# branch_d = layers.Flatten()(branch_d)

# branch_d = layers.Dense(500, activation='relu')(branch_d)





concatenated = layers.concatenate([branch_c,branch_a],axis=-1)

output_tensor = layers.Dense(12, activation='softmax')(concatenated)



model = Model(input_tensor, output_tensor)

model.summary()



#optimizer = optimizers.SGD(lr=0.01, clipnorm=1.)

optimizer= optimizers.Adadelta()#lr=0.8, clipnorm=1.)

#optimizer = optimizers.Adamax()



model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])



#epochs = 1

epochs = 600

batchsize = 35



history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batchsize, shuffle=True,

                     validation_data=(validate_images, validate_labels))

results = model.evaluate(test_images, test_labels)



print(epochs)

print(batchsize)

print(model.metrics_names)

print(results)

print(history.history.keys())
model.save('../working/Model.h5')
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()

#'loss', 'acc', 'val_loss', 'val_acc'



acc = history.history['acc']

val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
from IPython.display import Image

Image("../input/result/acc.jpg")
Image("../input/result/loss.jpg")
from os import walk

for (dirpath, dirnames, filenames) in walk("../"):

    print("Directory path: ", dirpath)

    print("Folder name: ", dirnames)

    print("File name: ", filenames)
#Prediction

# from PIL import Image

# img = Image.open('test image path').convert('LA')

# img = img.resize((27, 27), Image.ANTIALIAS)

# x = np.array(img)

# print(x.shape)

# x = x[:,:,1:1]

# print(x.shape)

# mean normalize image before prediction

# coll= np.array([x])

# from tensorflow.keras.models import load_model

# model = load_model('Model.h5')

# model.predict(coll)
