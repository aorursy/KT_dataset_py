import cv2

import matplotlib.pyplot as plt 
# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import os ,shutil 
orginal_data_dir = '/kaggle/input/natural-images/natural_images'

# This is the directory where all our images_ data are stored 

basedir = os.getcwd()+'/natural_images_all'

os.mkdir(basedir)

# This is the basedirectoy where all the test,train,crossvalidation images are stored 
# Sample picture 

import cv2

import matplotlib.pyplot as plt

img=cv2.imread(orginal_data_dir+'/motorbike/motorbike_0006.jpg')

plt.imshow(img)

print(type(img))

#Creating test,train,crossvalidationdirctories to store the images 

train_dir = os.path.join(basedir,'train')

os.mkdir(train_dir)





validation_dir = os.path.join(basedir,'validation')

os.mkdir(validation_dir)



test_dir = os.path.join(basedir,'test')

os.mkdir(test_dir)
train_dir
def makesubfolders(folderPath , name):

    x = os.path.join(folderPath,name)

    os.mkdir(x)

    return x

    
# Creating subfolders for training classes 

train_cats_dir =makesubfolders(train_dir,'cats')



train_dogs_dir=makesubfolders(train_dir,'dogs')



train_fruits_dir=makesubfolders(train_dir,'fruits')



train_persons_dir=makesubfolders(train_dir,'persons')



train_motorbikes_dir=makesubfolders(train_dir,'motorbikes')



train_airplanes_dir=makesubfolders(train_dir,'airplanes')



train_cars_dir=makesubfolders(train_dir,'cars')



train_flowers_dir=makesubfolders(train_dir,'flowers')





# Doing the same for crossvalidation data 

validation_cats_dir =makesubfolders(validation_dir,'cats')



validation_dogs_dir=makesubfolders(validation_dir,'dogs')



validation_fruits_dir=makesubfolders(validation_dir,'fruits')



validation_persons_dir=makesubfolders(validation_dir,'persons')



validation_motorbikes_dir=makesubfolders(validation_dir,'motorbikes')



validation_airplanes_dir=makesubfolders(validation_dir,'airplanes')



validation_cars_dir=makesubfolders(validation_dir,'cars')



validation_flowers_dir=makesubfolders(validation_dir,'flowers')





# Doing the same for test data 

test_cats_dir =makesubfolders(test_dir,'cats')



test_dogs_dir=makesubfolders(test_dir,'dogs')



test_fruits_dir=makesubfolders(test_dir,'fruits')



test_persons_dir=makesubfolders(test_dir,'persons')



test_motorbikes_dir=makesubfolders(test_dir,'motorbikes')



test_airplanes_dir=makesubfolders(test_dir,'airplanes')



test_cars_dir=makesubfolders(test_dir,'cars')



test_flowers_dir=makesubfolders(test_dir,'flowers')





print(len(os.listdir(test_dir)))

print(len(os.listdir(train_dir)))
def Copyfiles(folderName,z,dstName):

    list_imgs = os.listdir(orginal_data_dir+folderName)

   

    train_len = int(.64*len(list_imgs))

    test_len = int(.2*len(list_imgs))+train_len

    validation_len = int(.16*len(list_imgs))+test_len

    

    

    train_fnames = [list_imgs[i] for i in range (train_len)]

    for i in train_fnames: 

        y = z+i

        src = os.path.join(orginal_data_dir , y)

        t = os.path.join(train_dir,dstName)

        dst = os.path.join(t,i)

        shutil.copyfile(src,dst)



    test_fnames = [list_imgs[i] for i in range (train_len,test_len)]

    for i in test_fnames:

        y = z+i

        src = os.path.join(orginal_data_dir , y)

        test = os.path.join(test_dir,dstName)

        dst = os.path.join(test,i)

        shutil.copyfile(src,dst)

        

    validation_fnames=[list_imgs[i] for i in range(test_len,validation_len)]

    for i in validation_fnames: 

        y = z+i

        src = os.path.join(orginal_data_dir , y)

        validation = os.path.join(validation_dir,dstName)

        dst = os.path.join(validation,i)

        shutil.copyfile(src,dst)

    

    

    
# Copying the cats into train,test,validaiton folder 

Copyfiles('/cat','cat/','cats')

Copyfiles('/dog','dog/','dogs')

Copyfiles('/car','car/','cars')

Copyfiles('/airplane','airplane/','airplanes')

Copyfiles('/flower','flower/','flowers')

Copyfiles('/motorbike','motorbike/','motorbikes')

Copyfiles('/fruit','fruit/','fruits')

Copyfiles('/person','person/','persons')

print('successfully copied to destination folders ')

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale = 1/255)

# Rescales the image every pixel to 0 to 1 



train_generator=train_datagen.flow_from_directory(

    train_dir,# the train data directory path where both cats and dogs images present in different directories 

    target_size=(150,150),#resizing every image by 150 X 150 size so all the images will be of the same size and shape 

    batch_size = 20 ,# 20 images will be grouped together and soteres as single object and it will be returned at one iteration 

    class_mode ='categorical'# because we have 8 to classes we need to go with bianaty it labels the data based on the directories 



)#this method loads the data from the directory and iterates over all the files 



validation_datagen = ImageDataGenerator(rescale=1/255)



validation_generator=validation_datagen.flow_from_directory(

    validation_dir,# the train data directory path where both cats and dogs images present in different directories 

    target_size=(150,150),#resizing every image by 150 X 150 size so all the images will be of the same size and shape 

    batch_size = 20 ,# 20 images will be grouped together and soteres as single object and it will be returned at one iteration 

    class_mode ='categorical'# because we have only to classes we need to go with bianaty it labels the data based on the directories 



)

for data_batch,label_batch in validation_generator:

    print("The shape of the 1st data batch is " ,data_batch.shape)

    print("The shape of the 1st label batch is " ,label_batch.shape)

    break
for data_batch,label_batch in train_generator:

    print(label_batch[0])

    plt.imshow(data_batch[0])

   

    break
#Building a simple network 

from keras import layers 

from keras import models 

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape =(150,150,3)))

# here we are defining a conv layer with the basic config after this layer will get a image with the depth of 32 and 148 x 148 because we are not doing pooling 

model.add(layers.MaxPooling2D((2,2)))

# Here again applied a Maxpooling layer of the 2 x 2 here zero parameter will be trained 

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))

# above is the second conv layer 

model.add(layers.MaxPooling2D((2,2)))

# Again Maxpooling layer 

model.add(layers.Conv2D(128,(3,3),activation = 'relu'))



model.add(layers.MaxPooling2D((2,2)))



model.add(layers.Conv2D(128,(3,3),activation = 'relu'))



model.add(layers.MaxPooling2D((2,2)))



model.add(layers.Flatten())



model.add(layers.Dense(512,activation='relu'))



model.add(layers.Dense(8,activation = 'softmax'))
model.summary()
from keras import optimizers 

model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
history = model.fit_generator(train_generator,#The generator we are sending as the data 

                              steps_per_epoch =100 ,# this parameters specificies how many batches it need to iterate utill it gets the entire data are one epoch is completed 

                              epochs = 30,

                              validation_data = validation_generator,# we can also send the validation data so it validates at the same time 

                              validation_steps=50 # number of iteration it need to run before it gets the entire validation data from each batch 

                             )
import matplotlib.pyplot as plt



acc = history.history['accuracy']

# returns the training accuracies while training 

val_acc = history.history['val_accuracy']

# returns the validation accuracy at different levels 

loss = history.history['loss']

# returns the loss at different levels over training data 

val_loss = history.history['val_loss']

#returns the validation loss through put the process 

epochs = range(1,len(acc)+1)



plt.plot(epochs,acc,'bo',label='Training accuracy',color='pink')

plt.plot(epochs,val_acc,'b',label='validation accuracy',color='black')

plt.title('Training and validation accuracy ')

plt.legend()

plt.figure()



plt.plot(epochs,loss,'bo',label='Training loss',color='pink')

plt.plot(epochs,val_loss,'b',label='Validation loss',color='black')

plt.title('Training and validation loss')

plt.legend()



plt.show()

os.chdir(basedir)

os.mkdir(os.getcwd()+'/model')

os.chdir(os.getcwd()+'/model')

model.save('model1')
test_datagen = ImageDataGenerator(rescale=1/255)



test_generator=validation_datagen.flow_from_directory(

    test_dir,# the train data directory path where both cats and dogs images present in different directories 

    target_size=(150,150),#resizing every image by 150 X 150 size so all the images will be of the same size and shape 

    batch_size = 20 ,# 20 images will be grouped together and soteres as single object and it will be returned at one iteration 

    class_mode ='categorical'# because we have only to classes we need to go with bianaty it labels the data based on the directories 



)
model.evaluate_generator(test_generator)
x =0 

for i,j in test_generator:

   

        print('Class predicted by model ',model.predict_classes(i[19].reshape((1,)+i[19].shape)))

        print('The image belongs to ',j[19])

        plt.imshow(i[19])

        break



datagen = ImageDataGenerator(rotation_range = 60,  # how much rotaion of the imgage need to be done between (0-180)

                             width_shift_range = 0.2, # how much width the image can be shifted of the total size of the image 

                             height_shift_range = 0.2, # similar with height 

                             shear_range = .2,

                             zoom_range = .2 ,# how much can an image can be zoomed 

                             fill_mode = 'nearest' # after moving image there may be few empty pixels so we are making them to fill by nearest pixels 

                             

                

                            

                            

                            )
from keras.preprocessing import image 

fname = os.path.join(train_cats_dir,os.listdir(train_cats_dir)[78])

img = image.load_img(fname , target_size=(150,150))

x = image.img_to_array(img) #Converts image into array 

x = x.reshape((1,)+x.shape)

i=0

for batch in datagen.flow(x,batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i = i+1

    if i%4 == 0 :

        break

plt.show()
#Building a simple network 

from keras import layers 

from keras import models 

model1 = models.Sequential()

model1.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape =(150,150,3)))

# here we are defining a conv layer with the basic config after this layer will get a image with the depth of 32 and 148 x 148 because we are not doing pooling 

model1.add(layers.MaxPooling2D((2,2)))

# Here again applied a Maxpooling layer of the 2 x 2 here zero parameter will be trained 

model1.add(layers.Conv2D(64,(3,3),activation = 'relu'))

# above is the second conv layer 

model1.add(layers.MaxPooling2D((2,2)))

# Again Maxpooling layer 

model1.add(layers.Conv2D(128,(3,3),activation = 'relu'))



model1.add(layers.MaxPooling2D((2,2)))



model1.add(layers.Conv2D(128,(3,3),activation = 'relu'))



model1.add(layers.MaxPooling2D((2,2)))



model1.add(layers.Flatten())



model1.add(layers.Dropout(.5)) # Addding extra droup out layer that retains every neuron with the probability of .5



model1.add(layers.Dense(512,activation='relu'))



model1.add(layers.Dense(8,activation = 'softmax'))
model1.summary()
model1.compile(loss = 'categorical_crossentropy',optimizer = optimizers.RMSprop(lr = 1e-4),metrics = ['accuracy'])

train_datagen1 = ImageDataGenerator(rescale=1/250,rotation_range = 40,  # how much rotaion of the imgage need to be done between (0-180)

                             width_shift_range = 0.2, # how much width the image can be shifted of the total size of the image 

                             height_shift_range = 0.2, # similar with height 

                             shear_range = .2,

                             zoom_range = .2 ,# how much can an image can be zoomed 

                             fill_mode = 'nearest', # after moving image there may be few empty pixels so we are making them to fill by nearest pixels 

                            horizontal_flip = True

                                  )

validation_datagen1 = ImageDataGenerator(rescale = 1/255)# we don't need to do dataaugmentation on the validation and the training data 



test_datagen1 = ImageDataGenerator(rescale = 1/255)



train_generator1 = train_datagen1.flow_from_directory(

    train_dir,# the train data directory path where both cats and dogs images present in different directories 

    target_size=(150,150),#resizing every image by 150 X 150 size so all the images will be of the same size and shape 

    batch_size = 20 ,# 20 images will be grouped together and soteres as single object and it will be returned at one iteration 

    class_mode ='categorical'# because we have 8 to classes we need to go with bianaty it labels the data based on the directories 



)#this method loads the data from the directory and iterates over all the files 



validation_generator1 = validation_datagen1.flow_from_directory(

    validation_dir,# the train data directory path where both cats and dogs images present in different directories 

    target_size=(150,150),#resizing every image by 150 X 150 size so all the images will be of the same size and shape 

    batch_size = 20 ,# 20 images will be grouped together and soteres as single object and it will be returned at one iteration 

    class_mode ='categorical'# because we have 8 to classes we need to go with bianaty it labels the data based on the directories 



)#this method loads the data from the directory and iterates over all the files 



test_generator1 = train_datagen1.flow_from_directory(

    test_dir,# the train data directory path where both cats and dogs images present in different directories 

    target_size=(150,150),#resizing every image by 150 X 150 size so all the images will be of the same size and shape 

    batch_size = 20 ,# 20 images will be grouped together and soteres as single object and it will be returned at one iteration 

    class_mode ='categorical'# because we have 8 to classes we need to go with bianaty it labels the data based on the directories 



)#this method loads the data from the directory and iterates over all the files 





history1 = model1.fit_generator(train_generator1,steps_per_epoch =100 , epochs = 50 ,validation_data = validation_generator1,validation_steps = 50)
model1.evaluate_generator(test_generator1)