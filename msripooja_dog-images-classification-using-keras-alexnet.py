import os

import cv2

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.layers.normalization import BatchNormalization



print("Loaded all libraries")
fpath = "../input/images/Images/"

random_seed = 42



categories = os.listdir(fpath)

categories = categories[:20]

print("List of categories = ",categories,"\n\nNo. of categories = ", len(categories))
def load_images_and_labels(categories):

    img_lst=[]

    labels=[]

    for index, category in enumerate(categories):

        for image_name in os.listdir(fpath+"/"+category):

            img = cv2.imread(fpath+"/"+category+"/"+image_name)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            

            img_array = Image.fromarray(img, 'RGB')

            

            #resize image to 227 x 227 because the input image resolution for AlexNet is 227 x 227

            resized_img = img_array.resize((227, 227))

            

            img_lst.append(np.array(resized_img))

            

            labels.append(index)

    return img_lst, labels



images, labels = load_images_and_labels(categories)

print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))

print(type(images),type(labels))
images = np.array(images)

labels = np.array(labels)



print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)

print(type(images),type(labels))
def display_rand_images(images, labels):

    plt.figure(1 , figsize = (19 , 10))

    n = 0 

    for i in range(9):

        n += 1 

        r = np.random.randint(0 , images.shape[0] , 1)

        

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)

        plt.imshow(images[r[0]])

        

        plt.title('Dog breed : {}'.format(labels[r[0]]))

        plt.xticks([])

        plt.yticks([])

        

    plt.show()

    

display_rand_images(images, labels)
#1-step in data shuffling



#get equally spaced numbers in a given range

n = np.arange(images.shape[0])

print("'n' values before shuffling = ",n)



#shuffle all the equally spaced values in list 'n'

np.random.seed(random_seed)

np.random.shuffle(n)

print("\n'n' values after shuffling = ",n)
#2-step in data shuffling



#shuffle images and corresponding labels data in both the lists

images = images[n]

labels = labels[n]



print("Images shape after shuffling = ",images.shape,"\nLabels shape after shuffling = ",labels.shape)
images = images.astype(np.float32)

labels = labels.astype(np.int32)

images = images/255

print("Images shape after normalization = ",images.shape)
display_rand_images(images, labels)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = random_seed)



print("x_train shape = ",x_train.shape)

print("y_train shape = ",y_train.shape)

print("\nx_test shape = ",x_test.shape)

print("y_test shape = ",y_test.shape)
display_rand_images(x_train, y_train)
model=Sequential()



#1 conv layer

model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding="valid",activation="relu",input_shape=(227,227,3)))



#1 max pool layer

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))



model.add(BatchNormalization())



#2 conv layer

model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="valid",activation="relu"))



#2 max pool layer

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))



model.add(BatchNormalization())



#3 conv layer

model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))



#4 conv layer

model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))



#5 conv layer

model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))



#3 max pool layer

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))



model.add(BatchNormalization())





model.add(Flatten())



#1 dense layer

model.add(Dense(4096,input_shape=(227,227,3),activation="relu"))



model.add(Dropout(0.4))



model.add(BatchNormalization())



#2 dense layer

model.add(Dense(4096,activation="relu"))



model.add(Dropout(0.4))



model.add(BatchNormalization())



#3 dense layer

model.add(Dense(1000,activation="relu"))



model.add(Dropout(0.4))



model.add(BatchNormalization())



#output layer

model.add(Dense(20,activation="softmax"))



model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
%%time

model.fit(x_train, y_train, epochs=100)
loss, accuracy = model.evaluate(x_test, y_test)



print(loss,accuracy)
pred = model.predict(x_test)



pred.shape
plt.figure(1 , figsize = (19 , 10))

n = 0 



for i in range(9):

    n += 1 

    r = np.random.randint( 0, x_test.shape[0], 1)

    

    plt.subplot(3, 3, n)

    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

    

    plt.imshow(x_test[r[0]])

    plt.title('Actual = {}, Predicted = {}'.format(y_test[r[0]] , y_test[r[0]]*pred[r[0]][y_test[r[0]]]) )

    plt.xticks([]) , plt.yticks([])



plt.show()