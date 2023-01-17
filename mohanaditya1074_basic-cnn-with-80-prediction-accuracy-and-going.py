import cv2

import numpy as np

import os

#from random import shuffle

from sklearn.utils import shuffle

from random import randint



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation, Flatten,Dense,BatchNormalization,Dropout



import matplotlib.pyplot as plt
classes=os.listdir('../input/seg_train/seg_train/')



#making a dictionary which maps class label to value.

# 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street

label={}

for instance in classes:

    if instance== 'forest': label[instance]=1

    elif instance== 'buildings': label[instance]=0

    elif instance== 'glacier': label[instance]=2

    elif instance== 'mountain': label[instance]=3

    elif instance== 'sea': label[instance]=4

    elif instance== 'street': label[instance]=5



def get_classlabel(class_code):

    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}

    return labels[class_code]
#assign every image a label according to its class.



img_size=150



def data_making(directory):

    

    data,labels=[],[]

    for folder in os.listdir(directory):

        path=os.path.join(directory,folder)

        

        for img in os.listdir(path):

            

            path_img=os.path.join(path,img)

            

            img_label=label[folder] #previously label_onehot was there. 

            

            img_data=cv2.resize(cv2.imread(path_img),(img_size,img_size)) #reading in bgr format. if need be use ,cv2.IMREAD_GRAYSCALE to read image in gray scale

            

            data.append((img_data))

            labels.append((img_label))

    

    

    

    #shuffle data 

    shuffle(data,labels,random_state=817328462)

    return data,labels
train_Images, train_Labels = data_making('../input/seg_train/seg_train/') #Extract the training images from the folders.



train_Images = np.array(train_Images) #converting the list of images to numpy array.

train_Labels = np.array(train_Labels)

test_Images, test_Labels = data_making('../input/seg_test/seg_test/') #Extract the testing images from the folders.



test_Images = np.array(test_Images) #converting the list of images to numpy array.

test_Labels = np.array(test_Labels)
f,ax = plt.subplots(4,4) 

f.subplots_adjust(0,0,3,3)#(left,bottom,vertical_distance b/w columns,)

for i in range(0,4,1):

    for j in range(0,4,1):

        rnd_number = randint(0,len(train_Images))

        ax[i,j].imshow(train_Images[rnd_number])

        ax[i,j].set_title(get_classlabel(train_Labels[rnd_number]))

        ax[i,j].axis('off')

        

model1=Sequential()



model1.add(Conv2D(256,(5,5),input_shape=train_Images.shape[1:]))

model1.add(Activation("relu"))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Dropout(0.5, noise_shape=None, seed=None))



model1.add(Conv2D(180,(3,3)))

model1.add(Activation("relu"))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Dropout(0.5, noise_shape=None, seed=None))



model1.add(Conv2D(140,(3,3)))

model1.add(Activation("relu"))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Dropout(0.3, noise_shape=None, seed=None))



model1.add(Conv2D(120,(3,3)))

model1.add(Activation("relu"))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Dropout(0.3, noise_shape=None, seed=None))





model1.add(Conv2D(96,(3,3)))

model1.add(Activation("relu"))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Dropout(0.3, noise_shape=None, seed=None))



model1.add(Flatten())



model1.add(Dense(128))

model1.add(Activation("relu"))



model1.add(Dense(128))

model1.add(Activation("relu"))



model1.add(Dense(6))

model1.add(Activation("softmax"))



model1.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#using categorical_crossentropy you have to convert the labels into one hot functions.........

#better option is to use `sparse_categorical_crossentropy`



model1.summary()
history=model1.fit(train_Images,train_Labels,batch_size=64,validation_split=0.1,epochs=17)
train_loss=history.history["loss"]

val_loss=history.history["val_loss"]



epoch_count=range(1,len(val_loss)+1)

plt.plot(epoch_count,train_loss,'r--')

plt.plot(epoch_count,val_loss,'b-')

plt.legend(["train loss","val loss"])

plt.xlabel("epoch")

plt.ylabel("loss")

plt.show()
model1.evaluate(test_Images,test_Labels,verbose=1)
def data_making_for_prediction(directory):

    

    data=[]

    for img in os.listdir(directory):

            

        path_img=os.path.join(directory,img)

        img_data=cv2.resize(cv2.imread(path_img),(img_size,img_size)) #reading in bgr format. if need be use ,cv2.IMREAD_GRAYSCALE to read image in gray scale

            

        data.append((img_data))

    

    

    

    #shuffle data 

    shuffle(data)

    return data
pred_Images=data_making_for_prediction('../input/seg_pred/seg_pred')

pred_Images = np.array(pred_Images) #converting the list of images to numpy array.

f,ax = plt.subplots(2,2) 

f.subplots_adjust(0,0,3,3)#(left,bottom,vertical_distance b/w columns,)

for i in range(0,2,1):

    for j in range(0,2,1):

        rnd_number = randint(0,len(pred_Images))

        ax[i,j].imshow(pred_Images[rnd_number])

        ax[i,j].set_title(get_classlabel(model1.predict_classes(np.array(pred_Images[rnd_number]).reshape(-1,150,150,3))[0]))

        ax[i,j].axis('off')