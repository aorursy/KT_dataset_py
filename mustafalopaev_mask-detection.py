import numpy as np

import cv2, os

from keras.utils import np_utils
data_path = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Train'

categories=os.listdir(data_path)

labels=[i for i in range(len(categories))]



label_dict=dict(zip(categories,labels)) #empty dictionary



print(label_dict)

print(categories)

print(labels)
#Get const image size to set

count = 0

# Arithmetic mean of shape data

arht_heigth = 0

arht_width = 0

arht_color = 0



data = (0, 0, 0)

for category in categories:

    folder_path=os.path.join(data_path,category)

    img_names=os.listdir(folder_path)

        

    for img_name in img_names:

        img_path=os.path.join(folder_path,img_name)

        img=cv2.imread(img_path)

        data += img.shape

        data = [x + y for x, y in zip(data, img.shape)]

        count += 1 



values = np.array(data)/count

for value in values:

    print("{:f}".format(value))
img_size= 153

data=[]

target=[]





for category in categories:

    folder_path=os.path.join(data_path,category)

    img_names=os.listdir(folder_path)

        

    for img_name in img_names:

        img_path=os.path.join(folder_path,img_name)

        img=cv2.imread(img_path)



        try:

            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           

            #Coverting the image into gray scale

            resized=cv2.resize(gray,(img_size,img_size))

            #resizing the gray scale into 153x153, since we need a fixed common size for all the images in the dataset

            data.append(resized)

            target.append(label_dict[category])

            #appending the image and the label(categorized) into the list (dataset)



        except Exception as e:

            print('Exception:',e)

            #if any exception rasied, the exception will be printed here. And pass to the next image
data=np.array(data)/255.0

data=np.reshape(data,(data.shape[0],img_size,img_size,1))

target=np.array(target)





new_target=np_utils.to_categorical(target)
np.save('data',data)

np.save('target',new_target)
# Model Training Libraries



from keras.models import Sequential

from keras.layers import Dense,Activation,Flatten,Dropout

from keras.layers import Conv2D,MaxPooling2D

from keras.callbacks import ModelCheckpoint



from sklearn.model_selection import train_test_split
data = np.load('./data.npy')

target = np.load('./target.npy')
model=Sequential()



model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#The first CNN layer followed by Relu and MaxPooling layers



model.add(Conv2D(100,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#The second convolution layer followed by Relu and MaxPooling layers



model.add(Flatten())

model.add(Dropout(0.5))

#Flatten layer to stack the output convolutions from second convolution layer

model.add(Dense(50,activation='relu'))

#Dense layer of 64 neurons

model.add(Dense(2,activation='softmax'))

#The Final layer with two outputs for two categories



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)