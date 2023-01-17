import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

from keras.preprocessing import image

from keras.utils import to_categorical
#load training data 

train = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

# get only using defective images

train = train[ train['EncodedPixels'].notnull() ]

#get the class id

train['ClassId'] = train['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
train.head(5)
def markArea(rle, imgshape):

    W = imgshape[0]

    H = imgshape[1]

    

    mark= np.zeros( W*H ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    #print(array)

    begain = array[0::2]

    lengths = array[1::2]

    

    #print(begain)



    current_position = 0

    for index, first in enumerate(begain):

        mark[int(first):int(first+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mark.reshape(H,W), k=1 ) )



fig=plt.figure(figsize=(20,100))

columns = 4

rows = 25

for i in range(1, 10+1):

    fig.add_subplot(rows, columns, i)

    

    imgname = train['ImageId_ClassId'].iloc[i].split('_')[0]

    img = cv2.imread( '../input/severstal-steel-defect-detection/train_images/'+imgname )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #print(train['EncodedPixels'].iloc[i])

    #print(img.shape)

    mark = markArea( train['EncodedPixels'].iloc[i], img.shape  )

    img[mark==1,0] = 255

    

    plt.imshow(img)

plt.show()
train_image = []

train_label = []



fig=plt.figure(figsize=(20,100))

columns = 4

rows = 25



for i in range(1, 10+1):

    fig.add_subplot(rows, columns, i)

    

    imgname = train['ImageId_ClassId'].iloc[i].split('_')[0]

    imgclass = train['ImageId_ClassId'].iloc[i].split('_')[1]

    img = cv2.imread( '../input/severstal-steel-defect-detection/train_images/'+imgname)



    mark = markArea( train['EncodedPixels'].iloc[i], img.shape  )

    img[mark==1,0] = 255

    #plt.imshow(mark)

    

    img=image.img_to_array(img)

    img=img/255

    

    train_image.append(img)

    train_label.append(imgclass)

    plt.imshow(img)

    

X = np.array(train_image)

y = np.array(train_label)

y = to_categorical(y)



print('done')
#print(X)

print(X.shape)

print(y.shape)

print('f')
#sequential modle is used to train

model = Sequential()

model.add(Conv2D(8, kernel_size=(3, 3),activation='relu',input_shape=(256,1600,3)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])



# Creating validation set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)



# Training the model

#model.fit(X_train, y_train)



#model.fit(X, y ,epochs=2)



model.fit(X_train, y_train, epochs=10, verbose=1, validation_split=0.25)



print('Done')