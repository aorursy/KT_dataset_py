# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/testing/test/test"))



# Any results you write to the current directory are saved as output.
from keras.utils import np_utils



from keras.models import Sequential



from keras.layers import Dense, Dropout, Activation, Flatten



from keras.layers import Convolution2D, MaxPooling2D

import numpy as np

import os

import cv2

from skimage import io

import sklearn

from sklearn.model_selection import train_test_split
DatasetPath = []



for i in os.listdir('../input/mahrukhface/mahrukh/mahrukh'):

    DatasetPath.append('../input/mahrukhface/mahrukh/mahrukh' +"/"+ str(i))

imageData = []

imageLabels = []
DatasetPath
for i in DatasetPath:



    imgRead = io.imread(i,as_grey=True)



    imageData.append(imgRead)

    labelRead = int(os.path.split(i)[1].split("_")[0])



    imageLabels.append(labelRead)



imageLabels
X_train, X_test, y_train, y_test = train_test_split(np.array(imageData),np.array(imageLabels), train_size=0.9, random_state = 4)
X_train = np.array(X_train)



X_test = np.array(X_test)



y_train = np.array(y_train) 



y_test = np.array(y_test)

nb_classes = 4



Y_train = np_utils.to_categorical(y_train, nb_classes)



Y_test = np_utils.to_categorical(y_test, nb_classes)
X_train = X_train.reshape(X_train.shape[0], 46, 46, 1)



X_test = X_test.reshape(X_test.shape[0], 46, 46, 1)







# input_shape is for the first layer of model.



# 46, 46, 1 means size 46*46 pixels, 1 channel(because of read as gray,not RGB)



input_shape = (46, 46, 1)







X_train = X_train.astype('float32')



X_test = X_test.astype('float32')







X_train /= 255



X_test /= 255







# then we start the build of model



model = Sequential()







model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))



model.add(Activation('relu'))



model.add(Convolution2D(32, 3, 3))



model.add(Activation('relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.25))







model.add(Convolution2D(64, 3, 3, border_mode='same'))



model.add(Activation('relu'))



model.add(Convolution2D(64, 3, 3))



model.add(Activation('relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.25))







model.add(Flatten())



model.add(Dense(512))



model.add(Activation('relu'))



model.add(Dropout(0.5))



model.add(Dense(nb_classes))



model.add(Activation('softmax'))







# then we compile this model



model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])







# and training



model.fit(X_train, Y_train, batch_size=32, epochs=20,



                 verbose=1, validation_data=(X_test, Y_test))



model=Sequential()
model_json = model.to_json()



with open("model.json", "w") as json_file:



    json_file.write(model_json)



# serialize weights to HDF5



model.save_weights("model.h5")



print("Saved model to disk")



from keras.models import model_from_json

json_file = open('model.json', 'r')



loaded_model_json = json_file.read()



json_file.close()





# In[77]:





loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

print(loaded_model)



print("Loaded model from disk")
DatasetPath = []



for i in os.listdir('../input/testing/test/test'):



    DatasetPath.append(os.path.join('../input/testing/test/test' +"/"+ str(i)))

DatasetPath

imageData = []



imageName = []
face_cascade = cv2.CascadeClassifier('C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')


for i in DatasetPath:



    imgRead = cv2.imread(i,0) # read the photo by gray



    imageName.append(str(i))



    faces = face_cascade.detectMultiScale(



        imgRead,



        scaleFactor=1.1,



        minNeighbors=5,



        minSize=(30, 30),



        flags=cv2.CASCADE_SCALE_IMAGE #OPENCV version 3.x



        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE #OPENCV version 2.x



    )



    for (x, y, w, h) in faces:



        x=x







    cropped = imgRead[y:y + h, x:x + w]



    result = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)  # OPENCV 3.x



    



    transformed_img = cv2.copyMakeBorder(result, 0, 0, 0, 0, cv2.BORDER_REPLICATE)







    for x in range(0, len(result)):



        for y in range(0, len(result[0])):



            center = result[x, y]



            top_left = get_pixel_else_0(result, x - 1, y - 1)



            top_up = get_pixel_else_0(result, x, y - 1)



            top_right = get_pixel_else_0(result, x + 1, y - 1)



            right = get_pixel_else_0(result, x + 1, y)



            left = get_pixel_else_0(result, x - 1, y)



            bottom_left = get_pixel_else_0(result, x - 1, y + 1)



            bottom_right = get_pixel_else_0(result, x + 1, y + 1)



            bottom_down = get_pixel_else_0(result, x, y + 1)







            values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,



                                          bottom_down, bottom_left, left])







            weights = [1, 2, 4, 8, 16, 32, 64, 128]



            res = 0



            for a in range(0, len(values)):



                res += weights[a] * values[a]







            transformed_img.itemset((x, y), res)







    # we only use the part (1,1) to (46,46) of the result img.



    # original img: 0-47, after resize: 1-46



    lbp = transformed_img[1:47, 1:47]  # here 1 included, 47 not included







    imageData.append(lbp)