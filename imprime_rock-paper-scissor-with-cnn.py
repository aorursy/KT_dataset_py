# # cap = cv2.VideoCapture(0) 



# # Check if the webcam is opened correctly 

# # if not cap.isOpened(): 

# #     raise IOError("Cannot open webcam")

    

# k = 0



# while True: 

#     # Read the current frame from webcam

# #     ret, frame = cap.read()

    

# #     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    





#     img_path = "Pics\\"+ "still" +str(k)+".png"

# #     cv2.imwrite(img_path, frame)



#     if not os.path.exists('rock_paper_static.csv'):

#         with open("rock_paper_static.csv", "a+", newline='') as csv_file:

#             writer = csv.DictWriter(csv_file, fieldnames=["Path", "Label"])

#             writer.writeheader()

#             writer = csv.writer(csv_file, delimiter=',')

#             writer.writerow([img_path, "Still"])

#     else:    

#         with open("rock_paper_static.csv", "a+", newline='') as csv_file:

#             writer = csv.writer(csv_file, delimiter=',')

#             writer.writerow([img_path, "Still"])

    

    

    

# #     c = cv2.waitKey(1)

# #     if c == 27: 

# #         break 

        

#     if k == 1000:

#         break 

    

# #     cv2.imshow('Webcam', frame)

    

#     k = k + 1



# cap.release()

# cv2.destroyAllWindows()



# import pandas as pd



# df = pd.read_csv('rock_paper_static.csv')



# df.tail()
# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers import Convolution2D,Dense,MaxPooling2D,Dropout,Flatten

import cv2

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import pandas as pd

import random

import ntpath
import os

from keras.utils.np_utils import to_categorical
dataset = pd.read_csv('/kaggle/input/rock_paper_static.csv')
dataset.tail()
def removePath(path):

    base,tail = ntpath.split(path)

    return tail
dataset['Path'] = dataset['Path'].apply(removePath)
dataset.tail()
datadir = '/kaggle/input/Pics/Pics/'
dataset['Path'][1]
def loadImageSteering(datadir,dataset):

    imagePath = []

    labels = []

    for i in range(len(dataset)):

        img = dataset.iloc[i][0]

        label = dataset.iloc[i][1]

        

        imagePath.append(os.path.join(datadir,img))

        labels.append(label)

        

    imagePath = np.asarray(imagePath)

    labels = np.asarray(labels)

    return imagePath,labels
len(dataset)
imagePath, label = loadImageSteering(datadir, dataset)
imagePath[0]
label[0]
label[-1]
def imagePreprocessing(img):

    img = mpimg.imread(img)

    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

    img = cv2.resize(img,(200,150))

    return img
image = imagePath[1]

image = mpimg.imread(image)

fig,axs = plt.subplots(1,2,figsize=(15,10))

fig.tight_layout()

axs[0].imshow(image)

axs[0].grid(False)

axs[0].set_title("Original Image")

axs[1].imshow(imagePreprocessing(imagePath[1]))

axs[1].grid(False)

axs[1].set_title("Precessed Image")

plt.show()
x = []

for im in imagePath:

    x.append(np.array(imagePreprocessing(im)))
x[0].shape
x = np.array(x)
x.shape
y = np.unique(label, return_inverse=True)[1]
y = to_categorical(y, 4)
y.shape
def nvidiaModel():

    model = Sequential()

    model.add(Convolution2D(24,(5,5),strides=(2,2),input_shape=(150,200,3),activation="elu"))

    model.add(Convolution2D(36,(5,5),strides=(2,2),activation="elu"))

    model.add(Convolution2D(48,(5,5),strides=(2,2),activation="elu")) 

    model.add(Convolution2D(64,(3,3),activation="elu"))   

    model.add(Convolution2D(64,(3,3),activation="elu"))

    model.add(Dropout(0.5))



    model.add(Flatten())



    model.add(Dense(100,activation="elu"))

    model.add(Dropout(0.5))



    model.add(Dense(50,activation="elu"))

    model.add(Dropout(0.5))



    model.add(Dense(10,activation="elu"))

    model.add(Dropout(0.5))



    model.add(Dense(4,activation='softmax'))

    model.compile(optimizer=Adam(lr=.001),loss="mse")



    return model
model = nvidiaModel()
model.summary()
history = model.fit(x,y,epochs=50,batch_size=100,shuffle=10,verbose=1)
label
model.save('rock_paper.h5')
imagePath[3050]
y[3050]
# Still = 3

# Scissor = 2

# Rock = 1

# Paper = 0
import cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
labels = {0:"Paper",1:"Rock",2:"Scissor",3:"Still"}
# cap = cv2.VideoCapture(0) 



# # Check if the webcam is opened correctly 

# if not cap.isOpened(): 

#     raise IOError("Cannot open webcam")

    

# ret, frame = cap.read()



# cap.release()

# cv2.destroyAllWindows()



test_image_path = imagePath[np.random.randint(0,4000)]

test_image = mpimg.imread(test_image_path)

plt.imshow(test_image)
def imagePreprocessing(img):

    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

    img = cv2.resize(img,(200,150))

    return img
plt.imshow(imagePreprocessing(test_image))
x = np.array(imagePreprocessing(test_image))
x = x.reshape(1, 150, 200, 3)
plt.imshow(x[0])
from keras.models import load_model
model = load_model('rock_paper.h5')
if(model.predict(x)[0][3] == 1):

    print("None!")

elif(model.predict(x)[0][1] == 1):

    print("Rock!")

elif(model.predict(x)[0][0] == 1):

    print("Paper!")

elif(model.predict(x)[0][2] == 1):

    print("Scissor!")

else:

    print(model.predict(x))