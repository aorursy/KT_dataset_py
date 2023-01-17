from __future__ import absolute_import, division, print_function

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

import os
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import InputLayer, Input

from tensorflow.python.keras.layers import Reshape,MaxPooling2D

from tensorflow.python.keras.layers import Conv2D,Dense,Flatten

infected = os.listdir('../input/cell_images/cell_images/Parasitized/') 

uninfected = os.listdir('../input/cell_images/cell_images/Uninfected/')
infected
data = []

labels = []



for i in infected:

    try:

    

        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((50 , 50))

        data.append(np.array(resize_img))

        labels.append(1)

        

    except AttributeError:

        print('')

    

for u in uninfected:

    try:

        

        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+u)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((50 , 50))

        data.append(np.array(resize_img))

        labels.append(0)

        

    except AttributeError:

        print('')

cells = np.array(data)

labels = np.array(labels)



np.save('Cells' , cells)

np.save('Labels' , labels)
print('Cells : {} | labels : {}'.format(cells.shape , labels.shape))
plt.figure(1 , figsize = (15 , 9))

n = 0 

for i in range(49):

    n += 1 

    r = np.random.randint(0 , cells.shape[0] , 1)

    plt.subplot(7 , 7 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(cells[r[0]])

    plt.title('{} : {}'.format('Infected' if labels[r[0]] == 1 else 'Unifected' ,

                               labels[r[0]]) )

    plt.xticks([]) , plt.yticks([])

    

n = np.arange(cells.shape[0])

np.random.shuffle(n)

cells = cells[n]

labels = labels[n]
cells = cells.astype(np.float32)

labels = labels.astype(np.int32)

cells = cells/255
img_size= 50

img_size_flat = img_size*img_size

img_shape = (img_size,img_size)

img_shape_full =(img_size,img_size,3)

num_channels = 3
from sklearn.model_selection import train_test_split



train_x , x , train_y , y = train_test_split(cells , labels , 

                                            test_size = 0.2 ,

                                            random_state = 111)



eval_x , test_x , eval_y , test_y = train_test_split(x , y , 

                                                    test_size = 0.5 , 

                                                    random_state = 111)
print('train data shape {} ,eval data shape {} , test data shape {}'.format(train_x.shape,

                                                                           eval_x.shape ,

                                                                           test_x.shape))
train_x.shape
model = Sequential()

model.add(InputLayer(input_shape=(50,50,3)))

model.add(Reshape(img_shape_full))

model.add(Conv2D(kernel_size=7,strides=4,filters=64,padding='same',activation='relu',name='Con_layer1'))

model.add(MaxPooling2D(pool_size=5,strides=1))

model.add(Conv2D(kernel_size=9,strides=4,filters=128,padding='same',activation='relu',name='Con_layer2'))

model.add(MaxPooling2D(pool_size=2,strides=1))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(2,activation='softmax'))
from tensorflow.python.keras.optimizers import Adam,Adagrad,Adadelta,Adamax



batch_size = [10, 20, 40, 60, 80, 100,128,256,512]

epochs = [10, 50, 100]



param_grid = dict(batch_size=batch_size, epochs=epochs)
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1)
model.compile(optimizer=Adam(lr=1e-3),loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(x=train_x,y=train_y,epochs=16,batch_size=64)
result = model.evaluate(x=test_x,

                        y=test_y)
y_pred=model.predict(x=test_x)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score

print (test_y.shape)
result
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))
print(format(confusion_matrix))
test_y
cls_pred=np.argmax(y_pred,axis=1)
confusion_matrix(test_y,cls_pred)
print('{} \n{} \n{}'.format(confusion_matrix(test_y , cls_pred) , 

                           classification_report(test_y , cls_pred) , 

                           accuracy_score(test_y , cls_pred)))

