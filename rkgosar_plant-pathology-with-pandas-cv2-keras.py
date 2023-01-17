# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2
train=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
img_size=80

train_image=[]

for name in train['image_id']:

    path='/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    train_image.append(image)
X_Train = np.ndarray(shape=(len(train_image), img_size, img_size, 3),dtype = np.float32)

i=0

for image in train_image:

    X_Train[i]=train_image[i]

    i=i+1

    

X_Train=X_Train/255

print('Train Shape: {}'.format(X_Train.shape))
test_image=[]

for name in test['image_id']:

    path='/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    test_image.append(image)
X_Test = np.ndarray(shape=(len(test_image), img_size, img_size, 3),dtype = np.float32)

i=0

for image in test_image:

    #X_Test[i]=img_to_array(image)

    X_Test[i]=test_image[i]

    i=i+1

    

X_Test=X_Test/255

print('Test Shape: {}'.format(X_Test.shape))
plt.imshow(X_Train[1])
y = train.copy()

del y['image_id']

y.head()
y_train = np.array(y.values)

print(y_train.shape,y_train[0])
import keras 

from keras.models import Sequential

from keras.layers import Conv2D,Dense,BatchNormalization,Dropout,Flatten,MaxPooling2D

from keras.optimizers import Adam
model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='SAME',

                 input_shape=(img_size,img_size,3),

                 activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=64,kernel_size=(5,5),padding='SAME',

                 activation='relu'))

model.add(Conv2D(filters=64,kernel_size=(5,5),padding='SAME',

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='SAME',

                 activation='relu'))

model.add(Flatten())

model.add(Dense(4,activation='softmax'))
optimizer=Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer, metrics=['accuracy'])
hist=model.fit(X_Train,y_train,batch_size=32,epochs=4)
result = model.predict(X_Test)

all_result = np.ndarray(shape = (test.shape[0],4),dtype = np.float32)

for i in range(0,test.shape[0]):

    for j in range(0,4):

        if result[i][j]==max(result[i]):

            all_result[i][j] = 1

        else:

            all_result[i][j] = 0 
healthy = [y_test[0] for y_test in all_result]

multiple_diseases = [y_test[1] for y_test in all_result]

rust = [y_test[2] for y_test in all_result]

scab = [y_test[3] for y_test in all_result]
dict = {'image_id':test.image_id,'healthy':healthy,'multiple_diseases':multiple_diseases,'rust':rust,'scab':scab}
df = pd.DataFrame(dict)

df.tail()
df.to_csv('submission.csv',index = False)