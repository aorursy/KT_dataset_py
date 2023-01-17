# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
train.head(4)
x = train['image_id'][0]

f = '/kaggle/input/plant-pathology-2020-fgvc7/images/'+x+'.jpg'

f
from PIL import Image

import glob

train_img = []



for file in train['image_id']:

    img = Image.open('/kaggle/input/plant-pathology-2020-fgvc7/images/'+file+'.jpg')

    img = img.resize((32,32))

    train_img.append(img)

    
import matplotlib.pyplot as plt

plt.imshow(train_img[0],cmap = 'gray')
test.head(2)
test_img = []



for file in test['image_id']:

    img = Image.open('/kaggle/input/plant-pathology-2020-fgvc7/images/'+file+'.jpg')

    img = img.resize((32,32))

    test_img.append(img)

    
print(len(train_img),len(test_img))
from keras.preprocessing.image import img_to_array
train_x = np.ndarray(shape = (1821,32,32,3),dtype = np.float32)

i = 0

for img in train_img:

    train_x[i] = img_to_array(img)

    i+=1

print(i) 
test_x = np.ndarray(shape = (1821,32,32,3),dtype = np.float32)

i = 0

for img in test_img:

    test_x[i] = img_to_array(img)

    i+=1

print(i) 
df = train.copy()

del df['image_id']

df.head(2)

train_y = np.array(df.values)

print(train_y.shape,train_y[0])
import keras

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32,kernel_size = (2,2),input_shape = (32,32,3),activation = 'relu'))

model.add(keras.layers.Conv2D(32,kernel_size = (2,2),activation = 'relu'))

model.add(keras.layers.Conv2D(32,kernel_size = (2,2),activation = 'relu'))

model.add(keras.layers.Conv2D(32,kernel_size = (2,2),activation = 'relu'))

model.add(keras.layers.Conv2D(32,kernel_size = (2,2),activation = 'relu'))



model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))



model.add(keras.layers.Conv2D(32,kernel_size = (2,2),activation = 'relu'))

model.add(keras.layers.Conv2D(32,kernel_size =(2,2) ,activation = 'relu'))

model.add(keras.layers.Conv2D(32,kernel_size = (2,2) ,activation = 'relu'))

model.add(keras.layers.Conv2D(32,kernel_size =(2,2) ,activation = 'relu'))

model.add(keras.layers.Conv2D(32,kernel_size = (2,2),activation = 'relu'))



model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))



model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(32,activation = 'relu'))

model.add(keras.layers.Dropout(0.01))

model.add(keras.layers.Dense(4,activation = 'softmax'))
from tensorflow.keras import optimizers



model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy'])
# Train the model

history = model.fit(train_x,train_y,epochs=80)
yp = model.predict(test_x)
yp[0]
c = np.ndarray(shape = (1821,4),dtype = np.float32)

for i in range(1821):

    for j in range(4):

        if yp[i][j]==max(yp[i]):

            c[i][j] = 1

        else:

            c[i][j] = 0 
healthy = [y[0] for y in c]

multiple_diseases = [y[1] for y in c]

rust = [y[2] for y in c]

scab = [y[3] for y in c]
print(len(rust),len(scab))
df = {'image_id':test.image_id,'healthy':healthy,'multiple_diseases':multiple_diseases,'rust':rust,'scab':scab}
data = pd.DataFrame(df)

data.head(5)
data.to_csv('submission.csv',index = False)