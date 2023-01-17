# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

import cv2

print(os.listdir("../input/nnfl-lab-1")) 
filenames = os.listdir("/kaggle/input/nnfl-lab-1/training/training")

categories = []

for filename in filenames:

    category = filename.split('_')[0]

    if category == 'chair':

        categories.append(0)

    elif category == 'kitchen':

        categories.append(1)

    elif category == 'knife':

        categories.append(2)

    elif category == 'saucepan':

        categories.append(3)





df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
train_images = []       

train_labels = []

shape = (128,128)  

train_path = '/kaggle/input/nnfl-lab-1/training/training/'



for filename in os.listdir('/kaggle/input/nnfl-lab-1/training/training/'):

    if filename.split('.')[1] == 'jpg':

        img = cv2.imread(os.path.join(train_path,filename))

        

        name=filename.split('_')[0]

        if name == 'chair':

            train_labels.append(0)

        elif name == 'kitchen':

            train_labels.append(1)

        elif name == 'knife':

            train_labels.append(2)

        elif name == 'saucepan':

            train_labels.append(3)

        

        img = cv2.resize(img,shape)

        

        train_images.append(img)





train_labels = pd.get_dummies(train_labels).values

train_images = np.array(train_images)

x_train,x_val,y_train,y_val = train_test_split(train_images,train_labels,random_state=42)
test_images = []

test_labels = []

shape = (128,128)

test_path = '/kaggle/input/nnfl-lab-1/testing/testing'



for filename in os.listdir('/kaggle/input/nnfl-lab-1/testing/testing'):

    if filename.split('.')[1] == 'jpg':

        img = cv2.imread(os.path.join(test_path,filename))

        img = cv2.resize(img,shape)

        test_labels.append(filename.split('_')[0])

        

        test_images.append(img)

        



test_images = np.array(test_images)


plt.imshow(test_images[0])
FAST_RUN = True

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax')) 



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



model.summary()
history = model.fit(x_train,y_train,epochs=70,batch_size=20,validation_data=(x_val,y_val))
model.save_weights("model.h5")
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

evaluate = model.evaluate(x_val,y_val)

print(evaluate)
prediction = model.predict(test_images)
mapping = []

for i in range(len(prediction)):

    entry = []

    entry.append(test_labels[i])

    entry.append(np.argmax(prediction[i]))

    mapping.append(entry)

output = pd.DataFrame(mapping)

output = output.rename(columns = {0: "id", 1: "label"})
output.to_csv('CSV.csv', index=False)
from IPython.display import HTML 

import pandas as pd 

import numpy as np  

import base64 

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False) 

    b64 =  base64.b64encode(csv.encode()) 

    payload = b64.decode()  

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"  target="_blank">{title}</a>'  

    html = html.format(payload=payload,title=title,filename=filename)  

return HTML(html)  

create_download_link(output)  