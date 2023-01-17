# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import image

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")

df1.head()
# path of the dogs images

img_file='/kaggle/input/dog-breed-identification/train/'



df=df1.assign(img_path=lambda x: img_file + x['id'] +'.jpg')

print(df.shape)
from keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator



X=np.array([img_to_array(load_img(img, target_size=(96, 96))) for img in df['img_path'].values.tolist()])

X.shape
# X = X.reshape(-1,96,96,1)
X.shape
Y = pd.get_dummies(df['breed'])

Y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)



print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense,Activation,Conv2D,Flatten,MaxPool2D,Dropout
model = Sequential()



model.add(Conv2D(64,(3,3),input_shape=(96,96,3)))

model.add(Activation('relu'))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(32,(3,3)))

model.add(Activation('relu'))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(16,(3,3)))

model.add(Activation('relu'))



# model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(8,(3,3)))

model.add(Activation('relu'))



model.add(Flatten())



model.add(Dropout(0.25))



model.add(Dense(100))

model.add(Activation('relu'))



model.add(Dense(100))

model.add(Activation('relu'))



model.add(Dense(100))

model.add(Activation('relu'))



model.add(Dropout(0.25))



model.add(Dense(Y.shape[1]))

model.add(Activation('softmax'))



model.summary()
model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['accuracy'])
model.fit(X_train,Y_train,

         validation_data=(X_test,Y_test),

         batch_size=32,

         epochs=75,

         verbose=2)
from glob import glob

test_files = glob('../input/dog-breed-identification/test/*.jpg')

type_files = np.asarray(test_files)

type(test_files)
data = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")

t = data['id']

test = pd.DataFrame(data=t,columns=['id'])

test.head()
test['img_path'] = test_files
s = int(224)

type(s)
test_img = np.array([img_to_array(load_img(img,target_size=(96,96))) for img in test['img_path'].values.tolist()])

test_img.shape
preds = model.predict(test_img)

preds.shape
id = [test['img_path'][i][-36:-4] for i in range(len(test))]

id = np.asarray(id)
labels = list(Y.keys())

predictions = pd.DataFrame(data=preds,

                 columns=labels)
predictions['id'] = id
predictions
cols = predictions.columns.tolist()

cols = cols[-1:] + cols[:-1]

predictions = predictions[cols]

predictions.head(5)
predictions.to_csv('../working/submission.csv', index=False)