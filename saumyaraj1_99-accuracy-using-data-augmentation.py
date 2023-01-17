# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
train.head()
train.shape
predictor = train.drop("label",axis = 1)
target = train.label
predictor = predictor/255
test = test/255
predictor1= predictor.values.reshape(42000,28,28,1)
test1 = test.values.reshape(28000,28,28,1)
sns.countplot(target)
target.value_counts(normalize = True)*100
target1 = to_categorical(target)
plt.imshow(predictor1[5][:,:,0])
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center = False,samplewise_center = False,featurewise_std_normalization = False,samplewise_std_normalization = False,zca_whitening = False,rotation_range = 20,width_shift_range = .15,height_shift_range = .15,horizontal_flip = False,vertical_flip = False,zoom_range=.2)
datagen.fit(predictor1)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))


model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
model.add(Dropout(.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
model.add(Dropout(.25))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(256,activation = "relu"))
model.add(Dropout(.5))
model.add(Dense(10, activation = "softmax"))





model.compile(loss="categorical_crossentropy",metrics = ["accuracy"],optimizer = "adam")
from keras.callbacks import ReduceLROnPlateau
lr = ReduceLROnPlateau(monitor='val_acc', patience=2,  verbose=1,  factor=0.2,  min_lr=0.00001)
model.fit_generator(datagen.flow(predictor1,target1,batch_size=128),epochs = 60,steps_per_epoch = predictor1.shape[0]//128,callbacks = [lr])
y1 = pd.DataFrame(np.argmax(model.predict(test1),axis=1))
a = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y1],axis = 1)
a.columns = ["ImageId","label"]
a.to_csv("cnn_mnist_datagen.csv",index=False)


