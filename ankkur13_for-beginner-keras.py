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
train=pd.read_csv('../input/train.csv')
train.info()
train.head()
train.describe()
# put labels into y_train variable

y_train = train["Id"]

# Drop 'Id' column

X_train = train.drop(labels = ["Id"], axis = 1)

y_train.head()
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input



def prepareImages(train, shape, path):

    

    x_train = np.zeros((shape, 100, 100, 3))

    count = 0

    

    for fig in train['Image']:

        

        #load images into images of size 100x100x3

        img = image.load_img("../input/"+path+"/"+fig, target_size=(100, 100, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        x_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

        

    return x_train
x_train = prepareImages(train, train.shape[0], "train")
x_train = x_train / 255.0

print("x_train shape: ",x_train.shape)
x_train
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 5005)
y_train.shape
#start building the model



from keras.layers import Activation, BatchNormalization

from keras.layers import MaxPooling2D, Dropout

from keras.models import Sequential

from keras.layers import Flatten, Dense

from keras.layers import Conv2D

from keras.optimizers import Adam



model2 = Sequential()

model2.add(Conv2D(32, (5,5), input_shape = (x_train.shape[1:]), padding = 'same'))

model2.add(Activation('relu'))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(pool_size =  (2,2), strides = (2,2)))

model2.add(Dropout(0.2))



model2.add(Conv2D(32, (3,3), padding = 'same'))

model2.add(Activation('relu'))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(pool_size =  (2,2), strides = (2,2)))

model2.add(Dropout(0.2))



# model1.add(Conv2D(128, (3,3), padding = 'same'))

# model1.add(Activation('relu'))

# model1.add(BatchNormalization())

# model1.add(MaxPooling2D(pool_size =  (2,2)))



model2.add(Flatten())



model2.add(Dense(128))

model2.add(Activation('relu'))

model2.add(BatchNormalization())

model2.add(Dropout(0.5))



model2.add(Dense(y_train.shape[1]))

model2.add(Activation('softmax'))



model2.summary()

#compile the model

optim = Adam(lr = 0.001) #using the already available learning rate scheduler

model2.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
#fit the model on our dataset

history1 = model2.fit(x_train, y_train, epochs = 10, batch_size = 64)
test_data = os.listdir("../input/test/")

print(len(test_data))
test_data = pd.DataFrame(test_data, columns = ['Image'])

test_data['Id'] = ''
x_test = prepareImages(test_data, test_data.shape[0], "test")

x_test = x_test.astype('float32') / 255
predictions = model2.predict(np.array(x_test), verbose = 1)
from sklearn.preprocessing import LabelEncoder



for i, pred in enumerate(predictions):

    test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_data.to_csv('model_submission4.csv', index = False)