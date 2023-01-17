import os

import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras import layers, models



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")
x = train.drop('label', axis=1)

y = train['label']
y = to_categorical( y,num_classes=len(y.unique()))
x = x/255
x = x.values.reshape(-1,28,28,1)
plt.imshow(x[1][:,:,0])

plt.title(train['label'][1])
model = Sequential()

model.add(Conv2D(filters = 32 ,kernel_size =(5,5),activation = 'relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(filters = 64 ,kernel_size =(3,3),activation = 'relu'))

model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Dropout(0.3))





model.add(Conv2D(filters = 128 ,kernel_size =(3,3),activation = 'relu'))

model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Dropout(0.3))

          

model.add(Flatten())

          

model.add(Dense(64,activation= 'relu'))

model.add(Dropout(0.5))



model.add(Dense(10, activation = "softmax"))

model.summary()              

          
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x,y,epochs=10, batch_size=86,validation_split = 0.2)
# test.drop('id',axis=1,inplace=True)

x_test = test/255

x_test = x_test.values.reshape(-1,28,28,1)

predicted = model.predict(x_test)
pos = 99

plt.imshow(x_test[pos,:,:,0])

print("Predicted Value " +str(np.argmax(predicted[pos])))
results = np.argmax(predicted,axis = 1)

results = pd.Series(results,name="label")
final=pd.DataFrame({"id": list(range(1,len(results)+1)),"label": results})

final.to_csv("submission.csv", index=False, header=True)