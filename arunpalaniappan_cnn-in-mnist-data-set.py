import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')

labels = train['label'].values
images = train.drop(labels = ['label'],axis = 1).values
del train
images = images/255
images = images.reshape(-1,28,28,1)
images.shape
from sklearn.model_selection import train_test_split
train_images,test_images,train_labels, test_labels = train_test_split(images,labels, test_size = 0.2, random_state = 0)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32,kernel_size = (5,5),activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu'))
model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu'))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.compile(optimizer=tf.train.AdamOptimizer(), 
             loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
model.summary()
model.fit(train_images,train_labels,epochs = 20)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
result = model.predict(test_images)
digit = []
for i in range(0,len(result)):
    digit.append(np.argmax(result[i]))
digit = np.array(digit,int)
CM = [ [ 0 for j in range(0,10) ] for i in range(0,10) ]
for i in range(0,len(digit)):
    CM[test_labels[i]][digit[i]] +=1
for i in CM:
    print (i)
test = pd.read_csv('../input/test.csv')

images = test.values
del test

images = images/255
images = images.reshape(-1,28,28,1)
images.shape
result = model.predict(images)
digit = []
for i in range(0,len(result)):
    digit.append(np.argmax(result[i]))
digit = np.array(digit,int)
submissions=pd.DataFrame({"ImageId": list(range(1,len(digit)+1)),
                         "Label": digit})
submissions.to_csv("output.csv", index=False, header=True)