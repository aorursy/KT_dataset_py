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
import tensorflow as tf

import matplotlib.pyplot as plt
# reading the dataset using pandas



trainset = pd.read_csv('../input/digit-recognizer/train.csv')

testset = pd.read_csv('../input/digit-recognizer/test.csv')

trainset.head()
#splitting the training dataset into label and input



label = trainset.iloc[:,0]

train_input = trainset.iloc[:,1:]

train_input.head()
#splitting the tranin dataset in trainset and validationset(ratio 80% train and 20% test)



from sklearn.model_selection import train_test_split



xtrain, xval, ytrain, yval = train_test_split(train_input, label, test_size = 0.2, random_state = 42)



print('shape of xtrain {}\nshape of xval {}\nshape of ytrain {}\nshape of yval {}'.format(xtrain.shape, 

                                                                                          xval.shape,

                                                                                          ytrain.shape,

                                                                                          yval.shape))
# onehotencoding of label (label are from 0 to 9)

ytrain = tf.keras.utils.to_categorical(ytrain, 10)

yval = tf.keras.utils.to_categorical(yval, 10)



ytrain.shape
# resizeing the image pixel from (0 to255) to (0 to 1) by dividing 255

#because ml model work best with small numbers



xtrain = xtrain/255.

xval = xval/255.

testset = testset/255.
# building the keras model



model = tf.keras.Sequential([

    tf.keras.layers.Dense(512, activation = 'relu', input_shape = (None,784)),

    tf.keras.layers.Dense(256, activation = 'relu'),

    tf.keras.layers.Dense(128, activation = 'relu'),

    tf.keras.layers.Dense(64, activation = 'relu'),

    tf.keras.layers.Dense(32, activation = 'relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(10, activation = 'softmax')

    

])



model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics = ['acc'])
#training the model 



epoch = 20

history = model.fit(xtrain, ytrain, batch_size = 512, epochs = epoch, 

                    validation_data = (xval, yval))
# visualizing the result using matplotlib 



# ploting acc vs epoch graph

acc = history.history['acc']

val_acc = history.history['val_acc']

epoch_list = [i for i in range(epoch)]

plt.plot(epoch_list, acc,'r-', label = 'acc')

plt.plot(epoch_list,val_acc ,'b-', label = 'val_acc')

plt.xlabel('epoch count')

plt.ylabel('accuracy')

plt.title('acc vs epoch')

plt.legend()

plt.show()
# visualizing the result using matplotlib 



# ploting loss vs epoch graph

loss = history.history['loss']

val_loss = history.history['val_loss']

epoch_list = [i for i in range(epoch)]

plt.plot(epoch_list, loss,'r-', label = 'loss')

plt.plot(epoch_list,val_loss ,'b-', label = 'val_loss')

plt.xlabel('epoch count')

plt.ylabel('loss')

plt.title('loss vs epoch')

plt.legend()

plt.show()
#making prediction



prediction = model.predict(testset)
labels = []

label_name = [0,1,2,3,4,5,6,7,8,9]

for i in range(28000):

    label = label_name[np.argmax(prediction[i])]

    labels.append(label)
index = [i for i in range(1,28001)]

df = pd.DataFrame({'ImageId': index, 'Label': labels})



df.head()
df.to_csv('/kaggle/working/answer1.csv', index = False)