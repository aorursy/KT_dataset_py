import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import zipfile
import cv2
import sklearn.model_selection
from keras import backend as K
from keras.utils import to_categorical

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
          pass
        

def pre_process(name): 
  data = pd.read_csv(name+'.csv')
  x = np.array(data['image'])
  la = np.array(data['category'])
  new = []
  for i in x:
    y = cv2.imread('/kaggle/input/game-of-deep-learning-ship-datasets/train/images/'+i , 1)
    res = cv2.resize(y , (210,126))
    new.append(res)
  new = np.array(new)
  print(data.info())
  print(data.describe())
  return new , la


w , labels = pre_process('/kaggle/input/game-of-deep-learning-ship-datasets/train/train')
# #################### Train_test_split #################### #

x_train , x_test , y_train , y_test = sklearn.model_selection.train_test_split(w , labels , test_size=0.1)
print(x_train.shape , y_train.shape)
print(x_test.shape , y_test.shape)
y_dummy = y_test

# ############## Making one hot coded train label ################ #
y_train = to_categorical(y_train)
print(y_train.shape)

# ############## Making one hot coded test label ################ #
y_test = to_categorical(y_test)
print(y_test.shape)
# ################# Model creation ################ #

x_train = x_train/255.0
x_test = x_test/255.0

if K.image_data_format() == 'channels_first':
    input_shape = (3, 126 , 210) 
else:
    input_shape = (126, 210, 3)

model = keras.Sequential([

############## First Convolutional layer so we will specify input_shape here #####################              
keras.layers.Conv2D(64, (3,3), activation='relu', input_shape = input_shape  ),
keras.layers.MaxPooling2D(2, 2),

############## Second Convolutional layer || not specifying input shape #########################
keras.layers.Conv2D(64, (3,3), activation='relu'),
keras.layers.MaxPooling2D(2, 2),

############## Third Convolutional layer || not specifying input shape #########################
keras.layers.Conv2D(64, (3,3), activation='relu'),
keras.layers.MaxPooling2D(2, 2),

############## Fourth Convolutional layer || not specifying input shape ########################
keras.layers.Conv2D(64, (3,3), activation='relu'),
keras.layers.MaxPooling2D(2, 2),


############### Flattening input ################## 
keras.layers.Flatten(),

############## Neurons droped per E-poch ##########
keras.layers.Dropout(0.5),
# 512 neuron hidden layer
keras.layers.Dense(512, activation='relu'),
keras.layers.Dense(6 , activation='softmax')

])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

# ###### Fitting model ######
model.fit(x_train , y_train , epochs=8 , validation_data=(x_test, y_test))

loss , acc = model.evaluate( x_test , y_test )

print(acc)
# ############### Saving model as hd5 file ################ #

model.save('Ship_Class_2.h5')
# ############ loading the saved model from Keras ################ #

from keras.models import load_model

model = load_model('/content/Ship_Class_2.h5')

print(model.summary())
z , lab = pre_process('/kaggle/input/game-of-deep-learning-ship-datasets/test_ApKoW4T')
predictions = model.predict(z)

# ############# Label names ################### #

my_dic = {
1:'Cargo' ,
2:'Military',
3:'Carrier',
4:'Cruise',
5:'Tankers', 
}

for i in range(len(predictions)):
    print( 'Pre : ' + my_dic[np.argmax(predictions[i])]  )
z , lab = pre_process('sample_submission_ns2btKE')
predictions = model.predict(z)

%matplotlib inline
import matplotlib.pyplot as plt

var = 0
print(my_dic[lab[var]])

plt.imshow(z[var])
plt.xlabel( ' Actuall :  ' + my_dic[lab[var]] )
plt.ylabel( 'Predicted : ' + my_dic[np.argmax(predictions[var])] )

plt.show()