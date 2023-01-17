import os
print(os.listdir("../input/carsbikes/"))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input/"))



from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras import backend as K

from collections import namedtuple
# import numpy as np
import cv2


img_rows=400
img_cols=400
num_images=1986
x=np.load('../input/carsbikes/withcarsbikes400400_innpy.npy')
# print(x.shape)
x=x.reshape(num_images, img_rows, img_cols, 1)
x=x/255
y=np.load('../input/carsbikes/withcarsbikes400400_outnpy.npy')



# def loss_func(y_true,y_pred):
#     mask = np.array([False, False, False,False,True])
#     mask1 = np.array([True, True, True,True,False])
#     maskedy = tf.boolean_mask(y_true,mask)
#     #tf_print(maskedy)
#     temp = K.mean(tf.subtract(tf.boolean_mask(y_true,mask),tf.boolean_mask(y_pred,mask)))
# #     temp = K.tf.reshape(temp,[])
#     print("loss = "+str(temp))
#     if temp<500:
#         return K.mean(K.square(tf.boolean_mask(y_pred,mask1) - tf.boolean_mask(y_true,mask1)), axis=-1)
#     else:
#         return temp
# def tf_print(op, tensors, message=None):
#     def print_message(x):
#         sys.stdout.write(message + " %s\n" % x)
#         return x

#     prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
#     with tf.control_dependencies(prints):
#         op = tf.identity(op)
#     return op


ctr1=0

for ii in y:
    if y[ctr1][4] == 1000 or y[ctr1][4] == 1:
        y[ctr1][4] = 1.0
#         print('a')
    if y[ctr1][5] == 1000 or y[ctr1][4] == 1:
        y[ctr1][5] = 1.0
#         print('b')
    if y[ctr1][6] == 1000 or y[ctr1][4] == 1:
        y[ctr1][6] = 1.0
    if y[ctr1][5]==1 and y[ctr1][6]==1:
        y[ctr1][5]=0
        y[ctr1][6]=0
#         print('c')
    ctr1+=1
# ctr1=0
# for ii in y:
#     print(ii)
#     ctr1+=1
# penalty = 10000

# def loss_func(y_true,y_pred):
    
#     mean_square = K.mean(K.square( tf.matmul( tf.reshape(y_true[:,4],[tf.shape(y_true[:,4])[0],1]) ,(y_true[:,0:4]- y_pred[:,0:4]) ,transpose_a=True) ), axis=-1)
    
#     return mean_square 
# print(y[:,4])

main_input = Input(shape=((img_rows, img_cols, 1)),  name='main_input')

ship_model = Conv2D(16,kernel_size=(3,3),activation='relu')(main_input)
ship_model = MaxPooling2D(pool_size = (2,2))(ship_model)
ship_model = Conv2D(32,kernel_size=(3,3),activation='relu')(ship_model)
ship_model = MaxPooling2D(pool_size = (3,3))(ship_model)
ship_model = Conv2D(64,kernel_size=(2,2),activation='relu')(ship_model)
ship_model = MaxPooling2D(pool_size = (3,3))(ship_model)
ship_model = Conv2D(128,kernel_size=(2,2),activation='relu')(ship_model)
ship_model = MaxPooling2D(pool_size = (3,3))(ship_model)
ship_model = Conv2D(256,kernel_size=(2,2),activation='relu')(ship_model)
# ship_model = MaxPooling2D(pool_size = (3,3))(ship_model)
ship_model = Conv2D(512,kernel_size=(2,2),activation='relu')(ship_model)
# # ship_model = MaxPooling2D(pool_size = (3,3))(ship_model)
# ship_model = Conv2D(1024,kernel_size=(2,2),activation='relu')(ship_model)
# ship_model = MaxPooling2D(pool_size = (3,3))(ship_model)
ship_model = Flatten()(ship_model)
ship_model = Dense(64,activation='relu')(ship_model)

further = Dense(32,activation='relu')(ship_model)
further = Dense(16,activation='relu')(further)

further_reg = Dense(16,activation='relu')(ship_model)
out_class = Dense(3,activation='softmax')(further)
out_reg = Dense(4,activation='relu')(ship_model)
from keras.optimizers import Adam #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = Model(inputs=[main_input], outputs=[out_class, out_reg])
model.compile(optimizer=Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss=['categorical_crossentropy','mean_squared_error'])
# print(model.summary())
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
# ship_model = Sequential()
# ship_model.add(Conv2D(16,kernel_size=(4,4),activation='relu',input_shape=((img_rows, img_cols, 1))))
# # ship_model.add(MaxPooling2D(pool_size = (4,4)))
# ship_model.add(Conv2D(32,kernel_size=(2,2),activation='relu'))
# ship_model.add(MaxPooling2D(pool_size = (3,3)))
# ship_model.add(Dropout(0.2))
# ship_model.add(Conv2D(64,kernel_size=(2,2),activation='relu'))
# ship_model.add(MaxPooling2D(pool_size = (2,2)))
# # ship_model.add(Conv2D(64,kernel_size=(2,2),activation='relu'))
# # ship_model.add(Dropout(0.2))
# # ship_model.add(MaxPooling2D(pool_size = (2,2)))


# ship_model.add(Flatten())
# # ship_model.add(Dense(100,activation='relu'))
# ship_model.add(Dense(50,activation='relu'))
# # ship_model.add(Dense(30,activation='relu'))
# # ship_model.add(Dense(50,activation='relu'))
# # ship_model.add(Dense(20,activation='relu'))
# # ship_model.add(Dense(50,activation='relu'))
# ship_model.add(Dense(5,activation='relu'))

# import keras
# # print("-------------------\n\n\n\n\n\n\n\n\n------------------\n\n")
# ship_model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['accuracy'])


# history = ship_model.fit(x, y,
#           batch_size=32,
#           epochs=10,
# #           callbacks=[plot_losses],
#           validation_split = 0.2)
history  = model.fit(x, [y[:,4:], y[:,0:4]],  epochs=60, batch_size=32)
# ship_model.save('24augtrainedmodel1.h5')
print(os.listdir("../input/testimages"))
# import matplotlib.pyplot as plt

# history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
import os
for filename in os.listdir('../input/testimages'):
    if filename.endswith(".JPEG"):
        im = cv2.imread('../input/testimages/'+filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im=cv2.resize(im,(400,400))
        print(im.shape)
#         im=im[:,:,0]
        im1=im.reshape(1, 400, 400, 1)
        im1=im1/255
        vals=model.predict(im1)
        print(str(filename))
        print(vals)
#         cv2.rectangle(im_a,(int(vals[0]),int(vals[1])),(int(vals[2]),int(vals[3])),(0,255,0))
#         cv2.imshow('img',im)
        
model.save('8novtrainedmodel4.h5')
# newm = keras.models.load_model('7novtrainedmodel.h5',custom_objects={'loss_func': loss_func})

