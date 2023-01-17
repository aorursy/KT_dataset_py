import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import random



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, LSTM

from keras.layers import Conv2D

from keras.utils import plot_model





from numpy.random import seed

seed(48)

from tensorflow import set_random_seed

set_random_seed(438)



from sklearn.model_selection import train_test_split



from IPython.display import Image
nine_route_x = 0*np.arange(40)

nine_route_y = 0.5*np.arange(40)

eight_route_x = np.hstack((0*np.arange(20),0.25*np.arange(20)))

eight_route_y = np.hstack((0.5*np.arange(20),20*0.5+0.3*np.arange(20)))

seven_route_x = np.hstack((0*np.arange(20),-0.25*np.arange(20)))

seven_route_y = np.hstack((0.5*np.arange(20),20*0.5+0.3*np.arange(20)))

six_route_x = np.hstack((0*np.arange(20),0.3*np.arange(20)))

six_route_y = np.hstack((0.5*np.arange(20),20*0.5+0*np.arange(20)))

five_route_x = np.hstack((0*np.arange(20),-0.3*np.arange(20)))

five_route_y = np.hstack((0.5*np.arange(20),20*0.5+0*np.arange(20)))



plt.plot(nine_route_x,nine_route_y)

plt.plot(eight_route_x,eight_route_y)

plt.plot(seven_route_x,seven_route_y)

plt.plot(six_route_x,six_route_y)

plt.plot(five_route_x,five_route_y)

plt.legend(['Go route', 'Post route', 'Corner route', 'In route', 'Out route'])
num_route = 60

route_coordinate = []

route_type = []



for i in range(5):

    for j in range(num_route):

        if i == 0:

            nine_route_x,nine_route_y,direction = 0*np.arange(40)+random.random()*36+8, 0.5*np.arange(40)-2*random.random(),270*np.ones(40)

            route_coordinate.append(np.vstack((nine_route_x,nine_route_y,direction)))

            route_type.append(4)

        if i == 1:

            eight_route_x,eight_route_y,direction = np.hstack((0*np.arange(20),0.25*np.arange(20)))+random.random()*36+8, np.hstack((0.5*np.arange(20),20*0.5+0.3*np.arange(20)))-2*random.random(),np.hstack((270*np.ones(20),220*np.ones(20)))

            

            if(eight_route_x[0] > 25):

                eight_route_x = eight_route_x - 2* (eight_route_x - eight_route_x[0])   

                direction[20:] += 100

            route_coordinate.append(np.vstack((eight_route_x,eight_route_y,direction)))

            route_type.append(3)

        if i == 2:

            seven_route_x,seven_route_y,direction = np.hstack((0*np.arange(20),-0.25*np.arange(20)))+random.random()*36+8, np.hstack((0.5*np.arange(20),20*0.5+0.3*np.arange(20)))-2*random.random(),np.hstack((270*np.ones(20),320*np.ones(20)))

            if(seven_route_x[0] > 25):

                seven_route_x += 2* (seven_route_x[0] - seven_route_x)     

                direction[20:] -= 100

            route_coordinate.append(np.vstack((seven_route_x,seven_route_y,direction)))

            route_type.append(2)

        if i == 3:

            six_route_x,six_route_y,direction = np.hstack((0*np.arange(20),0.3*np.arange(20)))+random.random()*36+8, np.hstack((0.5*np.arange(20),20*0.5+0*np.arange(20)))-2*random.random(),np.hstack((270*np.ones(20),180*np.ones(20)))

            

            if(six_route_x[0] > 25):

                six_route_x = six_route_x - 2* (six_route_x - six_route_x[0])

                direction[20:] -= 180

            route_coordinate.append(np.vstack((six_route_x,six_route_y,direction)))

            route_type.append(1)

        if i == 4:

            five_route_x,five_route_y,direction = np.hstack((0*np.arange(20),-0.3*np.arange(20)))+random.random()*36+8, np.hstack((0.5*np.arange(20),20*0.5+0*np.arange(20)))-2*random.random(),np.hstack((270*np.ones(20),0*np.ones(20)))

            if(five_route_x[0] > 25):

                five_route_x += 2* (five_route_x[0] - five_route_x)

                direction[20:] += 180

            route_coordinate.append(np.vstack((five_route_x,five_route_y,direction)))

            route_type.append(0)        

plt.scatter(np.array(route_coordinate)[:60,0],np.array(route_coordinate)[:60,1])

plt.scatter(np.array(route_coordinate)[60:120,0],np.array(route_coordinate)[60:120,1])

plt.scatter(np.array(route_coordinate)[120:180,0],np.array(route_coordinate)[120:180,1])

plt.scatter(np.array(route_coordinate)[180:240,0],np.array(route_coordinate)[180:240,1])

plt.scatter(np.array(route_coordinate)[240:,0],np.array(route_coordinate)[240:,1])

plt.legend(['Go route', 'Post route', 'Corner route', 'In route', 'Out route'])
x_rescale = np.array(route_coordinate)[:,0]/50

y_rescale = (np.array(route_coordinate)[:,1]- np.array(route_coordinate)[:,1].min())/( np.array(route_coordinate)[:,1].max()- np.array(route_coordinate)[:,1].min())

direction_rescale = np.array(route_coordinate)[:,0]
route_data = np.dstack((x_rescale,y_rescale,direction_rescale)).reshape(-1,1,40,3)
plt.imshow(route_data[283])
X_train, X_test, y_train, y_test = train_test_split(route_data,route_type,test_size=0.2,random_state=1583)
y_train_transform = keras.utils.to_categorical(y_train,5)

y_test_transform = keras.utils.to_categorical(y_test,5)
num_classes = 5

model = Sequential()

model.add(Conv2D(50, kernel_size=(1,5),

                 activation='relu',

                 input_shape=(1,40,3)))

model.add(Conv2D(100, (1, 5), activation='relu'))



model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

Image("../working/model_plot.png")
model.fit(X_train, y_train_transform,

          batch_size=64,

          epochs=500,

          verbose=1,

          validation_data=(X_test, y_test_transform))

score = model.evaluate(X_test, y_test_transform, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
route_data_rnn = np.dstack((x_rescale,y_rescale,direction_rescale)).reshape(-1,40,3)
X_train, X_test, y_train, y_test = train_test_split(route_data_rnn,route_type,test_size=0.2,random_state=42)
y_train_transform = keras.utils.to_categorical(y_train,5)

y_test_transform = keras.utils.to_categorical(y_test,5)
num_classes = 5

model = Sequential()

model.add(LSTM(50, input_shape=(40, 3), return_sequences=True))

model.add(LSTM(100))

model.add(Dropout(0.5))



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
model.fit(X_train, y_train_transform,

          batch_size=128,

          epochs=200,

          verbose=1,

          validation_data=(X_test, y_test_transform))

score = model.evaluate(X_test, y_test_transform, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])