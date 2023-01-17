import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers import Input

from keras.models import Model

from keras.layers.merge import concatenate



import os

import cv2

from sklearn import preprocessing

from pathlib import Path


train_path = []

label_train = []



path_train = "../input/fingers/train/"



for filename in os.listdir(path_train):

    

    train_path.append(path_train+filename)

    whole_label = filename.split('_')[1]

    useful_label = whole_label.split('.')[0]

    label_train.append(useful_label)



print("Number of train images: ", len(train_path))

print("First 6 labels: ", label_train[:6])
test_path = []

label_test = []



path_train = "../input/fingers/test/"



for filename in os.listdir(path_train):

    

    test_path.append(path_train+filename)

    whole_label = filename.split('_')[1]

    useful_label = whole_label.split('.')[0]

    label_test.append(useful_label)



print("Number of test images: ", len(test_path))

print("First 6 labels: ", label_train[:6])
train_path[0]
# checking train path

image = cv2.imread(train_path[0]) 



# the first image bleongs to clean directory under train

plt.imshow(image)

plt.title(label_train[0], fontsize = 20)

plt.axis('off')

plt.show()
# checking train path

image = cv2.imread(test_path[95]) 



# the first image bleongs to clean directory under train

plt.imshow(image)

plt.title(label_test[95], fontsize = 20)

plt.axis('off')

plt.show()
X_train = []

X_test = []



# reading images for train data

for path in train_path:

    

    image = cv2.imread(path)        

    image =  cv2.resize(image, (50,50))    

    X_train.append(image)

    

# reading images for test data

for path in test_path:

    

    image = cv2.imread(path)        

    image =  cv2.resize(image, (50,50))    

    X_test.append(image)



X_test = np.array(X_test)

X_train = np.array(X_train)
print("Shape of X_train: ", X_train.shape)

print("Shape of X_test: ", X_test.shape)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



X_train /= 255

X_test /= 255
lable_encoder = preprocessing.LabelEncoder()

y_train_temp = lable_encoder.fit_transform(label_train)

y_test_temp = lable_encoder.fit_transform(label_test)



print("Integer encoded values for train: ", y_train_temp)

print("Integer encoded values for test: ", y_test_temp)
y_train = keras.utils.to_categorical(y_train_temp, 12)

y_test = keras.utils.to_categorical(y_test_temp, 12)



print("Categorical values for y_train:", y_train)

print("Categorical values for y_test:", y_test)
X_train_A , X_train_B = X_train[:9000], X_train[-9000:]

y_train_A , y_train_B = y_train[:9000], y_train[-9000:]
print("Shape of X_train_A: ", X_train_A.shape, ", shape of X_train_B: ", X_train_B.shape)
# uncomment to check if they are different or not

# X_train_A == X_train_B
model_seq = Sequential()



# input shape for first layer is 50,50,3 -> 50 * 50 pixles and 3 channels

model_seq.add(Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3), activation="relu"))

model_seq.add(Conv2D(32, (3, 3), activation="relu"))



# maxpooling will take highest value from a filter of 2*2 shape

model_seq.add(MaxPooling2D(pool_size=(2, 2)))



# it will prevent overfitting

model_seq.add(Dropout(0.25))



model_seq.add(Conv2D(64, (3, 3), padding='same', activation="relu"))

model_seq.add(Conv2D(64, (3, 3), activation="relu"))

model_seq.add(MaxPooling2D(pool_size=(2, 2)))

model_seq.add(Dropout(0.25))



model_seq.add(Flatten())

model_seq.add(Dense(512, activation="relu"))

model_seq.add(Dropout(0.5))



# last layer predicts 12 labels

model_seq.add(Dense(12, activation="softmax"))



# Compile the model

model_seq.compile(

    loss='categorical_crossentropy',

    optimizer="adam",

    metrics=['accuracy']

)



model_seq.summary()
keras.utils.plot_model(model_seq, "keras_seq_model.png", show_shapes=True)
# two inputs

input_1 = keras.Input(shape=(50, 50, 3))

input_2 = keras.Input(shape=(50, 50, 3))





# for input 1

conv_1_1 = Conv2D(32, (3, 3), padding='same', activation="relu")(input_1)

conv_1_2 = Conv2D(32, (3, 3), activation="relu")(conv_1_1)

max_1_1 = MaxPooling2D(pool_size=(2, 2))(conv_1_2)

drop_1_1 = Dropout(0.25)(max_1_1)

conv_1_3 = Conv2D(64, (3, 3), padding='same', activation="relu")(drop_1_1)

conv_1_4 = Conv2D(64, (3, 3), activation="relu")(conv_1_3)

max_1_2 = MaxPooling2D(pool_size=(2, 2))(conv_1_4)

drop_1_2 = Dropout(0.25)(max_1_2)

flat_1 = Flatten()(drop_1_2)



# for input 2

conv_2_1 = Conv2D(32, (3, 3), padding='same', activation="relu")(input_2)

conv_2_2 = Conv2D(32, (3, 3), activation="relu")(conv_2_1)

max_2_1 = MaxPooling2D(pool_size=(2, 2))(conv_2_2)

drop_2_1 = Dropout(0.3)(max_2_1)

conv_2_3 = Conv2D(64, (3, 3), padding='same', activation="relu")(drop_2_1)

conv_2_4 = Conv2D(64, (3, 3), activation="relu")(conv_2_3)

max_2_2 = MaxPooling2D(pool_size=(2, 2))(conv_2_4)

drop_2_2 = Dropout(0.3)(max_2_2)

flat_2 = Flatten()(drop_2_2)



# merge both falt layers

merge = concatenate([flat_1, flat_2])



dense = Dense(512, activation="relu")(merge)

drop = Dropout(0.5)(dense)



output = Dense(12, activation="softmax")(drop)



# creating model

model_fun = Model(inputs = [input_1,input_2], outputs = output, name="functional_model")



# compile the model

model_fun.compile(

    loss='categorical_crossentropy',

    optimizer="adam",

    metrics=['accuracy']

)



model_fun.summary()

keras.utils.plot_model(model_fun, "keras_func_model.png", show_shapes=True)
# training the model

history_seq = model_seq.fit(

    X_train,

    y_train,

    batch_size=50,

    epochs=30,

    validation_split=0.2,

    shuffle=True

)
history_fun = model_fun.fit(

    [X_train[:9000], X_train[-9000:]],

    y_train,

    batch_size=50,

    epochs=30,

    validation_split=0.2,

    shuffle=True

)
# for sequential model

# saving the structure of the model

model_structure = model_seq.to_json()

f = Path("model_seq_structure.json")

f.write_text(model_structure)



# saving the neural network's trained weights

model_seq.save_weights("model_seq_weights.h5")







# for functional model

# saving the structure of the model

model_structure = model_fun.to_json()

f = Path("model_fun_structure.json")

f.write_text(model_structure)



# saving the neural network's trained weights

model_fun.save_weights("model_fun_weights.h5")



# displaying the model accuracy



fig, axs = plt.subplots(1, 2 , figsize = [10,5])



plt.suptitle("For Sequential Model", fontsize = 20)



axs[0].plot(history_seq.history['accuracy'], label='train', color="red")

axs[0].plot(history_seq.history['val_accuracy'], label='validation', color="blue")

axs[0].set_title('Model accuracy')

axs[0].legend(loc='upper left')

axs[0].set_ylabel('accuracy')

axs[0].set_xlabel('epoch')



axs[1].plot(history_seq.history['loss'], label='train', color="red")

axs[1].plot(history_seq.history['val_loss'], label='validation', color="blue")

axs[1].set_title('Model loss')

axs[1].legend(loc='upper left')

axs[1].set_xlabel('epoch')

axs[1].set_ylabel('loss')



plt.show()



fig, axs = plt.subplots(1, 2 , figsize = [10,5])



plt.suptitle("For functional api model", fontsize = 20)



axs[0].plot(history_fun.history['accuracy'], label='train', color="red")

axs[0].plot(history_fun.history['val_accuracy'], label='validation', color="blue")

axs[0].set_title('Model accuracy')

axs[0].legend(loc='upper left')

axs[0].set_ylabel('accuracy')

axs[0].set_xlabel('epoch')



axs[1].plot(history_fun.history['loss'], label='train', color="red")

axs[1].plot(history_fun.history['val_loss'], label='validation', color="blue")

axs[1].set_title('Model loss')

axs[1].legend(loc='upper left')

axs[1].set_xlabel('epoch')

axs[1].set_ylabel('loss')



plt.show()

print("For sequential model: ")

score, accuracy = model_seq.evaluate(X_test, y_test)

print('Test score achieved: ', score)

print('Test accuracy achieved: ', accuracy)
print("For functional API model: ")

score, accuracy = model_fun.evaluate([X_test[:9000], X_test[-9000:]], y_test)

print('Test score achieved: ', score)

print('Test accuracy achieved: ', accuracy)
pred = model_seq.predict(X_test)

pred[:10]
y_test[:10]
fig, axs= plt.subplots(2,5, figsize=[24,12])





count=0

for i in range(2):    

    for j in range(5):  

        

        img = cv2.imread(test_path[count])

        

        results = np.argsort(pred[count])[::-1]

      

        labels = lable_encoder.inverse_transform(results)

        

        axs[i][j].imshow(img)

        axs[i][j].set_title(labels[0], fontsize = 20)

        axs[i][j].axis('off')



        count+=1

        

plt.suptitle("Sequential Model : all predictions are shown in title", fontsize = 24)        

plt.show()
pred2 = model_fun.predict([X_test[:9000], X_test[-9000:]])

pred2[20:30]
fig, axs= plt.subplots(2,5, figsize=[24,12])





count=20

for i in range(2):    

    for j in range(5):  

        

        img = cv2.imread(test_path[count])

        

        results = np.argsort(pred2[count])[::-1]

      

        labels = lable_encoder.inverse_transform(results)

        

        axs[i][j].imshow(img)

        axs[i][j].set_title(labels[0], fontsize = 20)

        axs[i][j].axis('off')



        count+=1

        

plt.suptitle("Functional Model : all predictions are shown in title", fontsize = 24)        

plt.show()
!pip install keract
from keract import get_activations, display_heatmaps

keract_inputs = X_test[:1]

keract_targets = y_test[:1]

activations = get_activations(model_seq, keract_inputs)

display_heatmaps(activations, keract_inputs, save=False)