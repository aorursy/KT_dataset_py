import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



import os

import cv2

from sklearn import preprocessing

from pathlib import Path
# storing labels for train test

labels_train = []

labels_test = []



# storing path for train test

path_train = []

path_test = []
train_path = "../input/fruit-images-for-object-detection/train_zip/train/"

test_path = "../input/fruit-images-for-object-detection/test_zip/test/"



for filename in os.listdir(train_path):

    if(filename.split('.')[1]=="jpg"):

        labels_train.append(filename.split('_')[0])

        path_train.append(os.path.join(train_path, filename))



for filename in os.listdir(test_path):

    if(filename.split('.')[1]=="jpg"):

        labels_test.append(filename.split('_')[0])

        path_test.append(os.path.join(test_path, filename))





label_train_unique = np.unique(np.array(labels_train))

label_test_unique = np.unique(np.array(labels_test))





print("Unique labels for train are: ", label_train_unique)

print("Number of jpg images in train are: ", len(path_train))



print("\nUnique labels are for test are: ", label_test_unique)

print("Number of jpg images for train are: ", len(path_test))



image = cv2.imread(path_train[0])



rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 



plt.imshow(rgb_img)

plt.title("Label: " + labels_train[0])

plt.axis('off')

plt.show()
image = cv2.imread(path_test[0])



rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 



plt.imshow(rgb_img)

plt.title("Label: " + labels_test[0])

plt.axis('off')

plt.show()
X_train = []



for path in path_train:

    

    img = cv2.imread(path)

    

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 



    final_img =  cv2.resize(rgb_img, (50,50))

    

    X_train.append(final_img)



X_train = np.array(X_train)
X_test = []



for path in path_test:

    

    img = cv2.imread(path)

    

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 



    final_img =  cv2.resize(rgb_img, (50,50))

    

    X_test.append(final_img)



X_test = np.array(X_test)
fruits = {}



for i in range(len(label_train_unique)):

    fruits[label_train_unique[i]] = i

    

fruits    
# storing the values in a temporary list

temp_train = []

temp_test = []



# all the fruits names are being mapped

for label in labels_train:

    temp_train.append(fruits.get(label))



for label in labels_test:

    temp_test.append(fruits.get(label))



print("Length of train data: ", len(temp_train))

print("Length of test data: ", len(temp_test))
y_train = keras.utils.to_categorical(temp_train, 4)

y_test = keras.utils.to_categorical(temp_test, 4)
print("Length of X_train: ", len(X_train))

plt.imshow(X_train[34])

plt.title("Checking X_train"+str(y_train[34]))

plt.show()
print("Length of X_test: ", len(X_test))

plt.imshow(X_test[45])

plt.title("Checking X_test, Label: " + str(y_test[45]))

plt.show()
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
print("X_train shape: ", X_train.shape)

print("X_test shape: ", X_test.shape)



print("\ny_train shape: ", y_train.shape)

print("y_test shape: ", y_test.shape)

model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3), activation="relu"))

model.add(Conv2D(32, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(4, activation="softmax"))



# Compile the model

model.compile(

    loss='categorical_crossentropy',

    optimizer="adam",

    metrics=['accuracy']

)



model.summary()
# training the model

history = model.fit(

    X_train,

    y_train,

    batch_size=32,

    epochs=150,

    validation_split=0.2,

    shuffle=True

)

# displaying the model accuracy

plt.plot(history.history['accuracy'], label='train', color="red")

plt.plot(history.history['val_accuracy'], label='validation', color="blue")

plt.title('Model accuracy')

plt.legend(loc='upper left')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
# displaying the model loss

plt.plot(history.history['loss'], label='train', color="red")

plt.plot(history.history['val_loss'], label='validation', color="blue")

plt.title('Model loss')

plt.legend(loc='upper left')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
model_structure = model.to_json()

f = Path("model_structure.json")

f.write_text(model_structure)



# Save neural network's trained weights

model.save_weights("model_weights.h5")
score, accuracy = model.evaluate(X_test, y_test)

print('Test score achieved:', score)

print('Test accuracy achieved:', accuracy)
pred = model.predict(X_test)
fig, axs= plt.subplots(2,2, figsize=[10,10])

fig.subplots_adjust(hspace=.01)





count=0

for i in range(2):    

    for j in range(2):  

        

        img = cv2.imread(path_test[count])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        img = cv2.resize(img, (250,200))

        result = np.argsort(pred[count])[::-1]

        

        i_max = -1

        max_val = ""

        for (k,val) in enumerate(fruits.keys()):

            

            if(pred[count][k] > i_max):

                i_max = pred[count][k]

                max_val = val

        

        txt = str(max_val) + " with Probability "+ str("{:.4}%".format(i_max*100)) + " %"

            

        

        axs[i][j].imshow(img)

        axs[i][j].set_title(txt)

        axs[i][j].axis('off')



        count+=1

        

plt.suptitle("All predictions are shown in title", fontsize = 18)        

plt.show()