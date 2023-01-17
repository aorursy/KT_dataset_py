# importing basic libaries

import os # to get the path information

import cv2 # to deal up with images

import numpy as np # to deal with the maths to convert images to array

import seaborn as sns # to visualize the data

import matplotlib.pyplot as plt # to plot the data\



# importing keras libaries for preprocessing

import keras

from keras.utils import to_categorical # to convert the data to categorical foam(like one hot encoding)

from sklearn.model_selection import train_test_split # to split the dataset into test and train



# importing thee basic layers requried to make our CNN model

from keras.models import Sequential

from keras.layers.core import Dense, Flatten, Activation, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
PATH = '/kaggle/input/harpethockey-action-recognizition-pose-estimation/hockey/' # the working directory for our model it may vary acc to your computer
os.listdir(PATH)
IMG_SIZE = 64 # the size to which we will resize our images

Action = ['Backward', "Forward", "Passing", "Shooting"] # the four Actions in our dataset

Labels = [] #to get the labels(0-> Backward,

#                                  1-> Forward,

#                                  2-> Passing,

#                                  3-> shooting)

Dataset = [] # to append the images of a particular action

for action in Action:

    print("Getting data for Action: ", action)

    for image_from_folder in os.listdir(PATH + action):

        image = cv2.imread(PATH + action + '/' + image_from_folder)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        Dataset.append(image)

        Labels.append(Action.index(action))



print("\nDataset Images size:", len(Dataset))

print("Image Shape:", Dataset[0].shape) # geeting the shape of images after resize

print("Labels size:", len(Labels))
# plotting the bar chat for the no of each classes in our dataset

sns.countplot(x = Labels)
print("Count of Backward images:", Labels.count(Action.index("Backward")))

print("Count of Forward images:", Labels.count(Action.index("Forward")))

print("Count of Passing images:", Labels.count(Action.index("Passing")))

print("Count of Shooting images:", Labels.count(Action.index("Shooting")))
# looking at some of the random images from our dataset and showing them here 

index = np.random.randint(0, len(Dataset) - 1, size= 20)

plt.figure(figsize=(15,10))



for i, ind in enumerate(index, 1):

    img = Dataset[ind]

    lab = Labels[ind]

    lab = Action[lab]

    plt.subplot(4, 5, i)

    plt.title(lab)

    plt.axis('off')

    plt.imshow(img)
# converting the dataset into the array for the processing of the data

Dataset = np.array(Dataset)

Dataset = Dataset.astype("float32") / 255.0 # normalization of the data



#Applying the One hot encode labels

Labels = np.array(Labels)

Labels = to_categorical(Labels)



# Split Dataset to train\test, keeping the 80% train dataset and 20% for test dataset

(trainX, testX, trainY, testY) = train_test_split(Dataset, Labels, test_size=0.2, random_state=42)



print("X Train shape:", trainX.shape)

print("X Test shape:", testX.shape)

print("Y Train shape:", trainY.shape)

print("Y Test shape:", testY.shape)
# training of our CNN model 

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(64,64,3)))

model.add(Conv2D(64 , kernel_size = (5 , 5) , activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(32 , activation = 'relu'))

model.add(Dense(16 , activation = 'softmax'))

model.add(Dense(4, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer= 'adam',metrics=['accuracy'])
# fitting the model over 50 epoches

history = model.fit(trainX, trainY,

          batch_size=16,

          epochs=50,

          verbose=1,

          validation_data=(testX, testY))
#plotting the train and test accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#plotting the train and test losses

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# making predictions on our model

y_pred = model.predict(testX)
# checking the result we have got after training our model

fig = plt.figure(figsize=(20, 40))

for i, idx in enumerate(np.random.choice(testX.shape[0], size=24, replace=False)):

    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(testX[idx]))

    pred_idx = np.argmax(y_pred[idx])

    true_idx = np.argmax(testY[idx])

    ax.set_title("\nActual:-{}\n Predicted:-{}".format(Action[pred_idx], Action[true_idx]),

                 color=("green" if pred_idx == true_idx else "red"))