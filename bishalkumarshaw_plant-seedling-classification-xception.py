import cv2

import pandas as pd

from glob import glob

import numpy as np

from matplotlib import pyplot as plt

import gc



dim = 160  # px to scale



path = '../input/plant-seedlings-classification/train/*/*.png' 

files = glob(path)



trainImg = []

trainLabel = []



j = 1

num = len(files)



# Obtain images and resizing, obtain labels

for img in files:

    print(str(j) + "/" + str(num), end="\r")

    trainImg.append(cv2.resize(cv2.imread(img), (dim, dim)))  # Get image (with resizing)

    trainLabel.append(img.split('/')[-2])  # Get image label (folder name)

    j += 1



trainImg = np.asarray(trainImg)  # Train images set

trainLabel = pd.DataFrame(trainLabel)  # Train labels set
# Show some of the train images

fig=plt.figure(figsize=(10, 10))

for i in range(8):

    img = fig.add_subplot(2, 4, i + 1)

    index = np.random.randint(num)

    plt.xticks([]),plt.yticks([])

    img.title.set_text(trainLabel[0][index])

    plt.imshow(trainImg[index])

plt.tight_layout()

plt.show()
def preProcessImage(Img_arr, getEx=True):

    clearImg = []

    for img in Img_arr:

        # Use gaussian blur

        blurImg = cv2.GaussianBlur(img, (5, 5), 0)   



        # Convert to HSV image

        hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  



        # Create mask (parameters - green color range)

        lower_green = (25, 40, 50)

        upper_green = (75, 255, 255)



        mask = cv2.inRange(hsvImg, lower_green, upper_green)  

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)



        # Create bool mask

        bMask = mask > 0  



        # Apply the mask

        clear = np.zeros_like(img, np.uint8)  # Create empty image

        clear[bMask] = img[bMask]  # Apply boolean mask to the origin image



        clearImg.append(clear)  # Append image without backgroung



        # Show examples

        if getEx:

            fig = plt.figure(figsize=(10, 10))

            imagels = [img,blurImg,hsvImg,mask,bMask,clear]

            titlels = ['Original Image','Blur Image','HSV Image','Mask','Boolean Mask', 'Clear Image']

            for i in range(6):

                plot = fig.add_subplot(2, 3, i + 1)

                plt.xticks([]),plt.yticks([])

                plot.title.set_text(titlels[i])

                plt.imshow(imagels[i])

            plt.tight_layout()

            plt.show()

            getEx = False



    return(np.asarray(clearImg))
clearTrainImg = preProcessImage(trainImg)
fig=plt.figure(figsize=(10, 10))

for i in range(8):

    img = fig.add_subplot(2, 4, i + 1)

    index = np.random.randint(num)

    plt.xticks([]),plt.yticks([])

    img.title.set_text(trainLabel[0][index])

    plt.imshow(clearTrainImg[index])

plt.tight_layout()

plt.show()
# Normalizing the train Images

x_train = clearTrainImg / 255.0
from sklearn.preprocessing import OneHotEncoder



# Encode labels and create classes

enc = OneHotEncoder(categories='auto')

y_train = enc.fit_transform(trainLabel).toarray()
from sklearn.model_selection import StratifiedShuffleSplit



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.16, random_state=42) # Want a balanced split for all the classes

for train_index, test_index in sss.split(x_train, y_train):

    print("Using {} for training and {} for validation".format(len(train_index), len(test_index)))

    x_train, x_valid = x_train[train_index], x_train[test_index]

    y_train, y_valid = y_train[train_index], y_train[test_index]
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(rotation_range=20,

                            zoom_range=0.15,

                            width_shift_range=0.2,

                            height_shift_range=0.2,

                            shear_range=0.15,

                            horizontal_flip=True,

                            vertical_flip=True,

                            brightness_range=[0.4,1],

                            rescale=1.0/255.0)

datagen.fit(x_train)
from keras import optimizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import BatchNormalization, GlobalAveragePooling2D

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.applications import Xception





num_classes = 12

learning_rate = 0.001

batch_size = 32



base_model = Xception(input_shape=(dim, dim, 3), include_top=False,weights='imagenet')



base_model.trainable = False



model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(100, activation="relu"),

    BatchNormalization(trainable = True,axis=1),

    

    Dropout(0.5),

    

    Dense(50, activation="relu"),

    BatchNormalization(trainable = True,axis=1),

    

    Dense(num_classes,activation='softmax')

])





model.compile(optimizer = optimizers.Nadam(learning_rate=learning_rate),

              loss = 'categorical_crossentropy',

              metrics=['accuracy'])



# callbacks = [ EarlyStopping(monitor='val_loss', patience=5, verbose=0), 

#               ModelCheckpoint(weights, monitor='val_loss', save_best_only=True, verbose=0),

#               ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]



result = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), verbose = 1,

                   batch_size=batch_size, epochs=25, validation_data=(x_valid, y_valid))



(loss, accuracy) = model.evaluate(x_valid, y_valid)



print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
plt.plot(result.history['accuracy'], label='train')

plt.plot(result.history['val_accuracy'], label='valid')

plt.legend(loc='upper left')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()



plt.plot(result.history['loss'], label='train')

plt.plot(result.history['val_loss'], label='valid')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
base_model.trainable = True

model.get_layer('xception').trainable
model.compile(optimizer=optimizers.Nadam(learning_rate=0.0006), loss='categorical_crossentropy', metrics=['accuracy'])
result = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=50, 

                   initial_epoch=25, validation_data=(x_valid, y_valid),verbose=1)
plt.plot(result.history['accuracy'], label='train')

plt.plot(result.history['val_accuracy'], label='valid')

plt.legend(loc='upper left')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()



plt.plot(result.history['loss'], label='train')

plt.plot(result.history['val_loss'], label='valid')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
gc.collect()
model.compile(optimizer=optimizers.Nadam(learning_rate=0.00006), loss='categorical_crossentropy', metrics=['accuracy'])
result = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=75, 

                   initial_epoch=50, validation_data=(x_valid, y_valid),verbose=1)
plt.plot(result.history['accuracy'], label='train')

plt.plot(result.history['val_accuracy'], label='valid')

plt.legend(loc='upper left')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()



plt.plot(result.history['loss'], label='train')

plt.plot(result.history['val_loss'], label='valid')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
gc.collect()
path = '../input/plant-seedlings-classification/test/*.png'

files = glob(path)



testImg = []

testId = []

j = 1

num = len(files)



# Obtain images and resizing, obtain labels

for img in files:

    print("Obtain images: " + str(j) + "/" + str(num), end='\r')

    testId.append(img.split('/')[-1])  # Images id's

    testImg.append(cv2.resize(cv2.imread(img), (dim, dim)))

    j += 1



testImg = np.asarray(testImg)  # Train images set
# Show some of the test images

fig=plt.figure(figsize=(10, 10))

for i in range(8):

    img = fig.add_subplot(2, 4, i + 1)

    index = np.random.randint(num)

    plt.xticks([]),plt.yticks([])

    plt.imshow(testImg[index])

plt.tight_layout()

plt.show()
clearTestImg = preProcessImage(testImg,getEx=True)
# Normalisation of the test images

clearTestImg = clearTestImg / 255
pred = model.predict(clearTestImg)
predNum = np.argmax(pred, axis=1)

predStr = []

for i in range(len(predNum)):

    predStr.append(enc.categories_[0][predNum[i]])

    

res = {'file': testId, 'species': predStr}

res = pd.DataFrame(res)

res.to_csv("submission.csv", index=False)
model.save("saved_model")