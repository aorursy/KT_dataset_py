import cv2

import os

from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import *

from tensorflow.keras.layers import *

from tensorflow.keras.models import *

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import *

from tensorflow.keras.callbacks import *
imagePaths = []

for dirname, _, filenames in os.walk('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/'):

    for filename in filenames:

        if (filename[-3:] == 'png'):

            imagePaths.append(os.path.join(dirname, filename))
imgSize = 224
X = []

Y = []

hmap = {'Viral Pneumonia': 'Pneumonia', 'NORMAL': 'Normal', 'COVID-19': 'Covid-19'}

for imagePath in tqdm(imagePaths):

    label = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (imgSize, imgSize))



    X.append(image)

    Y.append(hmap[label])

    
print('Covid-19:',Y.count('Covid-19'))

print('Normal:',Y.count('Normal'))

print('Pneumonia: ',Y.count('Pneumonia'))
le = LabelEncoder()

Y = le.fit_transform(Y)

Y = to_categorical(Y)
(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.20, stratify=Y, random_state=42)
del X

del Y
print(len(trainY))

ntimes = 6

trainY = trainY.tolist()

for i in tqdm(range(len(trainX))):

    if (trainY[i][0] == 1):

        trainX += [trainX[i]]*ntimes

        trainY += [trainY[i]]*ntimes

        

trainY = np.array(trainY)



print(len(trainY))
trainX = np.array(trainX).astype('float16')/255



testX = np.array(testX).astype('float16')/255
trainAug = ImageDataGenerator(rotation_range=20, horizontal_flip = True,fill_mode="nearest")
best_val_acc = 0

best_train_acc = 0

def saveModel(epoch,logs):

    val_acc = logs['val_accuracy']

    train_acc = logs['accuracy']

    global best_val_acc

    global best_train_acc



    if val_acc > best_val_acc:

        best_val_acc = val_acc

        model.save('model.h5')

    elif val_acc == best_val_acc:

        if train_acc > best_train_acc:

            best_train_acc= train_acc

            model.save('model.h5')
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(imgSize, imgSize, 3)))

headModel = baseModel.output

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(64, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:

    layer.trainable = False
INIT_LR = 3e-4

EPOCHS = 50

BS = 32

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit_generator(

    trainAug.flow(trainX, trainY, batch_size=BS),

    steps_per_epoch=len(trainX) // BS,

    validation_data=(testX, testY),

    validation_steps=len(testX) // BS,

    callbacks= [LambdaCallback(on_epoch_end=saveModel),

#              EarlyStopping(monitor='val_accuracy', patience=3),

#              ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,patience=2),

              ],

    epochs=EPOCHS)
N = EPOCHS

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on COVID-19 Dataset")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="lower left")
model= load_model('model.h5')
predIdxs = model.predict(trainX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(trainY.argmax(axis=1), predIdxs, target_names=le.classes_, digits = 5))
predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=le.classes_, digits = 5))