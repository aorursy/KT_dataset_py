import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
train_csv = pd.read_csv('../input/train-csv/legend.csv')
train_csv['emotion'] = train_csv['emotion'].str.lower()
train_csv.groupby('emotion').count()
train_csv.replace("contempt", "anger", inplace=True)
train_csv.groupby('emotion').count()
mapping_emotion = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'neutral': 6, 'sadness': 4, 'surprise': 5}
train_csv['label'] = train_csv['emotion'].map(mapping_emotion)
train_csv.head()
import glob
import cv2 as cv
import os
trained = '../input/trainedimages'
#os.mkdir(trained)
face_cascade = cv.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')
image_train = '../input/trainimages/images_train/images'
deleting = ['Abdul_Majeed_Shobokshi_0001.jpg',
 'Arsinee_Khanjian_0001.jpg',
 'Avinash_30.jpg',
 'Colin_Montgomerie_0004.jpg',
 'Colin_Powell_0048.jpg',
 'David_McCullough_0001.jpg',
 'Donald_Rumsfeld_0117.jpg',
 'Fernando_Vargas_0004.jpg',
 'Franz_Muentefering_0003.jpg',
 'George_HW_Bush_0003.jpg',
 'George_Pataki_0002.jpg',
 'Hans_Blix_0016.jpg',
 'Isaiah_Washington_0002.jpg',
 'Jeff_Feldman_0001.jpg',
 'Jiang_Zemin_0002.jpg',
 'Jiang_Zemin_0007.jpg',
 'Joe_Vandever_0001.jpg',
 'John_Wright_0001.jpg',
 'Kimberly_Bruckner_0001.jpg',
 'Kimberly_Stewart_0001.jpg',
 'Kimi_Raikkonen_0001.jpg',
 'Kimi_Raikkonen_0002.jpg',
 'Kimi_Raikkonen_0003.jpg',
 'Kimora_Lee_0001.jpg',
 'Lin_Yi-fu_0001.jpg',
 'Luciano_Pavarotti_0002.jpg',
 'Lynne_Thigpen_0001.jpg',
 'Michael_Powell_0003.jpg',
 'Miguel_Contreras_0001.jpg',
 'Morgan_Freeman_0002.jpg',
 'Padraig_Harrington_0004.jpg',
 'Paul_Bremer_0014.jpg',
 'Pedro_Malan_0003.jpg',
 'Pierce_Brosnan_0007.jpg',
 'Pyar_Jung_Thapa_0001.jpg',
 'Richard_Gephardt_0007.jpg',
 'Robert_Horan_0002.jpg',
 'Robert_Zoellick_0005.jpg',
 'Rob_Moore_0001.jpg',
 'Scott_McNealy_0001.jpg',
 'Thomas_Daily_0001.jpg',
 'Tony_Blair_0090.jpg',
 'William_Bulger_0002.jpg',
 'Will_Ferrell_0001.jpg']
data = []
labels = []
i = 0
for img in glob.glob(image_train+"/*.jpg"):
    image = cv.imread(img)
    name = img.split('/')[-1]
    
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert to greyscale
    height, width = image.shape[:2]
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 1)
    if isinstance(faces, tuple):
        resized_image = cv.resize(gray_image, (48, 48))
        cv.imwrite(trained+'/'+name,resized_image)
    #print(faces)
    elif isinstance(faces, np.ndarray):
        for (x,y,w,h) in faces:
            if w * h < (height * width) / 3:
                resized_image = cv.resize(gray_image, (48, 48)) 
                cv.imwrite(trained+'/'+name,resized_image)
            else:
                
                #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_image[y:y+h, x:x+w]
                #print(len(roi_gray))
                resized_image = cv.resize(roi_gray, (48, 48))
                cv.imwrite(trained+'/'+name, resized_image)
    if not name in deleting:
        data.append(img_to_array(resized_image))
        label = int(train_csv[ train_csv['image'] == name][['label']].values)
        #print(label, type(label), name)
        labels.append(label)
    """if i == 300:
        break
    i = i + 1"""
int(train_csv[ train_csv['image'] == 'Al_Sharpton_0004.jpg'][['label']].values)
len(labels), len(data)
type(data), type(labels)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
print(data.shape, labels.shape)
from sklearn.preprocessing import LabelBinarizer
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
from sklearn.model_selection import train_test_split
(trainX, valX, trainY, valY) = train_test_split(data,labels, test_size=0.3, random_state=42)
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
def buildModel(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL
		model.add(Conv2D(64, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
model = buildModel(width=48, height=48,depth=1, classes=len(lb.classes_))
from keras.optimizers import Adam
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(valX, valY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
model.save('../input/emotion.model')
import pickle
f = open("../input/lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()
image_path = '../input/prediction/prediction/57b.jpg'

image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = cv.resize(image, (48, 48))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image.shape
df_test = pd.read_csv('../input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv')
df_test = df_test.sample(frac=0.10)
X_test = []
test_index = df_test.index
for item in df_test.index:
    pixels = df_test.pixels[item]
    pixels = pixels.split(' ')
    piarray = np.asarray(pixels, dtype=np.int64)
    re = piarray.reshape(48,48)
    X_test.append(re)
X_test = np.asarray(X_test)
X_test = np.expand_dims(X_test, axis=-1)
X_test[0].shape
"""proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]
"""
Y_predict = []
for item in X_test:
    item = np.expand_dims(item, axis = 0)
    predict = model.predict(item)
    idx = np.argmax(predict)
    label = lb.classes_[idx]
    Y_predict.append(label)
Y_true = df_test.emotion.values
from sklearn.metrics import accuracy_score
accuracy_score(Y_true, Y_predict)

