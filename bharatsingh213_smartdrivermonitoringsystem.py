# Import Libraries

import gc

import glob

import os

import random

import cv2



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf

from keras import Input

from keras import Model

from keras.activations import elu, softmax

from keras.applications import VGG16

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import BatchNormalization, Activation

from keras.layers import Conv2D

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import GlobalAveragePooling2D

from keras.layers import MaxPooling2D

from keras.layers.core import Flatten

from keras.losses import categorical_crossentropy

from keras.models import Sequential

from keras.optimizers import SGD,Adam

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
## Hypermeters & Constants

kaggleDir = '/kaggle/input/state-farm-distracted-driver-detection/'

train_img_dir = 'train/'

test_img_dir = 'test/'

CLASSES = {"c0": "safe driving", "c1": "texting - right", "c2": "talking on the phone - right", "c3": "texting - left",

           "c4": "talking on the phone - left", "c5": "operating the radio", "c6": "drinking", "c7": "reaching behind",

           "c8": "hair and makeup", "c9": " talking to passenger"}

IMG_DIM = 299

CHANNEL_SIZE = 3

SEED_VAL = 41

BATCH_SIZE = 28

EPOCHS = 150  # Total Number of epoch
# Set the seed value for repreducing the results

tf.set_random_seed(SEED_VAL)

gc.enable()

np.random.seed(SEED_VAL)

random.seed(SEED_VAL)
# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
## Load Data

df_train = pd.read_csv(kaggleDir + 'driver_imgs_list.csv', low_memory=True)

print('Number of Samples in trainset : {}'.format(df_train.shape[0]))

print('Number Of districted Classes : {}'.format(len((df_train.classname).unique())))



df_train = shuffle(df_train)
print("[INFO] : Load all the images.....")

trainImgDir = os.path.join(kaggleDir, train_img_dir)

testImgDir = os.path.join(kaggleDir, test_img_dir)

trainImgs = glob.glob(trainImgDir + '*/*.jpg')

testImgs = glob.glob(testImgDir + '*.jpg')

len(trainImgs), len(testImgs)
for x in trainImgs:

    print(x)

    break



for x in testImgs:

    print(x)

    break
# Display top five record in csv

df_train.head()
# Dispaly Last five samples from CSV.

df_train.tail()
class_freq_count = df_train.classname.value_counts()



class_freq_count.plot(kind='bar', label='index')

plt.title('Sample Per Class');

plt.show()



plt.pie(class_freq_count, autopct='%1.1f%%', shadow=True, labels=CLASSES.values())

plt.title('Sample % per class');

plt.show()
imgPath = os.path.join(kaggleDir, train_img_dir, "c6/img_20687.jpg")

img = load_img(imgPath)

plt.suptitle(CLASSES['c6'])

plt.imshow(img)
def draw_driver(imgs, df, classId='c0'):

    fig, axis = plt.subplots(2, 3, figsize=(20, 7))

    for idnx, (idx, row) in enumerate(imgs.iterrows()):

        imgPath = os.path.join(kaggleDir, train_img_dir, f"{classId}/{row['img']}")

        row = idnx // 3

        col = idnx % 3 

        img = load_img(imgPath)

        #         img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plt.imshow(img)

        axis[row, col].imshow(img)

    plt.suptitle(CLASSES[classId])

    plt.show()
draw_driver(df_train[df_train.classname == 'c0'].head(6), df_train, classId='c0')
draw_driver(df_train[df_train.classname == 'c1'].head(6), df_train, classId='c1')
draw_driver(df_train[df_train.classname == 'c2'].head(6), df_train, classId='c2')
draw_driver(df_train[df_train.classname == 'c3'].head(6), df_train, classId='c3')


draw_driver(df_train[df_train.classname == 'c4'].head(6), df_train, classId='c4')
draw_driver(df_train[df_train.classname == 'c5'].head(6), df_train, classId='c5')
draw_driver(df_train[df_train.classname == 'c6'].head(6), df_train, classId='c6')
draw_driver(df_train[df_train.classname == 'c7'].head(6), df_train, classId='c7')
draw_driver(df_train[df_train.classname == 'c8'].head(6), df_train, classId='c8')
draw_driver(df_train[df_train.classname == 'c9'].head(6), df_train, classId='c9')
dfY = df_train.classname

x_train, x_test, y_train, y_test = train_test_split(df_train, dfY, test_size=0.15, stratify=dfY)

print('Number of Samples in XTrain : {} Ytrain: {}'.format(x_train.shape[0], y_train.shape[0]))

print('Number of Samples in Xtest : {} Ytest: {}'.format(x_test.shape[0], y_test.shape[0]))
df_train.head()
df_train['file_name']=df_train.img.apply(lambda  x:x[:-4])
df_train.head()
class SimplePreprocessor:

    def __init__(self, width, height, inter=cv2.INTER_AREA):

        #         print("[INFO] : Simple PreProcessor invoked...!")

        self.width = width

        self.height = height

        self.inter = inter



    def preprocess(self, image):

        #         print("[INFO] : Prepossess Resizing invoked...!")

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

class ImageToArrayPreprocessor:

    def __init__(self, dataFormat=None):

        self.dataFormat = dataFormat



    def preprocess(self, image):

        return img_to_array(image, data_format=self.dataFormat)
class SimpleDataLoader:

    def __init__(self, preprocessors=None):

        self.preprocessors = preprocessors

        if self.preprocessors is None:

            self.preprocessors = []



    def load(self, trainImgs, verbose=-1):

        imgData = []

        imgLabels = []

        for (idx, imgPath) in enumerate(trainImgs):

            tmpImg = cv2.imread(imgPath)

            classLabel = imgPath.split(os.path.sep)[-2]



            if self.preprocessors is not None:

                for preprocesor in self.preprocessors:

                    img = preprocesor.preprocess(tmpImg)

                    gc.collect()

                imgData.append(tmpImg)

                imgLabels.append(imgLabels)



            if verbose > 0 and idx > 0 and (idx + 1) % verbose == 0:

                print('[INFO]: Processed {}/{}'.format((idx + 1), len(trainImgs)))

        print(len(imgData), len(imgLabels))

        return np.array(imgData), np.array(imgLabels)
print("[INFO] : Loading data from desk and scale the raw pixel intensities to the range [0,1] ....!")

# sp=SimplePreprocessor(IMG_DIM,IMG_DIM)

# iap=ImageToArrayPreprocessor()

# loader=SimpleDataLoader(preprocessors=[sp,iap])

# (data,labels)=loader.load(trainImgs, verbose=5000)

# # Re-scale the image
# data = data.astype('float') / 255.0

# xtrain,xtest,ytrain,ytest= train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# print("[INFO] : Split dataset....!")

# labelBi = LabelBinarizer()

# trainY = labelBi.fit_transform(ytrain)

# testY = labelBi.fit_transform(ytest)
imgPath = os.path.join(kaggleDir, train_img_dir, "c6/img_20687.jpg")

image=load_img(imgPath)

image=img_to_array(image)

image=np.expand_dims(image, axis=0)

generator = ImageDataGenerator(rotation_range=30,

                               height_shift_range=0.1,

                               width_shift_range=0.1,

                               shear_range=0.2,

                               zoom_range=0.2,

#                                horizontal_flip=True,

                               fill_mode='nearest') 

imageGen=generator.flow(image,batch_size=1)
for i in range(6):

    nextImg=imageGen.next()

    plt.subplot(230 + 1 + i)

    image = nextImg[0].astype('uint8')

    plt.imshow(image)

plt.show()
generator = ImageDataGenerator(rescale=1 / 255.0,

                               zoom_range=30,

                               samplewise_center=True,

                               height_shift_range=0.2,

                               width_shift_range=0.2,

                               shear_range=0.2, 

                               fill_mode='nearest',

                               validation_split=0.15)


train_generator = generator.flow_from_directory(directory=os.path.join(kaggleDir, train_img_dir),

                                                classes=CLASSES.keys(),

                                                class_mode='categorical',

                                                color_mode="rgb",

                                                target_size=(IMG_DIM, IMG_DIM),

                                                shuffle=True,

                                                seed=SEED_VAL,

                                                subset='training')

valid_generator = generator.flow_from_directory(directory=os.path.join(kaggleDir, train_img_dir),

                                                classes=CLASSES.keys(),

                                                class_mode='categorical',

                                                color_mode="rgb",

                                                target_size=(IMG_DIM, IMG_DIM),

                                                shuffle=True,

                                                seed=SEED_VAL,

                                                subset='validation')

train_generator.class_indices

gc.collect()
train_generator.class_indices,valid_generator.samples
trainImgs[:5]
earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=8, verbose=1, min_delta=0.0000001)

reduceRL = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=7, factor=0.001, min_delta=0.0001, verbose=1,

                             min_lr=1e-6)

callbacks = [reduceRL, earlyStop]


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=[IMG_DIM, IMG_DIM, 1], activation=elu))

model.add(Activation(activation=elu))

model.add(MaxPooling2D())

model.add(BatchNormalization())



model.add(Conv2D(filters=32, kernel_size=(3, 3), activation=elu))

model.add(Activation(activation=elu))

model.add(MaxPooling2D())

model.add(BatchNormalization())



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=elu))

model.add(Activation(activation=elu))

model.add(MaxPooling2D())

model.add(BatchNormalization())



model.add(Conv2D(filters=128, kernel_size=(3, 3), activation=elu))

model.add(Activation(activation=elu))

model.add(MaxPooling2D())

model.add(BatchNormalization())



model.add(Conv2D(filters=256, kernel_size=(3, 3), activation=elu))

model.add(Activation(activation=elu))

model.add(MaxPooling2D())

model.add(BatchNormalization())



model.add(Conv2D(filters=512, kernel_size=(3, 3), activation=elu))

model.add(Activation(activation=elu))

model.add(MaxPooling2D())

model.add(BatchNormalization())



# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation=elu))

# model.add(Activation(activation=elu))

# model.add(MaxPooling2D())

# model.add(BatchNormalization())



model.add(GlobalAveragePooling2D())

model.add(Dense(3000))

model.add(Activation(activation=elu))

model.add(Dropout(rate=0.25))

model.add(Dense(2000))

model.add(Activation(activation=elu))

model.add(Dropout(rate=0.25))

model.add(Dense(len(CLASSES)))

model.add(Activation(activation=softmax))

model.summary()
# opt = SGD()#lr=0.0001

from keras.optimizers import adam

opt=adam()

model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['acc'])

# History = model.fit_generator(train_generator,

#                               steps_per_epoch=train_generator.samples // BATCH_SIZE,

#                               validation_data=valid_generator,

#                               validation_steps=valid_generator.samples // BATCH_SIZE,

#                               epochs=EPOCHS,

#                               verbose=1).history  # , callbacks=callbacks
# eval_loss, eval_acc = model.evaluate_generator(valid_generator, steps=valid_generator.samples / BATCH_SIZE);

# print('[INFO] : Evaluation Accuracy : {:.2f}%'.format(eval_acc * 100))

# print('[INFO] : Evaluation Loss : {}'.format(eval_loss))
# History.keys()
# plt.style.use('ggplot')

# plt.figure()

# plt.plot(np.arange(0, EPOCHS), History['acc'], label='Train_Acc')



# plt.plot(np.arange(0, EPOCHS), History['val_acc'], label='Valid_Acc')

# plt.plot(np.arange(0, EPOCHS), History['val_loss'], label='Valid_Loss')

# plt.plot(np.arange(0, EPOCHS), History['loss'], label='Train_Loss')

# plt.xlabel('Epochs#')

# plt.ylabel('Accuracy and Loss#')

# plt.title("Loss and Accuracy")

# plt.legend()

# plt.show()
class FCHeadNet:

    @staticmethod

    def build(baseModel, classes, D):

        # initialize the head model that will be placed on top of the base then ad a Fully Connected Layer

        headModel = baseModel.output

        headModel = Flatten(name='flatten')(headModel)

        headModel = Dense(D, activation='elu')(headModel)

        headModel = Dropout(0.2)(headModel) 



        # add Softmax Layer

        headModel = Dense(classes, activation='softmax')(headModel)



        # Return the model

        return headModel
baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_DIM, IMG_DIM,3)))
# initialize the new head of the network, a set of FC layers followed by a softmax classifier

headModel = FCHeadNet.build(baseModel, len(CLASSES), 10000)
# place the head FC model on top of the base model -- this will become the actual model we will train

model = Model(inputs=baseModel.input, outputs=headModel)



# loop over all layers in the base model and freeze them so they will *not* be updated during the training process

for layer in baseModel.layers:

    layer.trainable = False



# compile our model (this needs to be done after our setting our  layers to being non-trainable

print("[INFO] compiling model...")

opt = Adam(lr=0.001)

model.compile(optimizer=opt, loss=['categorical_crossentropy'], metrics=['acc'])



# train the head  of the network for a  few epochs (all other layers are frozen) -- this will allow to the new FC

# layers to start to become initialized with actual  'learned ' values versus pure random

model.summary()
# loop over the layers in the network and display them to the console

for (idx, layer) in enumerate(model.layers):

    print('[INFO] : {} \t {}'.format(idx, layer.__class__.__name__))

History = model.fit_generator(train_generator,

                              steps_per_epoch=train_generator.samples // BATCH_SIZE,

                              validation_data=valid_generator,

                              validation_steps=valid_generator.samples // BATCH_SIZE,

                              epochs=EPOCHS,

                              verbose=1 , callbacks=callbacks).history  #
eval_loss, eval_acc = model.evaluate_generator(valid_generator, steps=valid_generator.samples / BATCH_SIZE);

print('[INFO] : Evaluation Accuracy : {:.2f}%'.format(eval_acc * 100))

print('[INFO] : Evaluation Loss : {}'.format(eval_loss))
plt.style.use('ggplot')

plt.figure()

plt.plot(np.arange(0, EPOCHS), History['acc'], label='Train_Acc')



plt.plot(np.arange(0, EPOCHS), History['val_acc'], label='Valid_Acc')

plt.plot(np.arange(0, EPOCHS), History['val_loss'], label='Valid_Loss')

plt.plot(np.arange(0, EPOCHS), History['loss'], label='Train_Loss')

plt.xlabel('Epochs#')

plt.ylabel('Accuracy and Loss#')

plt.title("Loss and Accuracy")

plt.legend()

plt.show()
from keras.applications import DenseNet201



model=DenseNet201(include_top=True, weights='imagenet')