import numpy as np 

import pandas as pd

import os

import cv2

import random

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from sklearn.preprocessing import MinMaxScaler



import keras.backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense, Flatten, BatchNormalization, Dropout

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint
imagesPath = '/kaggle/input/utkface-images/utkfaceimages/UTKFaceImages/'

labelsPath = '/kaggle/input/utkface-images/'
files = os.listdir(labelsPath)

labels = pd.read_csv(labelsPath+files[0])
labels = labels[labels.ethnicity != '20170109150557335.jpg.chip.jpg']

labels = labels[labels.ethnicity != '20170116174525125.jpg.chip.jpg']

labels = labels[labels.ethnicity != '20170109142408075.jpg.chip.jpg']



labels = labels.astype({'ethnicity': 'int64'})
images = os.listdir(imagesPath)
def groupAge(age):

#     [0, 5, 18, 24, 26, 27, 30, 34, 38, 46, 55, 65, len(ages)])

    if age>=0 and age<5:

        return 0

    elif age>=5 and age<18:

        return 1

    elif age>=18 and age<24:

        return 2

    elif age>=24 and age<26:

        return 3

    elif age>=26 and age<27:

        return 4

    elif age>=27 and age<30:

        return 5

    elif age>=30 and age<34:

        return 6

    elif age>=34 and age<38:

        return 7

    elif age>=38 and age<46:

        return 8

    elif age>=46 and age<55:

        return 9

    elif age>=55 and age<65:

        return 10

    else:

        return 11
# train:validation:test = 60:10:30 = 14225:948:8535

partitions = {'train': [],

             'validation': [],

             'test': []}

labels_dict = {'train_age': [], 'train_gender': [], 'train_ethnicity': [],

          'validation_age': [], 'validation_gender': [], 'validation_ethnicity': [],

         'test_age': [], 'test_gender': [], 'test_ethnicity': []}



discared_data = []



random.seed(1)

random.shuffle(images)



print("[INFO] Preparing train data....")

for ID in range(14225):

    try:

        data = labels.loc[labels['image_id'] == images[ID][:-4]].values

        labels_dict['train_age'].append(to_categorical(groupAge(data[0][1]), num_classes=12, dtype='float32'))

        labels_dict['train_gender'].append(data[0][2])

        labels_dict['train_ethnicity'].append(to_categorical(data[0][3], num_classes=5, dtype='float32'))

        partitions['train'].append(images[ID])

    except IndexError:

        print("[ERROR]", images[ID])

        discared_data.append(images[ID])

print("[INFO] Done")



print("[INFO] Preparing validation data....")

for ID in range(14225, 15173):

    try:

        data = labels.loc[labels['image_id'] == images[ID][:-4]].values

        labels_dict['validation_age'].append(to_categorical(groupAge(data[0][1]), num_classes=12, dtype='float32'))

        labels_dict['validation_gender'].append(data[0][2])

        labels_dict['validation_ethnicity'].append(to_categorical(data[0][3], num_classes=5, dtype='float32'))

        partitions['validation'].append(images[ID])

    except IndexError:

        print("[ERROR]", images[ID])

        discared_data.append(images[ID])

print("[INFO] Done")



print("[INFO] Preparing test data....")

for ID in range(15173, len(images)):

    try:

        data = labels.loc[labels['image_id'] == images[ID][:-4]].values

        labels_dict['test_age'].append(to_categorical(groupAge(data[0][1]), num_classes=12, dtype='float32'))

        labels_dict['test_gender'].append(data[0][2])

        labels_dict['test_ethnicity'].append(to_categorical(data[0][3], num_classes=5, dtype='float32'))

        partitions['test'].append(images[ID])

    except IndexError:

        print("[ERROR]", images[ID])

        discared_data.append(images[ID])

print("[INFO] Done")
print("[INFO] Training Data")

print("Size of train data: ", len(partitions['train']))

print("Size of age as label: ", len(labels_dict['train_age']))

print("Size of gender as label: ", len(labels_dict['train_gender']))

print("Size of ethnicity as label: ", len(labels_dict['train_ethnicity']))

print("\n")

print("[INFO] Validation Data")

print("Size of validation data: ", len(partitions['validation']))

print("Size of age as label: ", len(labels_dict['validation_age']))

print("Size of gender as label: ", len(labels_dict['validation_gender']))

print("Size of ethnicity as label: ", len(labels_dict['validation_ethnicity']))

print("\n")

print("[INFO] Test Data")

print("Size of test data: ", len(partitions['test']))

print("Size of age as label: ", len(labels_dict['test_age']))

print("Size of gender as label: ", len(labels_dict['test_gender']))

print("Size of ethnicity as label: ", len(labels_dict['test_ethnicity']))
def buildModel():

    inputs = Input(shape=(200,200,3))

    vgg16 = VGG16(weights='imagenet', include_top=False)(inputs)

    x = Flatten()(vgg16)

    x = BatchNormalization()(x)

    x = Dense(1024, activation='relu')(x)

    x = Dense(512, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dense(128, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    x_gender = Dense(1, activation='sigmoid', name='gender')(x)

    x_ethnicity = Dense(5, activation='softmax', name='ethnicity')(x)

    x_age = Dense(12, activation='softmax', name='age')(x)



    model = Model(inputs=inputs, outputs=[x_gender, x_ethnicity, x_age])

    

    return model
model = buildModel()
model.summary()
def loadImages(images, imagesPath, discared_data):

    print("[INFO] Loading....")

    X = []

    count = 0

    for image in images:

        if image in discared_data:

            continue

        if count%1000==0:

            print("[INFO] {} images loaded".format(count))

        img = cv2.imread(imagesPath+image)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        X.append(img)

        count+=1

    print("[INFO] Done")

    return np.array(X)
print("[INFO] Training Data")

trainX = loadImages(partitions['train'], imagesPath, discared_data)

print("[INFO] Validation Data")

validationX = loadImages(partitions['validation'], imagesPath, discared_data)
print("[INFO] no. of Training Images: ", len(trainX))

print("[INFO] no. of Validation Images: ", len(validationX))
trainY = {

    'gender': np.array(labels_dict['train_gender']),

    'ethnicity': np.array(labels_dict['train_ethnicity']),

    'age': np.array(labels_dict['train_age'])

}



validationY = {

    'gender': np.array(labels_dict['validation_gender']),

    'ethnicity': np.array(labels_dict['validation_ethnicity']),

    'age': np.array(labels_dict['validation_age'])

}
trainY['gender'] = trainY['gender'].reshape(trainY['gender'].shape[0], 1)

validationY['gender'] = validationY['gender'].reshape(validationY['gender'].shape[0], 1)
epochs = 10

lr = 1e-3

batch_size = 32
losses = {

    'gender': 'binary_crossentropy',

    'ethnicity': 'categorical_crossentropy',

    'age': 'categorical_crossentropy'

}



losses_weights = {

    'gender': 1.0,

    'ethnicity': 1.0,

    'age': 1.0

}
class MultiOutputDataGenerator(ImageDataGenerator):

    def flow(self,

             x,

             y=None,

             batch_size=32,

             shuffle=True,

             sample_weight=None,

             seed=None,

             save_to_dir=None,

             save_prefix='',

             save_format='png',

             subset=None):



        targets = None

        target_lengths = {}

        ordered_outputs = []

        for output, target in y.items():

            if targets is None:

                targets = target

            else:

                targets = np.concatenate((targets, target), axis=1)

            target_lengths[output] = target.shape[1]

            ordered_outputs.append(output)

            

        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,

                                         shuffle=shuffle):

            target_dict = {}

            i = 0

            for output in ordered_outputs:

                target_length = target_lengths[output]

                target_dict[output] = flowy[:, i: i + target_length]

                i += target_length



            yield flowx, target_dict
multiclassgenerator = MultiOutputDataGenerator(ImageDataGenerator(rescale=1.0/255.0))
traingenerator = multiclassgenerator.flow(trainX, trainY)
validationgenerator = multiclassgenerator.flow(validationX, validationY)
earlyStopper = EarlyStopping(monitor='loss', patience=5)
checkpoint = ModelCheckpoint('{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# opt = Adam(lr=lr, decay=lr / epochs)

# model.compile(optimizer=opt, loss=losses, loss_weights=losses_weights, metrics=['accuracy'])
model.compile(optimizer='sgd', loss=losses, loss_weights=losses_weights, metrics=['accuracy'])
hist = model.fit_generator(traingenerator, validation_data=validationgenerator, epochs=epochs, 

                           steps_per_epoch=len(trainX)//batch_size, validation_steps=len(validationX)//batch_size, 

                           callbacks=[checkpoint, earlyStopper])
lossNames = ['loss', 'gender_loss', 'ethnicity_loss', 

             'age_loss']



plt.style.use("seaborn-whitegrid")

(fig, ax) = plt.subplots(4, 1, figsize=(13, 13))

 

# loop over the loss names

for (i, l) in enumerate(lossNames):

    title = "Loss for {}".format(l) if l != "loss" else "Total loss"

    ax[i].set_title(title)

    ax[i].set_xlabel("Epoch #")

    ax[i].set_ylabel("Loss")

    ax[i].set_xlim([0, epochs])

#     ax[i].set_ylim([0,max(max(hist.history[l]), max(hist.history["val_" + l]))])

    ax[i].plot(hist.history[l], label=l)

    ax[i].plot(hist.history["val_" + l],

		label="val_" + l)

    ax[i].legend()

 

# save the losses figure and create a new figure for the accuracies

plt.tight_layout()

# plt.savefig("{}_losses.png".format(args["plot"]))
lossNames = ['gender_acc', 'ethnicity_acc', 

             'age_acc']



plt.style.use("seaborn-whitegrid")

(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

 

# loop over the loss names

for (i, l) in enumerate(lossNames):

    ax[i].set_title("Accuracy for {}".format(l))

    ax[i].set_xlabel("Epoch #")

    ax[i].set_ylabel("Accuracy")

    ax[i].set_xlim([0, epochs])

#     ax[i].set_ylim([0,max(max(hist.history[l]), max(hist.history["val_" + l]))])

    ax[i].plot(hist.history[l], label=l)

    ax[i].plot(hist.history["val_" + l],

		label="val_" + l)

    ax[i].legend()

 

# save the losses figure and create a new figure for the accuracies

plt.tight_layout()

# plt.savefig("Accuracy_epochs{}.png".format(epochs))
model.save('age-gender-race_v2.h5')