import logging

logging.getLogger().setLevel(logging.ERROR)



import os, os.path

from os import listdir

from skimage.io import imread, imsave

from skimage.transform import resize

from skimage import exposure

import shutil

import numpy as np

import cv2



TRAIN_DATA_DIR="../input/chest_xray/chest_xray/train/"

VALIDATION_DATA_DIR="../input/chest_xray/chest_xray/val/"

TEST_DATA_DIR="../input/chest_xray/chest_xray/test/"



dir_to_transform = ['train', 'val', 'test']

classes = ['NORMAL', 'PNEUMONIA']



if os.path.isdir('../transformed'):

    shutil.rmtree('../transformed')

os.mkdir('../transformed')



for f in dir_to_transform:

    os.mkdir('../transformed/' + f)

    for c in classes:

        input_path = "../input/chest_xray/chest_xray/" + f + '/' + c

        output_path = '../transformed/' + f + '/' + c

        os.mkdir(output_path)

        for image_file in listdir(input_path):

            if image_file.endswith('.jpeg'):

                full_input_path = input_path + '/' + image_file

                full_output_path = output_path + '/' + image_file

                image = imread(full_input_path)

                image = resize(image, (256, 256), anti_aliasing=False)

                image = image[32:224,32:224]                

                image = image/255.

                image = exposure.equalize_adapthist(image)

                imsave(full_output_path, image)              
import matplotlib.pyplot as plt



image_file = "IM-0031-0001.jpeg"

raw_image = imread("../input/chest_xray/chest_xray/test/NORMAL/" + image_file)

raw_image = resize(raw_image, (288, 288), anti_aliasing=False)

transformed_image = imread("../transformed/test/NORMAL/" +  image_file)



fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(10,10))

axs[0].imshow(raw_image, cmap='gray')

axs[0].set_title('Raw image')

axs[0].axis('off')

axs[1].imshow(transformed_image, cmap='gray')

axs[1].set_title('CLAHE image')

axs[1].axis('off')

plt.show()
from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D, Add, Concatenate, Subtract, Dot, Average, Lambda, SeparableConv2D, DepthwiseConv2D

from keras import regularizers



from keras import backend as K



input_shape = (192, 192, 3)

input_layer = Input(shape=input_shape, name='input_layer')



output = Conv2D(48, (1,1), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))(input_layer)

output = MaxPool2D(pool_size=(2, 2))(output)

output = BatchNormalization()(output)

output = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))(output)

output = MaxPool2D(pool_size=(2, 2))(output)

output = BatchNormalization()(output)

output = Conv2D(16, (5,5), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))(output)

output = BatchNormalization()(output)

output = Flatten()(output)

output = Dense(units=256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))(output)

output = BatchNormalization()(output)

output = Dropout(rate=0.2)(output)

output = Dense(units=256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))(output)

output = BatchNormalization()(output)

output = Dropout(rate=0.2)(output)

output = Dense(units=1, activation='sigmoid')(output)



model = Model(inputs=input_layer, outputs=output)

model.summary()
from keras.preprocessing.image import ImageDataGenerator



class Generators():

    train_dir = None

    validation_dir = None

    test_dir = None



    def __init__(self, train_dir, validation_dir, test_dir):

        self.train_dir = train_dir

        self.validation_dir = validation_dir

        self.test_dir = test_dir



    def train_generator(self, image_size, batch_size=32):

        datagen = ImageDataGenerator(rescale=1./255,

                                     rotation_range=8,

                                     width_shift_range=25,

                                     height_shift_range=25,

                                     zoom_range=(0.95, 1.1),

                                     brightness_range=(0.8,1.2),

                                     shear_range=10,

                                     fill_mode = "constant",

                                     horizontal_flip=True,

                                     vertical_flip=False,

                                     cval=0

                                    )

        data_generator = datagen.flow_from_directory(

            self.train_dir,

            target_size=(image_size, image_size),

            batch_size=batch_size,

            class_mode='binary')

        return data_generator



    def validation_generator(self, image_size, batch_size=32):

        datagen = ImageDataGenerator(rescale=1./255)

        data_generator = datagen.flow_from_directory(

            self.validation_dir,

            target_size=(image_size, image_size),

            batch_size=batch_size,

            class_mode='binary')

        return data_generator     



    def test_generator(self, image_size, batch_size=1, shuffle=False):

        datagen = ImageDataGenerator(rescale=1./255)

        data_generator = datagen.flow_from_directory(

            self.test_dir,

            target_size=(image_size, image_size),

            batch_size=batch_size,

            shuffle=shuffle,

            class_mode='binary')

        return data_generator
from keras.optimizers import Adam, Nadam, SGD

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping



optimizer = Adam()

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



modelCheckpoint = ModelCheckpoint(monitor='val_loss', filepath="simple_cnn_model", save_best_only=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.000001, verbose=1)

earlyStopping = EarlyStopping(patience=11, restore_best_weights=True)



TRAIN_DATA_DIR="../transformed/train/"

VALIDATION_DATA_DIR="../transformed/val/"

TEST_DATA_DIR="../transformed/test/"



generators = Generators(TRAIN_DATA_DIR, VALIDATION_DATA_DIR, TEST_DATA_DIR)

train_generator = generators.train_generator(192, batch_size=32)

validation_generator = generators.validation_generator(192, batch_size=16)



model.fit_generator(train_generator, 

                    epochs=100,

                    steps_per_epoch=100, 

                    validation_data=validation_generator, 

                    validation_steps=1, 

                    callbacks=[reduce_lr, modelCheckpoint, earlyStopping], 

                    verbose=1, 

                    class_weight={0:1.0, 1:0.33})


from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve



from keras.models import load_model

import matplotlib.pyplot as plt





class ResultUtils():

    def plot_confusion_matrix(self, predicted_classes, true_classes):

        cf_matrix = confusion_matrix(true_classes, predicted_classes)

        plot_confusion_matrix(conf_mat=cf_matrix, figsize=(5, 5))

        plt.xlabel('Predicted')

        plt.ylabel('True')

        plt.show()



        acc = accuracy_score(true_classes, predicted_classes)*100

        tn, fp, fn, tp = confusion_matrix(true_classes, predicted_classes).ravel()



        print('Accuracy: {0:.2f}%'.format(acc))

        print('Precision: {0:.2f}%'.format(tp/(tp+fp)*100))

        print('Recall: {0:.2f}%'.format(tp/(tp+fn)*100))

        

    def find_ratio_of_normal_class(self, true_classes):

        total = len(true_classes)

        normal = np.count_nonzero(true_classes != 1)

        return float(normal)/total



resultUtils = ResultUtils()



test_generator = generators.test_generator(192, batch_size=1)

true_classes = test_generator.classes



predicted_classes = model.predict_generator(test_generator, 624, verbose=1)



percentage_of_normal = resultUtils.find_ratio_of_normal_class(true_classes)

predicted_classes_after_threshold = predicted_classes  > np.percentile(predicted_classes, percentage_of_normal*100)



resultUtils.plot_confusion_matrix(true_classes, predicted_classes_after_threshold)
TEST_DATA_DIR="../transformed/test/"



tta_steps = 7

predictions = []



datagen = ImageDataGenerator(rescale=1./255,

                                     rotation_range=12,

                                     width_shift_range=15,

                                     height_shift_range=15,

                                     zoom_range=(0.95, 1.05),

                                     brightness_range=(0.8,1.2),

                                     shear_range=10,

                                     fill_mode = "constant",

                                     horizontal_flip=True,

                                     vertical_flip=False,

                                     cval=0

                                    )

data_generator = datagen.flow_from_directory(

    TEST_DATA_DIR,

    target_size=(192, 192),

    batch_size=1,

    class_mode='binary',

    shuffle=False)



for i in range(tta_steps):

    preds = model.predict_generator(data_generator, 624, verbose=1)

    predictions.append(preds)



predicted_classes = np.mean(predictions, axis=0)



resultUtils = ResultUtils()

true_classes = test_generator.classes



percentage_of_normal = resultUtils.find_ratio_of_normal_class(true_classes)

predicted_classes_after_threshold = predicted_classes  > 0.85



resultUtils.plot_confusion_matrix(true_classes, predicted_classes_after_threshold)
import shap



TRAIN_DATA_DIR="../transformed/train/"

VALIDATION_DATA_DIR="../transformed/val/"

TEST_DATA_DIR="../transformed/test/"



generators = Generators(TRAIN_DATA_DIR, VALIDATION_DATA_DIR, TEST_DATA_DIR)



train_generator = generators.train_generator(192, batch_size=100)

test_generator = generators.test_generator(192, batch_size=5, shuffle=True)



train_images = next(train_generator)[0]



test_set = next(test_generator)

test_images = test_set[0]

test_images_labels = test_set[1]



test_images_labels = np.reshape(test_images_labels, (5,1))

test_images_labels = np.where(test_images_labels > 0.5, 'PNEUMONIA', 'NORMAL')



e = shap.DeepExplainer(model, train_images)

shap_values = e.shap_values(test_images)

shap.image_plot(shap_values, test_images, labels=test_images_labels)