# Helper libraries

import tensorflow

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import cv2

import os

import random

from tqdm import tqdm_notebook as tqdm



%matplotlib inline
RANDOM_SEED = 2019

np.random.seed(RANDOM_SEED)

tensorflow.set_random_seed(RANDOM_SEED)

random.seed(RANDOM_SEED) # Python

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
!pip install keras_efficientnets
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_df['id_code'] = train_df['id_code'].apply(lambda x:'../input/aptos2019-blindness-detection/train_images/' + x + '.png')

train_df['diagnosis'] = train_df['diagnosis'].astype(str)

num_classes = train_df['diagnosis'].nunique()

diag_text = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']



train_df.diagnosis.value_counts()
old_train_df = pd.read_csv('../input/diabetic-retinopathy-resized/trainLabels.csv')

old_train_df = old_train_df[['image','level']]

old_train_df.columns = train_df.columns

old_train_df['id_code'] = old_train_df['id_code'].apply(lambda x:'../input/diabetic-retinopathy-resized/resized_train/resized_train/' + x + '.jpeg')

old_train_df['diagnosis'] = old_train_df['diagnosis'].astype(str)



old_train_df.diagnosis.value_counts()
def display_raw_images(df, columns = 4, rows = 2):

    fig=plt.figure(figsize = (5 * columns, 4 * rows))

    for i in range(columns * rows):

        image_name = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        img = cv2.imread(image_name)[...,[2, 1, 0]]

        fig.add_subplot(rows, columns, i + 1)

        plt.title(diag_text[int(image_id)])

        plt.imshow(img)

    plt.tight_layout()
display_raw_images(train_df)
display_raw_images(old_train_df)
unique, counts = np.unique(train_df['diagnosis'], return_counts=True)

plt.bar(unique, counts)

plt.title('Class Frequency')

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
unique, counts = np.unique(old_train_df['diagnosis'], return_counts = True)

plt.bar(unique, counts)

plt.title('Class Frequency')

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
# Divide by class

old_train_df_class_0 = old_train_df[old_train_df['diagnosis'] == '0']

old_train_df_class_1 = old_train_df[old_train_df['diagnosis'] == '1']

old_train_df_class_2 = old_train_df[old_train_df['diagnosis'] == '2']

old_train_df_class_3 = old_train_df[old_train_df['diagnosis'] == '3']

old_train_df_class_4 = old_train_df[old_train_df['diagnosis'] == '4']





train_df_plus = pd.concat([train_df,

                           old_train_df_class_0.sample(2000),

                           old_train_df_class_1.sample(2000),

                           old_train_df_class_2.sample(2000),

                           old_train_df_class_3.sample(873),

                           old_train_df_class_4.sample(708)],

                          axis=0)



train_df_plus = train_df_plus.sample(frac = 1).reset_index(drop = True)



print('After random under-sampling: ')

train_df_plus.diagnosis.value_counts()
unique, counts = np.unique(train_df_plus['diagnosis'], return_counts = True)

plt.bar(unique, counts)

plt.title('Class Frequency')

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
from sklearn.utils import class_weight



sklearn_class_weights = class_weight.compute_class_weight(

               'balanced',

                np.unique(train_df['diagnosis']), 

                train_df['diagnosis'])



print(sklearn_class_weights)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, LeakyReLU

from keras.models import Model, Sequential

from keras.optimizers import Adam 



from keras_efficientnets import EfficientNetB5



def create_effnetB5_model(input_shape, n_out):

    model = Sequential()

    base_model = EfficientNetB5(weights = 'imagenet', 

                                include_top = False,

                                input_shape = input_shape)

    base_model.name = 'base_model'

    model.add(base_model)

    model.add(Dropout(0.25))

    model.add(Dense(1024))

    model.add(LeakyReLU())

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))   

    model.add(Dense(n_out, activation = 'sigmoid'))

    return model
IMAGE_HEIGHT = 340

IMAGE_WIDTH = 340

model = create_effnetB5_model(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), n_out = num_classes)

model.summary()
y_train = pd.get_dummies(train_df['diagnosis']).values

y_train_multi = np.empty(y_train.shape, dtype = y_train.dtype)

y_train_multi[:, 4] = y_train[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i + 1])

    

x_train = train_df['id_code']
y_train_old = pd.get_dummies(old_train_df['diagnosis']).values

y_train_multi_old = np.empty(y_train_old.shape, dtype = y_train_old.dtype)

y_train_multi_old[:, 4] = y_train_old[:, 4]



for i in range(3, -1, -1):

    y_train_multi_old[:, i] = np.logical_or(y_train_old[:, i], y_train_multi_old[:, i + 1])

    

x_train_old = old_train_df['id_code']

y_train_old = y_train_multi_old
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size = 0.50, 

    random_state = RANDOM_SEED

)
import imgaug as ia

from imgaug import augmenters as iaa



sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.1), # horizontally flip 10% of all images

            iaa.Flipud(0.1), # vertically flip 10% of all images

            sometimes(iaa.Affine(

                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, # scale images to 95-105% of their size, individually per axis

                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -5 to +5 percent (per axis)

                rotate=(-180, 180), # rotate by -180 to +180 degrees

                shear=(-3, 3), # shear by -3 to +3 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 3),

                [

                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                    iaa.OneOf([

                        iaa.GaussianBlur((0, 0.5)), # blur images with a sigma between 0 and 0.5

                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7

                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7

                    ]),

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    # search either for all edges or for directed edges,

                    # blend the result with the original image using a blobby mask

                    iaa.SimplexNoiseAlpha(iaa.OneOf([

                        iaa.EdgeDetect(alpha=(0.5, 1.0)),

                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                    ])),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images

                    iaa.OneOf([

                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 5% of the pixels

                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),

                    ]),

                    iaa.Invert(0.01, per_channel = True), # invert color channels

                    iaa.Add((-2, 2), per_channel = 0.5), # change brightness of images (by -5 to 5 of original value)

                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation

                    # either change the brightness of the whole image (sometimes

                    # per channel) or change the brightness of subareas

                    iaa.OneOf([

                        iaa.Multiply((0.9, 1.1), per_channel = 0.5),

                        iaa.FrequencyNoiseAlpha(

                            exponent = (-1, 0),

                            first = iaa.Multiply((0.9, 1.1), per_channel = True),

                            second = iaa.ContrastNormalization((0.9, 1.1))

                        )

                    ]),

                    sometimes(iaa.ElasticTransformation(alpha = (0.5, 3.5), sigma = 0.25)), # move pixels locally around (with random strengths)

                    sometimes(iaa.PiecewiseAffine(scale = (0.01, 0.05))), # sometimes move parts of the image around

                    sometimes(iaa.PerspectiveTransform(scale = (0.01, 0.1)))

                ],

                random_order = True

            )

        ],

        random_order = True)
def crop_image_from_gray(img, tol = 7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis = -1)

    #         print(img.shape)

        return img

    

def process_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    image=cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)

    return image

from keras.utils import Sequence

from sklearn.utils import shuffle



class Aptos2019Generator(Sequence):



    def __init__(self, image_filenames, labels,

                 batch_size, is_train = True,

                 mix = False, augment = False):

        self.image_filenames = image_filenames

        self.labels = labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.is_augment = augment

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        else:

            return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

        else:

            pass

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)             

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread(sample)

#            img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))

            img = process_image(img)

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        if(self.is_mix):

            batch_images, batch_y = self.mix_up(batch_images, batch_y)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread(sample)

#            img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))

            img = process_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
BATCH_SIZE = 8

train_generator = Aptos2019Generator(x_train, y_train, BATCH_SIZE, is_train = True, augment = False, mix = False)

valid_generator = Aptos2019Generator(x_val, y_val, BATCH_SIZE, is_train = False)
from keras.callbacks import Callback

from sklearn.metrics import cohen_kappa_score



class QWK(Callback):

    def __init__(self, validation_data = (), batch_size = 64, interval=1):

        super(Callback, self).__init__()

        self.interval = interval

        self.batch_size = batch_size

        self.valid_generator, self.y_val = validation_data

        self.history = []

        self.max_score = float("-inf")



    def on_epoch_end(self, epoch, logs = {}):

        if epoch % self.interval == 0:

            validation_predictions_raw = self.model.predict_generator(generator=self.valid_generator,

                                                  steps = np.ceil(float(len(self.y_val)) / float(self.batch_size)),

                                                  workers = 1, use_multiprocessing=False,

                                                  verbose = 1)           

            validation_predictions = validation_predictions_raw > 0.5

            validation_predictions = validation_predictions.astype(int).sum(axis=1) - 1

            validation_truth = y_val.sum(axis=1) - 1              

            score = cohen_kappa_score(validation_predictions, validation_truth, weights = 'quadratic')

            self.history.append(score)

            print("epoch: %d - qwk_score: %.6f" % (epoch + 1, score))

            if score >= self.max_score:

                print('qwk_score improved from %.6f to %.6f, saving model to blindness_detector_best_qwk.h5' % (self.max_score, score))             

                self.model.save('../working/blindness_detector_best_qwk.h5')

                self.max_score = score



qwk = QWK(

    validation_data = (valid_generator, y_val), 

    batch_size = BATCH_SIZE, 

    interval = 1)



checkpoint = ModelCheckpoint(

    'blindness_detector_best.h5', 

    monitor = 'val_acc',  

    save_best_only = True, 

    save_weights_only = False,

    verbose = 1)



rlrop = ReduceLROnPlateau(

    monitor = 'val_loss', 

    patience = 3, 

    factor = 0.5, 

    min_lr = 1e-6, 

    verbose = 1)



stopping = EarlyStopping(

    monitor = 'val_acc', 

    patience = 8, 

    restore_best_weights = True, 

    verbose = 1)
WARMUP_EPOCHS = 3

WARMUP_LEARNING_RATE = 1e-3

    

for layer in model.layers:

    if layer.name == 'base_model':

        layer.trainable = False        

    else:

        layer.trainable = True       



model.compile(optimizer = Adam(lr = WARMUP_LEARNING_RATE),

              loss = 'binary_crossentropy',  

              metrics = ['accuracy'])



warmup_history = model.fit_generator(generator = train_generator,

                                     steps_per_epoch = len(train_generator),

                                     epochs = WARMUP_EPOCHS,

                                     validation_data = valid_generator,

                                     validation_steps = len(valid_generator),

                                     callbacks = [qwk],

                                     use_multiprocessing = True,

                                     verbose = 1).history
FINETUNING_EPOCHS = 20

FINETUNING_LEARNING_RATE = 1e-4



for layer in model.layers:

    layer.trainable = True

 #   if layer.name == 'base_model':   

 #       set_trainable = False

 #       for sub_layer in layer.layers:

 #           if sub_layer.name == 'multiply_16':

 #               set_trainable = True

 #           if set_trainable:

 #               sub_layer.trainable = True

 #           else:

 #               sub_layer.trainable = False    

    

model.compile(optimizer = Adam(lr = FINETUNING_LEARNING_RATE), 

              loss = 'binary_crossentropy',

              metrics = ['accuracy'])







train_generator_augmented = Aptos2019Generator(x_train, y_train, BATCH_SIZE, is_train = True, mix = True, augment = True)



finetune_history = model.fit_generator(

                              generator = train_generator_augmented,

#                              class_weight = sklearn_class_weights,

                              steps_per_epoch = len(train_generator_augmented),

                              validation_data = valid_generator,

                              validation_steps = len(valid_generator),

                              epochs = FINETUNING_EPOCHS,

                              callbacks = [rlrop, qwk],         

                              use_multiprocessing = True,

#                              workers = 2,

                              verbose = 1).history
training_accuracy = warmup_history['acc'] + finetune_history['acc'] 

validation_accuracy = warmup_history['val_acc'] + finetune_history['val_acc']

training_loss = warmup_history['loss'] + finetune_history['loss'] 

validation_loss = warmup_history['val_loss'] + finetune_history['val_loss'] 



plt.figure(figsize = (8, 8))

plt.subplot(2, 1, 1)

plt.plot(training_accuracy, label = 'Training Accuracy')

plt.plot(validation_accuracy, label = 'Validation Accuracy')

plt.legend(loc = 'lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()), 1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(training_loss, label = 'Training Loss')

plt.plot(validation_loss, label = 'Validation Loss')

plt.legend(loc = 'upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0, 1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
model.load_weights("../working/blindness_detector_best_qwk.h5")



for layer in model.layers:

    layer.trainable = True

    if layer.name == 'base_model':   

        for sub_layer in layer.layers:

            sub_layer.trainable = True

            

model.save('../working/blindness_detector_best_qwk.h5')  
validation_predictions_raw = model.predict_generator(

    valid_generator,

    steps = np.ceil(float(len(x_val)) / float(BATCH_SIZE)))

validation_predictions = validation_predictions_raw > 0.5

validation_predictions = validation_predictions.astype(int).sum(axis = 1) - 1

validation_truth = y_val.sum(axis = 1) - 1
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score



def plot_confusion_matrix(cm, target_names, title = 'Confusion matrix', cmap = plt.cm.Blues):

    plt.grid(False)

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(target_names))

    plt.xticks(tick_marks, target_names, rotation = 90)

    plt.yticks(tick_marks, target_names)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



np.set_printoptions(precision = 2)

cm = confusion_matrix(validation_truth, validation_predictions)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



plot_confusion_matrix(cm = cm, target_names = diag_text)

plt.show()



print('Confusion Matrix')

print(cm)



print('Classification Report')

print(classification_report(validation_truth, validation_predictions, target_names = diag_text))



print("Validation Cohen Kappa Score: %.3f" % cohen_kappa_score(validation_predictions, validation_truth, weights = 'quadratic'))