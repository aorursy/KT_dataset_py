!pip install efficientnet
import pandas as pd

import os

import time

t_start = time.time()



data_dir = '../regular-deepdrid/Regular_DeepDRiD'

#The address here is very important: regular_train  regular-test

train_data = '../input/regular-deepdrid/Regular_DeepDRiD/regular_train/'

valid_data = '../input/regular-deepdrid/Regular_DeepDRiD/regular_valid/'

test_data = '../input/regular-deepdrid/Regular_DeepDRiD/regular-test/'



train_df = pd.read_csv('../input/regular-deepdrid/DR_label/DR_label/regular-fundus-training.csv')

valid_df = pd.read_csv('../input/regular-deepdrid/DR_label/DR_label/regular-fundus-validation.csv')

test_df = pd.read_csv('../input/regular-deepdrid/DR_label/DR_label/Challenge1_upload.csv')



train_df['image_id'] = train_df['image_id'] + ".jpg"# Two meathods add jpg

valid_df['image_id'] = valid_df['image_id'] + ".jpg"# Two meathods add jpg

test_df["image_id"] = test_df["image_id"].apply(lambda x: x + ".jpg")





#add diagnosis：https://www.cnblogs.com/guxh/p/9420610.html

#df_left.insert(2, 'diagnosis', 0)

train_df['diagnosis']=None

for i in range(len(train_df)):

    if 'r' in train_df['image_id'][i]:

        train_df['diagnosis'][i]=train_df['right_eye_DR_Level'][i].astype('int')

    else:

        train_df['diagnosis'][i]=train_df['left_eye_DR_Level'][i].astype('int')

        





#df_left.insert(2, 'diagnosis', 0)

valid_df['diagnosis']=None



for i in range(len(valid_df)):

    if 'r' in train_df['image_id'][i]:

        valid_df['diagnosis'][i]=valid_df['right_eye_DR_Level'][i].astype('int')

    else:

        valid_df['diagnosis'][i]=valid_df['left_eye_DR_Level'][i].astype('int')



#valid_df['image_id'] = valid_df['image_id'] + ".jpg"

display(train_df.head())

display(valid_df.head())

display(test_df.head())



print('Number of train samples: ', train_df.shape[0])

print('Number of valid samples: ', valid_df.shape[0])

print('Number of test samples: ', test_df.shape[0])
import datetime

starttime = datetime.datetime.now()



import os

import sys

import cv2

import shutil

import random

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import multiprocessing as mp

import matplotlib.pyplot as plt



from keras.activations import elu



from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from keras import backend as K

from keras.models import Model

from keras.utils import to_categorical

from keras import optimizers, applications

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler





from sklearn.metrics import classification_report

from imgaug import augmenters as iaa







def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed) 

seed = 2020

seed_everything(seed)
# Model parameters



HEIGHT = 224

WIDTH = 224

CHANNELS = 3

TTA_STEPS = 5

BATCH_SIZE=32

def crop_image_from_gray(img, tol=7):

    """

    Applies masks to the orignal image and 

    returns the a preprocessed image with 

    3 channels

    

    :param img: A NumPy Array that will be cropped

    :param tol: The tolerance used for masking

    

    :return: A NumPy array containing the cropped image

    """

    # If for some reason we only have two channels

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    # If we have a normal RGB images

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def preprocess_image(image, sigmaX=10):

    """

    The whole preprocessing pipeline:

    1. Read in image

    2. Apply masks

    3. Resize image to desired size

    4. Add Gaussian noise to increase Robustness

    

    :param img: A NumPy Array that will be cropped

    :param sigmaX: Value used for add GaussianBlur to the image

    

    :return: A NumPy array containing the preprocessed image

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (WIDTH, HEIGHT))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)

    return image
import cv2

import math

def gamma_trans(img, gamma):  # gamma函数处理

    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表

    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数

    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

 

# 彩色图像进行自适应直方图均衡化，代码同上的地方不再添加注释

def hisEqulColor2(img):

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    channels = cv2.split(ycrcb)

 

    # 以下代码详细注释见官网：

    # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    clahe.apply(channels[0],channels[0])

 

    cv2.merge(channels,ycrcb)

    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)

    return img







def preprocess_image_gamma(image):

    res2=hisEqulColor2(image)

    img_gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

    mean = np.mean(img_gray)

    gamma_val = math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma

    image_gamma_correct = gamma_trans( res2, gamma_val)   # gamma变换



    image_gamma_correct=cv2.cvtColor(image_gamma_correct, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image_gamma_correct, (WIDTH, HEIGHT))

   

    return image
image1=cv2.imread('../input/regular-deepdrid/Regular_DeepDRiD/regular_train/101_l2.jpg')

res2=hisEqulColor2(image1)

img_gray = cv2.cvtColor( res2, cv2.COLOR_BGR2GRAY)

mean = np.mean(img_gray)

gamma_val = math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma

image_gamma_correct = gamma_trans( res2, gamma_val)   # gamma变换

image_gamma_correct=cv2.cvtColor(image_gamma_correct, cv2.COLOR_BGR2RGB)



image = cv2.resize(image_gamma_correct, (WIDTH, HEIGHT))

plt.imshow(image)
image2=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)



plt.imshow(image2)
image1=cv2.imread('../input/regular-deepdrid/Regular_DeepDRiD/regular_train/101_l2.jpg')

res2=hisEqulColor2(image1)

img_gray = cv2.cvtColor( res2, cv2.COLOR_BGR2GRAY)

mean = np.mean(img_gray)

gamma_val = math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma

image_gamma_correct = gamma_trans( res2, gamma_val)   # gamma变换

image = cv2.resize(image_gamma_correct, (WIDTH, HEIGHT))

plt.imshow(image)


# Add Image augmentation to our generator

datagen = ImageDataGenerator(rotation_range=360,

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                   #validation_split=0.15,

                                   preprocessing_function=preprocess_image_gamma, 

                                   rescale=1 / 128.)





train_generator=datagen.flow_from_dataframe(

                        dataframe=train_df,

                        directory=train_data,

                        x_col="image_id",

                        y_col="diagnosis",

                        class_mode="raw",

                        batch_size=BATCH_SIZE,

                        target_size=(HEIGHT, WIDTH),

                       

                        seed=seed)



valid_generator=datagen.flow_from_dataframe(

                        dataframe=valid_df,

                        directory=valid_data,

                        x_col="image_id",

                        y_col="diagnosis",

                        class_mode="raw",

                        batch_size=BATCH_SIZE,

                        target_size=(HEIGHT, WIDTH),

                       

                        seed=seed)



test_generator=datagen.flow_from_dataframe(  

                       dataframe=test_df,

                       directory=test_data,

                       x_col="image_id",

                       batch_size=1,

                       class_mode=None,

                       shuffle=False,

                       target_size=(HEIGHT, WIDTH),

                       seed=seed)
from keras.preprocessing import image

x,y=train_generator.next()

for i in range(0,4):

    image=x[i]

    label=y[i]

    print(label)

    plt.imshow(image)

    plt.show()
import efficientnet.keras as efn 



def create_model(input_shape):

    input_tensor = Input(shape=input_shape)

    base_model = efn.EfficientNetB0(weights='imagenet', 

                                include_top=False,

                                input_tensor=input_tensor)

    #base_model.load_weights('../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')





    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)



    x = Dense(5, activation=elu)(x)



    final_output = Dense(1, activation='linear', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model



model = create_model(input_shape=(HEIGHT, WIDTH, CHANNELS))

#model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.00005), loss='mse', metrics=['mse', 'acc'])

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)





rlr = ReduceLROnPlateau(monitor='val_loss', 

                        factor=0.5, 

                        patience=4, 

                        verbose=1, 

                        mode='auto', 

                        epsilon=0.0001)
# Begin training

history=model.fit_generator(train_generator,

                    steps_per_epoch=train_generator.samples // BATCH_SIZE,

                    epochs=30,

                    validation_data=valid_generator,

                    validation_steps = valid_generator.samples // BATCH_SIZE,

                    callbacks=[ es, rlr])
# Visualize mse

history_df = pd.DataFrame(model.history.history)

history_df[['loss', 'val_loss']].plot(figsize=(12,5))

plt.title("Loss (MSE)", fontsize=16, weight='bold')

plt.xlabel("Epoch")

plt.ylabel("Loss (MSE)")

history_df[['acc', 'val_acc']].plot(figsize=(12,5))

plt.title("Accuracy", fontsize=16, weight='bold')

plt.xlabel("Epoch")

plt.ylabel("% Accuracy");
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
# Create empty arays to keep the predictions and labels

df_preds = pd.DataFrame(columns=['label', 'pred', 'set'])

train_generator.reset()

valid_generator.reset()



# Add train predictions and labels

for i in range(STEP_SIZE_TRAIN + 1):

    im, lbl = next(train_generator)

    preds = model.predict(im, batch_size=train_generator.batch_size)

    for index in range(len(preds)):

        df_preds.loc[len(df_preds)] = [lbl[index], preds[index][0], 'train']



# Add validation predictions and labels

for i in range(STEP_SIZE_VALID + 1):

    im, lbl = next(valid_generator)

    preds = model.predict(im, batch_size=valid_generator.batch_size)

    for index in range(len(preds)):

        df_preds.loc[len(df_preds)] = [lbl[index], preds[index][0], 'validation']



df_preds['label'] = df_preds['label'].astype('int')



def classify(x):

    if x < 0.5:

        return 0

    elif x < 1.5:

        return 1

    elif x < 2.5:

        return 2

    elif x < 3.5:

        return 3

    return 4



# Classify predictions

df_preds['predictions'] = df_preds['pred'].apply(lambda x: classify(x))



train_preds = df_preds[df_preds['set'] == 'train']

validation_preds = df_preds[df_preds['set'] == 'validation']



labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']

def plot_confusion_matrix(train, validation, labels=labels):

    train_labels, train_preds = train

    validation_labels, validation_preds = validation

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(24, 7))

    train_cnf_matrix = confusion_matrix(train_labels, train_preds)

    validation_cnf_matrix = confusion_matrix(validation_labels, validation_preds)



    train_cnf_matrix_norm = train_cnf_matrix.astype('float') / train_cnf_matrix.sum(axis=1)[:, np.newaxis]

    validation_cnf_matrix_norm = validation_cnf_matrix.astype('float') / validation_cnf_matrix.sum(axis=1)[:, np.newaxis]



    train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=labels, columns=labels)

    validation_df_cm = pd.DataFrame(validation_cnf_matrix_norm, index=labels, columns=labels)



    sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues",ax=ax1).set_title('Train')

    sns.heatmap(validation_df_cm, annot=True, fmt='.2f', cmap=sns.cubehelix_palette(8),ax=ax2).set_title('Validation')

    plt.show()



plot_confusion_matrix((train_preds['label'], train_preds['predictions']), (validation_preds['label'], validation_preds['predictions']))



def evaluate_model(train, validation):

    train_labels, train_preds = train

    validation_labels, validation_preds = validation

    print("Train        Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds, train_labels, weights='quadratic'))

    print("Validation   Cohen Kappa score: %.3f" % cohen_kappa_score(validation_preds, validation_labels, weights='quadratic'))

    print("Complete set Cohen Kappa score: %.3f" % cohen_kappa_score(np.append(train_preds, validation_preds), np.append(train_labels, validation_labels), weights='quadratic'))

    

evaluate_model((train_preds['label'], train_preds['predictions']), (validation_preds['label'], validation_preds['predictions']))
from sklearn.metrics import classification_report

print(classification_report(validation_preds['label'], validation_preds['predictions'],digits=4))

def apply_tta(model, generator, steps=1):

    step_size = generator.n//generator.batch_size

    preds_tta = []

    for i in range(steps):

        generator.reset()

        preds = model.predict_generator(generator, steps=step_size)

        preds_tta.append(preds)



    return np.mean(preds_tta, axis=0)



preds = apply_tta(model, test_generator, TTA_STEPS)

predictions = [classify(x) for x in preds]
results = pd.DataFrame({'image_id':test_df['image_id'], 'DR_level':predictions})

results['image_id'] = results['image_id'].map(lambda x: str(x)[:-4])

results.to_csv('submission_ultra.csv', index=False)

display(results.head())
# Distribution of predictions

results['DR_level'].value_counts().sort_index().plot(kind="bar", 

                                                      figsize=(12,5), 

                                                      rot=0)

plt.title("Label Distribution (Predictions)", 

          weight='bold', 

          fontsize=18)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel("Label", fontsize=17)

plt.ylabel("Frequency", fontsize=17);
model.save_weights('../working/efficientnetb0-preprocess.h5')



# Check kernels run-time. GPU limit for this competition is set to ± 9 hours.

t_finish = time.time()

total_time = round((t_finish-t_start) / 3600, 4)

print('Kernel runtime = {} hours ({} minutes)'.format(total_time, 

                                                      int(total_time*60)))