# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

panda_path = '/kaggle/input/isic-2019/ISIC_2019_Training_GroundTruth.csv'

image_dir = '/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/'
ground_truth = pd.read_csv(panda_path)

ground_truth.drop('UNK', inplace=True, axis=1)
for index, img in enumerate(ground_truth.image):

    img = img+'.jpg'

    ground_truth.image[index]=img
#val_truth.head()
ground_truth.head()
ground_truth.tail()
val_rows = (np.random.rand(25330//5)*25330).astype(int)
val_rows
val_dataframe = ground_truth.iloc[val_rows]
val_dataframe.head()
val_dataframe.tail()
sums = []

for col in ground_truth.columns[1:]:

    sums.append(sum(val_dataframe[col]))
sums_full = []

for col in ground_truth.columns[1:]:

    sums_full.append(sum(ground_truth[col]))
#normalize

def avg(l):

    return sum(l)/len(l)

full_avg = avg(sums_full)

val_avg = avg(sums)

sums = [val/val_avg for val in sums]

sums_full = [val/full_avg for val in sums_full]
print(sums)

print(sums_full)
#create class weights

max_val = max(sums)

class_weights = {}

for ind, val in enumerate(sums, start=0):

    class_weights[ind] = max_val/sums[ind]

class_weights
ground_truth.drop(val_rows, inplace=True, axis=0)
print(f'There are {len(ground_truth)} many images in the train set')

print(f'There are {len(val_dataframe)} many images in the train set')
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.layers import Dense, Flatten, Dropout

import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications import vgg16

num_classes = ground_truth.columns.size

num_classes -= 1
def vgg(num_classes, trainable=False):

    vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    for layer in vgg_model.layers:

        layer.trainable = trainable



    x = Flatten()(vgg_model.output)

    x2 = Dropout(0.5)(x)

    prediction = Dense(num_classes, activation='softmax')(x2)

    full_model = Model(inputs=vgg_model.input, outputs=prediction)

    return full_model
vgg_model = vgg(num_classes, trainable=True)
vgg_model.summary()
model = Sequential()

model.add(ResNet50(weights='imagenet', input_shape=(224, 224, 3)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
layer_count=0

for layer in model.layers[0].layers:

    if layer_count > 165:

        layer.trainable = True

    else:

        layer.trainable = False
labels = ground_truth.columns

labels = labels[1:]

data_gen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)

train_gen = data_gen.flow_from_dataframe(dataframe=ground_truth, directory=image_dir, x_col='image', target_size=(224, 224),

                                     y_col=labels, class_mode='raw', batch_size=32)

val_gen = data_gen.flow_from_dataframe(dataframe=val_dataframe, directory=image_dir, 

                                       x_col='image', y_col=labels, class_mode='raw', batch_size=32, target_size=(224, 224))
def compute_class_freqs(labels):

    

    # total number of patients (rows)

    N = labels.shape[0]

    

    positive_frequencies = np.mean(labels, axis=0)

    negative_frequencies = 1 - positive_frequencies



    ### END CODE HERE ###

    return positive_frequencies, negative_frequencies
freq_pos, freq_neg = compute_class_freqs(train_gen.labels)

freq_pos
import seaborn as sns

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})

data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)

plt.xticks(rotation=90)

f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):

    

    

    def weighted_loss(y_true, y_pred):

       

        loss = 0.0

        



        for i in range(len(pos_weights)):

            # for each class, add average weighted loss for that class 

            loss += K.mean(-(pos_weights[i]*y_true[:, i]*K.log(y_pred[:, i]+epsilon)

                             + neg_weights[i]*(1-y_true[:, i])*K.log((1-y_pred[:, i])+epsilon)))

        return loss

    

    return weighted_loss
#focal loss function --> focusses on training mainly the hard examples, gives lower weights to easy examples

# def focal_loss(y_true, y_pred):

#     gamma = 2.0

#     alpha = 0.25

#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def focal_loss(gamma=2., alpha=4.):



    gamma = float(gamma)

    alpha = float(alpha)



    def focal_loss_fixed(y_true, y_pred):

        """Focal loss for multi-classification

        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)

        Notice: y_pred is probability after softmax

        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper

        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)

        Focal Loss for Dense Object Detection

        https://arxiv.org/abs/1708.02002



        Arguments:

            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]

            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]



        Keyword Arguments:

            gamma {float} -- (default: {2.0})

            alpha {float} -- (default: {4.0})



        Returns:

            [tensor] -- loss.

        """

        epsilon = 1.e-9

        y_true = tf.convert_to_tensor(y_true, tf.float32)

        y_pred = tf.convert_to_tensor(y_pred, tf.float32)



        model_out = tf.add(y_pred, epsilon)

        ce = tf.multiply(y_true, -K.log(model_out))

        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))

        fl = tf.multiply(alpha, tf.multiply(weight, ce))

        reduced_fl = tf.reduce_max(fl, axis=1)

        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed
from tensorflow.keras.metrics import AUC

from keras import backend as K

pos_weights = freq_neg

neg_weights = freq_pos

metrics = ['accuracy']

#vgg_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=get_weighted_loss(pos_weights, neg_weights), metrics=metrics)

vgg_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=focal_loss(), metrics=metrics)

print('model has compiled')
from tensorflow.keras.callbacks import EarlyStopping



callback = EarlyStopping(monitor='val_accuracy', patience=3)
history = vgg_model.fit(train_gen, steps_per_epoch=20747//32, 

                        epochs=20, validation_data=val_gen, validation_steps=5066//32)#, class_weight=class_weights)
reset_ind_val = val_dataframe.reset_index().drop(['index'], axis=1)

reset_ind_val.head()
reset_ind_val.iloc[0][0]

print(np.argmax(reset_ind_val.iloc[0][1:].values))
plt.plot(history.history['loss'])

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_loss'])

plt.plot(history.history['val_accuracy'])

plt.legend(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plt.show()
def test_image(index, model, dataframe=reset_ind_val, img_dir=image_dir, labels=labels):

    row = dataframe.iloc[index]

    path = os.path.join(img_dir, row[0])

    img = load_img(path, target_size=(224, 224))

    img_arr = img_to_array(img)

    image = np.expand_dims(img_arr, axis=0)

    image = preprocess_input(image)

    preds = model.predict(image)

    if np.argmax(row[1:].values) == np.argmax(preds):

        return True, labels[np.argmax(preds)]

    return False, labels[np.argmax(preds)]
reset_ind_val.head()
#copied from Coursera util package

from keras.preprocessing import image

from sklearn.metrics import roc_auc_score, roc_curve

from tensorflow.compat.v1.logging import INFO, set_verbosity

import cv2



def get_roc_curve(labels, predicted_vals, generator):

    auc_roc_vals = []

    for i in range(len(labels)):

        try:

            gt = generator.labels[:, i]

            pred = predicted_vals[:, i]

            auc_roc = roc_auc_score(gt, pred)

            auc_roc_vals.append(auc_roc)

            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)

            plt.figure(1, figsize=(10, 10))

            plt.plot([0, 1], [0, 1], 'k--')

            plt.plot(fpr_rf, tpr_rf,

                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")

            plt.xlabel('False positive rate')

            plt.ylabel('True positive rate')

            plt.title('ROC curve')

            plt.legend(loc='best')

        except:

            print(

                f"Error in generating ROC curve for {labels[i]}. "

                f"Dataset lacks enough examples."

            )

    plt.show()

    return auc_roc_vals
preds = vgg_model.predict_generator(val_gen)
preds.shape
class_nums = []

for arr in preds:

    class_nums.append(np.argmax(arr))

class_nums
val_gen.labels[:, 1]
auc_rocs = get_roc_curve(labels, preds, val_gen)

vgg_model.save_weights('third_try.h5')