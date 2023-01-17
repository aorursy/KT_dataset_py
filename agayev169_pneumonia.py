import numpy as np 

import pandas as pd

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

import scipy



import os
IMG_SIZE = 128
!ls ../input/chest_xray/chest_xray/test/NORMAL | wc -l

!ls ../input/chest_xray/chest_xray/train/NORMAL | wc -l

!ls ../input/chest_xray/chest_xray/val/NORMAL | wc -l



!ls ../input/chest_xray/chest_xray/test/PNEUMONIA | wc -l

!ls ../input/chest_xray/chest_xray/train/PNEUMONIA | wc -l

!ls ../input/chest_xray/chest_xray/val/PNEUMONIA | wc -l
train_path = "../input/chest_xray/chest_xray/train/"

test_path  = "../input/chest_xray/chest_xray/test/"

val_path   = "../input/chest_xray/chest_xray/val/"



train_n = 1341 + 3875

test_n  = 234 + 390

val_n   = 8 + 8
def display_img(path):

    img = cv2.imread(path)

    

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)

#     plt.axis("off")
display_img("/kaggle/input/chest_xray/chest_xray/train/PNEUMONIA/person480_virus_982.jpeg")
%%time

train_x = np.empty((train_n + val_n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

train_y = np.zeros((train_n + val_n), dtype=np.uint8)



train_normal_n = 0



i = 0

j = train_n + val_n - 1

for dirname, _, filenames in os.walk(train_path):

    for filename in tqdm(filenames):

        if "DS" in filename:

            continue

        img = cv2.imread(os.path.join(dirname, filename))

        if "NORMAL" in dirname:

            train_x[i] = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            train_y[i] = 0

            i += 1

        else:

            train_x[j] = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            train_y[j] = 1

            j -= 1

#         i += 1

        

for dirname, _, filenames in os.walk(val_path):

    for filename in tqdm(filenames):

        if "DS" in filename:

            continue

        img = cv2.imread(os.path.join(dirname, filename))

        if "NORMAL" in dirname:

            train_x[i] = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            train_y[i] = 0

            i += 1

        else:

            train_x[j] = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            train_y[j] = 1

            j -= 1

#         i += 1



train_normal_n = i
%%time

test_x = np.empty((test_n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

test_y = np.zeros((test_n), dtype=np.uint8)



i = 0

for dirname, _, filenames in os.walk(test_path):

    for filename in tqdm(filenames):

        if "DS" in filename:

            continue

        img = cv2.imread(os.path.join(dirname, filename))

        test_x[i] = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if "NORMAL" in dirname:

            test_y[i] = 0

        else:

            test_y[i] = 1

        i += 1
# train_pneumonia = np.sum(train_y)

# train_normal    = len(train_y) - train_pneumonia



# train_pneumonia = int(np.min([1.5 * np.min([train_normal, train_pneumonia]), train_pneumonia]))

# train_normal    = int(np.min([1.5 * np.min([train_normal, train_pneumonia]), train_normal]))



# print(train_normal, train_pneumonia)
val_x = np.empty((600, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

val_y = np.zeros((600), dtype=np.uint8)

IMG_SIZE

val_x[:300] = train_x[:300]

val_y[:300] = train_y[:300]



val_x[300:] = train_x[-300:]

val_y[300:] = train_y[-300:]



train_x = train_x[300:-300]

train_y = train_y[300:-300]
train_x = train_x[:-1000]

train_y = train_y[:-1000]
# pneumonia_begin = np.argmax(train_y)



# indexes = np.random.permutation(pneumonia_begin)

# np.random.shuffle(indexes)

# indexes = indexes[:train_normal]



# train_norm_x = train_x[indexes]

# train_norm_y = train_y[indexes]
# # train_x = train_x[pneumonia_begin:]

# # train_y = train_y[pneumonia_begin:]



# indexes = np.random.permutation(len(train_x) - len(train_norm_x))

# np.random.shuffle(indexes)

# indexes = indexes[:train_pneumonia]



# train_pn_x = train_x[indexes + np.argmax(train_y)]

# train_pn_y = train_y[indexes + np.argmax(train_y)]



# print(np.sum(train_pn_y), len(train_pn_y))
# train_x = np.concatenate([train_norm_x, train_pn_x])

# train_y = np.concatenate([train_norm_y, train_pn_y])
# indexes = np.random.permutation(len(train_x))

# train_x = train_x[indexes]

# train_y = train_y[indexes]



# val_beg = int(0.7 * len(train_x))

# val_x = train_x[val_beg:]

# val_y = train_y[val_beg:]

# train_x = train_x[:val_beg]

# train_y = train_y[:val_beg]
# train_x = train_x.reshape((-1, 224, 224, 1))

# val_x = val_x.reshape((-1, 224, 224, 1))

# test_x = test_x.reshape((-1, 224, 224, 1))



train_y = train_y.reshape((-1, 1))

val_y = val_y.reshape((-1, 1))

test_y = test_y.reshape((-1, 1))
print(np.sum(train_y), "of pneumonias and", int(len(train_y) - np.sum(train_y)), "of normals in train set")

print(np.sum(val_y), "of pneumonias and", int(len(val_y) - np.sum(val_y)), "of normals in validation set")

print(np.sum(test_y), "of pneumonias and", int(len(test_y) - np.sum(test_y)), "of normals in test set")
train_x = train_x.astype(np.float32)

train_x /= np.max(train_x)



val_x = val_x.astype(np.float32)

val_x /= np.max(val_x)



test_x = test_x.astype(np.float32)

test_x /= np.max(test_x)
def to_multilabel(x):

    res = np.empty((len(x), 2))

    for i in range(len(x)):

        if x[i][0] == 0:

            res[i] = [1, 0]

        else:

            res[i] = [1, 1]

    return res
train_y = to_multilabel(train_y)

val_y = to_multilabel(val_y)

test_y = to_multilabel(test_y)
# indexes_norm = np.arange(train_normal_n)

# indexes_perm = np.random.permutation(train_normal_n)



# dim = np.concatenate([[train_normal_n] ,train_x.shape[1:]]).ravel()



# train_ext_x = 0.5 * train_x[indexes_norm] + 0.5 * train_x[indexes_perm] + (np.random.rand(dim[0], dim[1], dim[2], dim[3]) * 0.1 - 0.05)

# train_ext_x = (train_ext_x - np.min(train_ext_x)) / (np.max(train_ext_x) - np.min(train_ext_x))



# train_ext_y = 0.5 * train_y[indexes_norm] + 0.5 * train_y[indexes_perm]



# del indexes_norm

# del indexes_perm
# train_x = np.concatenate([train_ext_x, train_x])

# train_y = np.concatenate([train_ext_y, train_y])



# del train_ext_x

# del train_ext_y
# train_y = train_y.astype(np.uint8)

# val_y   = val_y.astype(np.uint8)

# test_y  = test_y.astype(np.uint8)
# print(train_x.dtype, train_x.shape, np.min(train_x), np.max(train_x))

# print(val_x.dtype, val_x.shape, np.min(val_x), np.max(val_x))

# print(test_x.dtype, test_x.shape, np.min(test_x), np.max(test_x))



# print(train_y.dtype, train_y.shape)

# print(val_y.dtype, val_y.shape)

# print(test_y.dtype, test_y.shape)
# print(np.sum(train_y, axis=0)[1], "of pneumonias and", np.sum(train_y, axis=0)[0] - np.sum(train_y, axis=0)[1], "of normals in train set")

# print(np.sum(val_y, axis=0)[1], "of pneumonias and", np.sum(val_y, axis=0)[0] - np.sum(val_y, axis=0)[1], "of normals in validation set")

# print(np.sum(test_y, axis=0)[1], "of pneumonias and", np.sum(test_y, axis=0)[0] - np.sum(test_y, axis=0)[1], "of normals in test set")
fig, ax = plt.subplots(5, 5, figsize=(25, 25))



indexes = np.random.randint(0, len(train_x), 25)

for i in range(5):

    for j in range(5):

        ax[i][j].imshow(train_x[indexes[i * 5 + j]])

        ax[i][j].set_title(f"{indexes[i * 5 + j]} - {train_y[indexes[i * 5 + j]]}")

        ax[i][j].axis("off")
fig, ax = plt.subplots(5, 5, figsize=(25, 25))



indexes = np.random.randint(0, len(val_x), 25)

for i in range(5):

    for j in range(5):

        ax[i][j].imshow(val_x[indexes[i * 5 + j]])

        ax[i][j].set_title(f"{indexes[i * 5 + j]} - {val_y[indexes[i * 5 + j]]}")

        ax[i][j].axis("off")
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, SeparableConv2D, MaxPooling2D, Dropout, BatchNormalization, Input, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.keras.applications.densenet import DenseNet201

from tensorflow.keras.utils import Sequence



import imgaug as ia

import imgaug.augmenters as iaa
class Generator(Sequence):

    def __init__(self, train_x, train_y, batch_size=32, shuffle=True, mixup=True, augment=True):

        self.train_x = train_x

        self.train_y = train_y

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.mixup_ = mixup

        self.augment = augment

        

        if augment:

            sometimes = lambda aug: iaa.Sometimes(0.5, aug)

            self.seq = iaa.Sequential([

                sometimes(iaa.CropAndPad(

                    percent=(-0.05, 0.1),

                    pad_mode=ia.ALL,

                    pad_cval=(0, 255)

                )),

                sometimes(iaa.Affine(

                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis

    #                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)

                    rotate=(-20, 20), # rotate by -20 to +20 degrees

    #                 shear=(-10, 10), # shear by -16 to +16 degrees

                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                    cval=(0, 0), # if mode is constant, use a cval between 0 and 255

                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

                )),

                iaa.Fliplr(0.5),

                iaa.GaussianBlur(sigma=(0, 3.0))

            ])

        

        self.on_epoch_end()

        

    def __len__(self):

        return int(np.floor(len(self.train_x) / self.batch_size))

    

    def __getitem__(self, index):

        x = self.train_x[index * self.batch_size : (index + 1) * self.batch_size]

        if self.augment:

            x = (x * 255).astype(np.uint8)

            x = self.seq.augment_images(x)

        x = x.astype(np.float32) / np.max(x)

        y = self.train_y[index * self.batch_size : (index + 1) * self.batch_size]

        

        if self.mixup_:

            x, y = self.mixup(x, y)

        return x, y

    

    def on_epoch_end(self):

        if self.shuffle:

            indexes = np.random.permutation(len(self.train_x)).astype(np.uint16)

            self.train_x = self.train_x[indexes]

            self.train_y = self.train_y[indexes]

        

    def mixup(self, x, y):

        alphas = np.random.ranf(len(x))

        x_alphas = alphas.reshape((-1, 1, 1, 1))

        y_alphas = alphas.reshape((-1, 1))

        indexes = np.random.permutation(len(x))

        np.random.shuffle(indexes)

        x = x_alphas * x[indexes] + (1.0 - x_alphas) * x

        y = y_alphas * y[indexes] + (1.0 - y_alphas) * y

        return x, y

        

    def get_batch(self, index):

        return self.__getitem__(index)
BATCH_SIZE = 32

EPOCHS = 50
gen = Generator(train_x, train_y, batch_size=BATCH_SIZE, mixup=True, augment=True)
imgs, labels = gen.get_batch(0)

print(imgs.dtype, imgs.shape, np.min(imgs), np.max(imgs))



fig, ax = plt.subplots(5, 5, figsize=(25, 25))

fig.suptitle("Augmentations", fontsize=36)

for i in range(5):

    for j in range(5):

        ax[i][j].imshow(imgs[i * 5 + j])

        ax[i][j].set_title(labels[i * 5 + j])

        ax[i][j].axis("off")
class FScore(tf.keras.callbacks.Callback):

    def __init__(self, val_x, val_y):

        self.val_x = val_x

        self.val_y = val_y

        

        self.precisions = []

        self.recalls = []

        self.fscores = []

    

    def on_train_begin(self, logs={}):

        pass



    def on_epoch_end(self, epoch, logs={}):

        precision, recall, fscore = self.score()

        if np.isnan(precision):

            precision = 0

        if np.isnan(recall):

            recall = 0

        if np.isnan(fscore):

            fscore = 0

        

        self.precisions.append(precision)

        self.recalls.append(recall)

        self.fscores.append(fscore)

        

        print(f"precision: {precision}, recall: {recall}, fscore: {fscore}")

        

        if np.argmax(np.array(self.fscores)) == len(self.fscores) - 1:

            print(f"Best fscore to date. Saving the model in './weights_fscore.hdf5'")

            model.save("weights_fscore.hdf5")

            

            

    def score(self, x=[], y=[], threshold=0.5):

        if len(x) == 0:

            x = self.val_x

        if len(y) == 0:

            y = self.val_y

            

        y = (np.sum(y, axis=1) - 1).reshape((-1, 1))

        y = y.clip(0, 1)

        y = y.astype(np.uint8)

        

        y_pred = self.model.predict(x)

        y_pred[y_pred < threshold] = 0

        y_pred[y_pred != 0]        = 1

        y_pred = (np.sum(y_pred, axis=1) - 1).reshape((-1, 1))

        y_pred = y_pred.clip(0, 1)

        y_pred = y_pred.astype(np.uint8)

        

        cm = confusion_matrix(y, y_pred)

        tn, fp, fn, tp = cm.ravel()



        precision = tp / (tp + fp)

        recall    = tp / (tp + fn)

        fscore    = 2 * (precision * recall) / (precision + recall)

        return precision, recall, fscore

            

        

    def get_scores(self):

        return (self.precisions, self.recalls, self.fscores)
# model = Sequential()



# model.add(Input(batch_shape=(None, 224, 224, 3)))

# model.add(BatchNormalization())



# model.add(Conv2D(64, activation='relu', kernel_size=(5, 5)))

# model.add(Conv2D(64, activation='relu', kernel_size=(5, 5)))

# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

# model.add(Dropout(0.5))

# model.add(BatchNormalization())



base_model = Xception(include_top=False, weights='imagenet',

                      input_shape=(IMG_SIZE, IMG_SIZE, 3), 

                      pooling=None)



model = Sequential()

model.add(base_model)



model.add(GlobalAveragePooling2D())



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Dense(2, activation='sigmoid'))



opt = Adam(learning_rate=0.00005)

model.compile(optimizer=opt, loss='mean_squared_error')
model.summary()
es = EarlyStopping(monitor='val_loss', 

                   patience=10, 

                   verbose=1, 

                   mode='min')

mc = ModelCheckpoint("weights_loss.hdf5", 

                     monitor="val_loss",

                     save_weights_only=True,

                     save_best_only=True, 

                     verbose=1)

rl = ReduceLROnPlateau(monitor='val_loss', 

                       factor=0.5, 

                       patience=5, 

                       verbose=1, 

                       mode='min', 

                       cooldown=1)

cl = CSVLogger("training.log")

fs = FScore(val_x, val_y)



history = model.fit_generator(gen, validation_data=(val_x, val_y), use_multiprocessing=False,

                              epochs=50, callbacks=[es, mc, rl, cl, fs])
plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.legend(["val_loss", "loss"])
precisions, recalls, fscores = fs.get_scores()
plt.plot(precisions)

# plt.plot(history.history['precision'])

# plt.legend(["val_precision", "precision"])
plt.plot(recalls)

# plt.plot(history.history['recall'])

# plt.legend(["val_recall", "recall"])
plt.plot(fscores)

# plt.plot(history.history['fscore'])

# plt.legend(["val_fscore", "fscore"])
def compute_score_inv(threshold):

    _, _, score = fs.score(val_x, val_y, threshold)

    return 1 - score



simplex = scipy.optimize.minimize(

    compute_score_inv, 0.5#, method='nelder-mead'

)



best_threshold = simplex['x'][0]



print(best_threshold)
for dirname, _, filenames in os.walk('.'):

    for filename in filenames:

        if ".hdf5" in filename:

            try:

                model.load_weights(filename)

                pred = model.predict(val_x)

                pred[pred > 0.5] = 1

                pred[pred != 1] = 0

                pred = (np.sum(pred, axis=1) - 1).reshape((-1, 1))

                pred = pred.clip(0, 1)

                cm = confusion_matrix((np.sum(val_y, axis=1)).reshape((-1, 1)) - 1, pred)

                plt.figure()

                plot_confusion_matrix(cm,figsize=(6,4), hide_ticks=True, cmap=plt.cm.Blues)

                plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

                plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

                plt.show()





                tn, fp, fn, tp = cm.ravel()



                precision = tp / (tp + fp)

                recall    = tp / (tp + fn)

                fscore    = 2 * precision * recall / (precision + recall)



                print(filename)

                print("precision: " + str(precision))

                print("recall   : " + str(recall))

                print("fscore   : " + str(fscore))

            except:

                pass
for dirname, _, filenames in os.walk('.'):

    for filename in filenames:

        if ".hdf5" in filename:

            try:

                model.load_weights(filename)

                pred = model.predict(test_x)

                pred[pred > 0.5] = 1

                pred[pred != 1] = 0

                pred = (np.sum(pred, axis=1) - 1).reshape((-1, 1))

                pred = pred.clip(0, 1)

                cm = confusion_matrix((np.sum(test_y, axis=1)).reshape((-1, 1)) - 1, pred)

                plt.figure()

                plot_confusion_matrix(cm,figsize=(6,4), hide_ticks=True, cmap=plt.cm.Blues)

                plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

                plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

                plt.show()





                tn, fp, fn, tp = cm.ravel()



                precision = tp / (tp + fp)

                recall    = tp / (tp + fn)

                fscore    = 2 * precision * recall / (precision + recall)



                print(filename)

                print("precision: " + str(precision))

                print("recall   : " + str(recall))

                print("fscore   : " + str(fscore))

            except:

                pass