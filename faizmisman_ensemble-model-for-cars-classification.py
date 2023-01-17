%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from tqdm import tqdm

from keras.models import Model, Sequential

from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Dropout, Lambda, Reshape, Flatten

from keras import backend as K

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

from keras.preprocessing import image

from keras.applications.vgg19 import VGG19

from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3

from keras.applications.xception import Xception

from keras.applications.inception_resnet_v2 import InceptionResNetV2

import cv2

from keras.applications.inception_v3 import preprocess_input as inception_v3_pre

from keras.applications.resnet50 import preprocess_input as resnet50_pre

from keras.applications.vgg19 import preprocess_input as vgg19_pre

from keras.applications.xception import preprocess_input as xception_pre

from keras.applications.inception_resnet_v2 import preprocess_input as InceptionResNetV2_pre

from keras.models import load_model



import matplotlib.image as mpimg

import seaborn as sns

from scipy.io import loadmat

np.random.seed(2)
# Setup Input folder link and image sizes

from pathlib import Path



devkit_path = Path('../input/cars_stanford/cars_stanford/devkit')

train_path = '../input/cars_stanford/cars_stanford/cars_train/'

test_path = '../input/cars_stanford/cars_stanford/cars_test/'

cars_meta = loadmat(devkit_path/'cars_meta.mat')

cars_train_annos = loadmat(devkit_path/'cars_train_annos.mat')

cars_test_annos = loadmat(devkit_path/'cars_test_annos_withlabels.mat')



#img_path = folder

im_width = 299

im_heigth = 299

im_chan = 3
# Retreiving Class ids and names



frame1 = [[i.flat[0] for i in line] for line in cars_meta['class_names'][0]]

columns1 = ['names']

df_train1 = pd.DataFrame(frame1, columns=columns1)

df_train1.head()

df_train1.tail()
# Setup Train set dataframe of metadata with necessary information. Since this code focus on classification, the bounding boxes are excluded.



class_id = []



frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]

columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class_id', 'fname']

df_train = pd.DataFrame(frame, columns=columns)

df_train['class_id'] = df_train['class_id']-1 # Python indexing starts on zero.

df_train['fname'] = [train_path + f for f in df_train['fname']] #  Appending Path



for i in range (len(df_train['class_id'])):

    f = df_train['class_id'][i]

    class_id.append(df_train1['names'][f])





df_train['class_name'] = [class_id][0]

df_train = df_train.drop(['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'], axis = 1) ## Deleting unnecessary infomations

df_train.head()
class_id1 = []



frame = [[i.flat[0] for i in line] for line in cars_test_annos['annotations'][0]]

columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2','class_id', 'fname']

df_test = pd.DataFrame(frame, columns=columns)

df_test['class_id'] = df_test['class_id']-1 # Python indexing starts on zero.

df_test['fname'] = [test_path + f for f in df_test['fname']] #  Appending Path



for i in range (len(df_test['class_id'])):

    f = df_test['class_id'][i]

    class_id1.append(df_train1['names'][f])





df_test['class_name'] = [class_id1][0]

df_test = df_test.drop(['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'], axis = 1) ## Deleting unnecessary infomations

df_test.head()
from keras_tqdm import TQDMCallback, TQDMNotebookCallback

from tqdm import tqdm_notebook , tnrange

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

from skimage.transform import resize
x_imgs = np.zeros((len(df_train), im_heigth, im_width, im_chan), dtype=np.uint8)

y_imgs = np.zeros((len(df_train), 1), dtype=np.uint8)



for n, id_ in tqdm_notebook(enumerate(df_train['fname']), total=len(df_train['fname'])):

    imgs = load_img(df_train['fname'][n])

    imgs = img_to_array(imgs)

    imgs = resize(imgs, (im_width, im_heigth, im_chan), mode='constant', preserve_range=True, anti_aliasing=True)

    x_imgs[n] = imgs

    

    y_imgs[n] = np.int(df_train['class_id'][n])
def get_images(length, df, imgs):



    rand= np.random.randint(0,length,4)



    fig=plt.figure(figsize=(15, 15))

    columns = 2

    rows = 2

    for i in range(1, columns*rows +1):

        r = rand[i-1]

        img = imgs[r]

        cla_name = df['class_name'][r]

        fig.add_subplot(rows, columns, i)

        plt.title(cla_name)

        plt.imshow(img)



    plt.show()
len_train = x_imgs.shape[0]

get_images(len_train, df_train, x_imgs)
x_imgs1 = np.zeros((len(df_test), im_heigth, im_width, im_chan), dtype=np.uint8)

y_imgs1 = np.zeros((len(df_test), 1), dtype=np.uint8)



for n, id_ in tqdm_notebook(enumerate(df_test['fname']), total=len(df_test['fname'])):

    imgs1 = load_img(df_test['fname'][n])

    imgs1 = img_to_array(imgs1)

    imgs1 = resize(imgs1, (im_width, im_heigth, im_chan), mode='constant', preserve_range=True, anti_aliasing=True)

    x_imgs1[n] = imgs1

    

    y_imgs1[n] = np.int(df_test['class_id'][n])
print("test set x shape :", x_imgs1.shape)

print("test set y shape :", y_imgs1.shape)
len_test = x_imgs1.shape[0]

get_images(len_test, df_test, x_imgs1)
x_valid, x_test, y_valid, y_test =  train_test_split(x_imgs1, df_test['class_id'], test_size=0.8, random_state=99)
# Checking the class frequency



freq_labels = df_train.groupby('class_name').count()[['class_id']]

freq_labels = freq_labels.rename(columns={'class': 'count'})

freq_labels = freq_labels.sort_values(by='class_id', ascending=False)

freq_labels.head(10)
freq_labels.to_csv(r'cars_count.csv', header = True)
freq_labels.head(50).plot.bar(figsize=(15,10))

plt.xticks(rotation=90);

plt.xlabel("Cars");

plt.ylabel("Count");
from collections import Counter



print('The data set is imbalanced: {}'.format(Counter(df_train['class_id'])))
from sklearn.utils.class_weight import compute_class_weight



#y_integers = np.argmax(y_train1, axis=1)

class_weights = compute_class_weight('balanced', np.unique(df_train['class_id']), df_train['class_id'])

class_weights = dict(enumerate(class_weights))
def get_features(MODEL, data, batch_size, preprocess):

    

    cnn_model = MODEL(include_top=False, input_shape=(im_width, im_heigth, im_chan), weights='imagenet')

    

    inputs = Input((im_width, im_heigth, im_chan))

    x = inputs

    x = Lambda(preprocess, name='preprocessing')(x)

    x = cnn_model(x)

    x = GlobalAveragePooling2D()(x)

    cnn_model = Model(inputs, x)



    features = cnn_model.predict(data, batch_size=batch_size, verbose=0)

    return features### InceptionV3
X_train_inception = get_features(InceptionV3, x_imgs, 32, inception_v3_pre)

X_valid_inception = get_features(InceptionV3, x_valid, 32, inception_v3_pre)

X_test_inception = get_features(InceptionV3, x_test, 32, inception_v3_pre)
#X_train_xception = get_features(Xception, x_imgs, 32, xception_pre)

#X_valid_xception = get_features(Xception, x_valid, 32, xception_pre)

#X_test_xception = get_features(Xception, x_test, 32, xception_pre)
X_train_resnet = get_features(ResNet50, x_imgs, 32, resnet50_pre)

X_valid_resnet = get_features(ResNet50, x_valid, 32, resnet50_pre)

X_test_resnet = get_features(ResNet50, x_test, 32, resnet50_pre)
#X_train_inceptionresnet = get_features(InceptionResNetV2, x_imgs, 32, InceptionResNetV2_pre)

#X_valid_inceptionresnet = get_features(InceptionResNetV2, x_valid, 32, InceptionResNetV2_pre)

#X_test_inceptionresnet = get_features(InceptionResNetV2, x_test, 32, InceptionResNetV2_pre)
#X_test_inceptionresnet.shape[0]
#X_train_vgg = get_features(VGG19, x_imgs, 32, vgg19_pre)

#X_valid_vgg = get_features(VGG19, x_valid, 32, vgg19_pre)

#X_test_vgg = get_features(VGG19, x_test, 32, vgg19_pre)
#Converting Y into one-hot-encoding



from keras.utils import to_categorical



y_train = to_categorical(df_train['class_id'])

y_test = to_categorical(y_test)

y_valid = to_categorical(y_valid)
n_class = 196



Inception_model = Sequential()

Inception_model.add(Dense(512, activation='relu', input_shape=X_train_inception.shape[1:]))

Inception_model.add(Dropout(0.2))

Inception_model.add(Dense(n_class, activation='softmax'))



Inception_model.compile(optimizer='adam',

            loss='categorical_crossentropy',

            metrics=['accuracy'],

           )



Inception_model.summary()





#Xception_model = Sequential()

#Xception_model.add(Dense(512, activation='relu', input_shape=X_train_xception.shape[1:]))

#Xception_model.add(Dropout(0.2))

#Xception_model.add(Dense(n_class, activation='softmax'))

#

#Xception_model.compile(optimizer='adam',

#            loss='categorical_crossentropy',

#            metrics=['accuracy'],

#           )



#Xception_model.summary()





#VGG_model = Sequential()

#VGG_model.add(Dense(512, activation='relu', input_shape=X_train_vgg.shape[1:]))

#VGG_model.add(Dropout(0.2))

#VGG_model.add(Dense(n_class, activation='softmax'))

#

#VGG_model.compile(optimizer='adam',

#            loss='categorical_crossentropy',

#            metrics=['accuracy'], 

#           )

#

#VGG_model.summary()



Resnet_model = Sequential()

Resnet_model.add(Dense(512, activation='relu', input_shape=X_train_resnet.shape[1:]))

Resnet_model.add(Dropout(0.2))

Resnet_model.add(Dense(n_class, activation='softmax'))



Resnet_model.compile(optimizer='adam',

            loss='categorical_crossentropy',

            metrics=['accuracy'],

           )



Resnet_model.summary()



#InceptionResnet_model = Sequential()

#InceptionResnet_model.add(Dense(512, activation='relu', input_shape=X_train_inceptionresnet.shape[1:]))

#InceptionResnet_model.add(Dropout(0.2))

#InceptionResnet_model.add(Dense(n_class, activation='softmax'))

#

#InceptionResnet_model.compile(optimizer='adam',

#            loss='categorical_crossentropy',

#            metrics=['accuracy'],

#           )

#

#InceptionResnet_model.summary()
early_stopping = EarlyStopping(monitor='val_loss', mode = 'min',patience=100, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min',factor=0.5, patience=50, min_lr=0.00001, verbose=1)



inception_callbacks=[TQDMNotebookCallback(),

                     reduce_lr,

                      ModelCheckpoint(filepath='inception.best.from_features.hdf5', 

                               verbose=0, save_best_only=True),

                     early_stopping

                     ]



#xception_callbacks=[TQDMNotebookCallback(),

#                    reduce_lr,

#                      ModelCheckpoint(filepath='saved_models/xception.best.from_features.hdf5', 

#                               verbose=0, save_best_only=True),

#                    early_stopping

#                     ]



resnet_callbacks=[TQDMNotebookCallback(),

                    reduce_lr,

                    ModelCheckpoint(filepath='resnet.best.from_features.hdf5', 

                               verbose=0, save_best_only=True),

                  early_stopping

                     ]



#vgg_callbacks=[TQDMNotebookCallback(),

#                reduce_lr,

#                ModelCheckpoint(filepath='saved_models/vgg.best.from_features.hdf5', 

#                               verbose=0, save_best_only=True),

#               early_stopping

#                     ]



#inceptionresnet_callbacks=[TQDMNotebookCallback(),

#                reduce_lr,

#                ModelCheckpoint(filepath='saved_models/inceptionresnet.best.from_features.hdf5', 

#                               verbose=0, save_best_only=True),

#               early_stopping

#                     ]
epochs = 1000 # Increase this if you want more accurate results(It is recommended to run on personal computer in this case)



inception_history = Inception_model.fit(X_train_inception, y_train, 

          validation_data=(X_valid_inception, y_valid),

          epochs=epochs, 

          callbacks=inception_callbacks,

          class_weight=class_weights,

          batch_size=32, verbose=0)



#xception_history = Xception_model.fit(X_train_xception, y_train, 

#          validation_data=(X_valid_xception, y_valid),

#          epochs=epochs,                            

#          callbacks=xception_callbacks,

#          class_weight=class_weights,

#          batch_size=32, verbose=0)



resnet_history = Resnet_model.fit(X_train_resnet, y_train, 

          validation_data=(X_valid_resnet, y_valid),

          epochs=epochs, 

          callbacks=resnet_callbacks,

          class_weight=class_weights,

          batch_size=32, verbose=0)



#vgg_history = VGG_model.fit(X_train_vgg, y_train, 

#          validation_data=(X_valid_vgg, y_valid),

#          epochs=epochs, 

#          callbacks=vgg_callbacks,

#          class_weight=class_weights,

#          batch_size=32, verbose=0)



#inceptionresnet_history = InceptionResnet_model.fit(X_train_inceptionresnet, y_train, 

#                        validation_data=(X_valid_inceptionresnet, y_valid),

#                        epochs=epochs, 

#                        callbacks=inceptionresnet_callbacks,

#                        class_weight=class_weights,

#                       batch_size=32, verbose=0)
#Retrieve some information best train accuracy, validation accuracy, train loss and validation loss

def model_evaluation(history):

    

    index_acc = np.argmax(history.history["acc"])

    acc = history.history["acc"][index_acc]

    index_val_acc = np.argmax(history.history["val_acc"])

    val_acc = history.history["val_acc"][index_val_acc]



    index_loss = np.argmin(history.history["loss"])

    losses = history.history["loss"][index_loss]

    index_val_loss = np.argmin(history.history["val_loss"])

    val_losses = history.history["val_loss"][index_val_loss]



    acc_saved = history.history["acc"][index_val_loss]

    val_acc_saved = history.history["val_acc"][index_val_loss]

    train_loss_saved = losses = history.history["loss"][index_val_loss]



    print("best train accuracy {} on epoch {} ".format(acc, index_acc+1))

    print("best validation accuracy {} on epoch {} ".format(val_acc, index_val_acc+1))

    print("lowest train loss {} on epoch {} ".format(losses, index_loss+1))

    print("lowest validation loss {} on epoch {} ".format(val_losses, index_val_loss+1))

    print("saved accuracy : {}, val accuracy : {}, and train loss {} : ".format(acc_saved, val_acc_saved, train_loss_saved))



    # Plot the loss and accuracy curves for training and validation on InceptionV3

    fig, ax = plt.subplots(2,1)

    ax[0].plot(history.history['loss'], color='b', label="Training loss")

    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

    legend = ax[0].legend(loc='best', shadow=True)



    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

    legend = ax[1].legend(loc='best', shadow=True)
#Retrieve some information best train accuracy, validation accuracy, train loss and validation loss

model_evaluation(inception_history)
#Retrieve some information best train accuracy, validation accuracy, train loss and validation loss

#model_evaluation(xception_history)
#Retrieve some information best train accuracy, validation accuracy, train loss and validation loss

model_evaluation(resnet_history)
#Retrieve some information best train accuracy, validation accuracy, train loss and validation loss

#model_evaluation(vgg_history)
#Retrieve some information best train accuracy, validation accuracy, train loss and validation loss

#model_evaluation(inceptionresnet_history)
from keras.models import load_model



inception_best = 'inception.best.from_features.hdf5'

#xception_best = 'saved_models/xception.best.from_features.hdf5'

resnet_best = 'resnet.best.from_features.hdf5'

#vgg_best = 'saved_models/vgg.best.from_features.hdf5'

#inceptionresnet_best = 'saved_models/inceptionresnet.best.from_features.hdf5'
incept = load_model(inception_best)

#xcept = load_model(xception_best)

rest = load_model(resnet_best)

#vg = load_model(vgg_best)

#incres = load_model(inceptionresnet_best)
inc = incept.predict(X_train_inception, batch_size=32, verbose=0)

#xce = xcept.predict(X_train_xception, batch_size=32, verbose=0)

res = rest.predict(X_train_resnet, batch_size=32, verbose=0)

#vgg = vg.predict(X_train_vgg, batch_size=32, verbose=0)

#incs = incres.predict(X_train_inceptionresnet, batch_size=32, verbose=0)





ensemble_train_features = np.concatenate((inc,

                                          #xce,

                                          res

                                          #vgg,

                                          #incs

                                         ), axis = 1)
inc = incept.predict(X_valid_inception, batch_size=32, verbose=0)

#xce = xcept.predict(X_valid_xception, batch_size=32, verbose=0)

res = rest.predict(X_valid_resnet, batch_size=32, verbose=0)

#vgg = vg.predict(X_valid_vgg, batch_size=32, verbose=0)

#incs = incres.predict(X_valid_inceptionresnet, batch_size=32, verbose=0)





ensemble_valid_features = np.concatenate((inc,

                                          #xce,

                                          res

                                          #vgg,

                                          #incs

                                         ), axis = 1)
inc = incept.predict(X_test_inception, batch_size=32, verbose=0)

#xce = xcept.predict(X_test_xception, batch_size=32, verbose=0)

res = rest.predict(X_test_resnet, batch_size=32, verbose=0)

#vgg = vg.predict(X_test_vgg, batch_size=32, verbose=0)

#incs = incres.predict(X_test_inceptionresnet, batch_size=32, verbose=0)





ensemble_test_features = np.concatenate((inc,

                                         #xce,

                                         res

                                         #vgg,

                                         #incs

                                        ), axis = 1)
print("Train shape: {}, validation shape: {}, test shape: {}".format(ensemble_train_features.shape, ensemble_valid_features.shape, ensemble_test_features.shape))
ensemble_model = Sequential()

ensemble_model.add(Dense(1024, activation='relu', input_shape= ensemble_train_features.shape[1:]))

ensemble_model.add(Dropout(0.5))

ensemble_model.add(Dense(512, activation='relu'))

ensemble_model.add(Dropout(0.2))

ensemble_model.add(Dense(n_class, activation='softmax'))



ensemble_model.compile(optimizer='adam',

            loss='categorical_crossentropy',

            metrics=['accuracy'],

           )



ensemble_model.summary()



ensemble_callbacks=[TQDMNotebookCallback(),

                     reduce_lr,

                      ModelCheckpoint(filepath='ensemble.best.from_features.hdf5', 

                               verbose=0, save_best_only=True),

                     early_stopping

                     ]



ensemble_history = ensemble_model.fit(ensemble_train_features, y_train, 

                        validation_data=(ensemble_valid_features, y_valid),

                        epochs=1000, 

                        callbacks=ensemble_callbacks,

                        class_weight=class_weights,

                        batch_size=32, verbose=0)
#Retrieve some information best train accuracy, validation accuracy, train loss and validation loss

model_evaluation(ensemble_history)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score
def test_model(model_name, X_test, Y_test):



    # Look at confusion matrix 



    # Predict the values from the validation dataset

    Y_pred = model_name.predict(X_test)

    # Convert predictions classes to one hot vectors 

    Y_pred_classes = np.argmax(Y_pred,axis = 1) 

    # Convert validation observations to one hot vectors

    Y_true = np.argmax(Y_test,axis = 1) 

    # compute the confusion matrix

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    # plot the confusion matrix

    ax = sns.heatmap(confusion_mtx)



    #Calculating Precision-Recall Score



    # For each class

    precision = dict()

    recall = dict()

    average_precision = dict()

    for i in range(196):

        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],

                                                        Y_pred[:, i])

        average_precision[i] = average_precision_score(Y_test[:, i], Y_pred[:, i])



    # A "micro-average": quantifying score on all classes jointly

    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),

        Y_pred.ravel())

    average_precision["micro"] = average_precision_score(Y_test, Y_pred,

                                                     average="micro")



    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))



    #Calculating ROC AUC Score



    roc_score = roc_auc_score(Y_test, Y_pred)



    print("ROC AUC Score: ", roc_score)

    

    #calculating test accuracy

    

    acc_score = accuracy_score(Y_true, Y_pred_classes)

    

    print("Test accuracy: {}".format(acc_score))
test_model(incept, X_test_inception, y_test)
#test_model(xcept, X_test_xception, y_test)
test_model(rest, X_test_resnet, y_test)
#test_model(vg, X_test_vgg, y_test)
#test_model(incres, X_test_inceptionresnet, y_test)
# Load best saved model for ensemble model



ensemble = load_model("ensemble.best.from_features.hdf5")
test_model(ensemble, ensemble_test_features, y_test)
y_ens = ensemble.predict(ensemble_test_features)
length = y_ens.shape[0]

rand= np.random.randint(0,length,20)



fig=plt.figure(figsize=(20, 20))

columns = 2

rows = 5

for i in range(1, columns*rows +1):

        r = rand[i-1]

        img = x_test[r]

        y_id = np.argmax(y_test[r])

        cla_name = df_train1['names'][y_id]

        pred_id = np.argmax(y_ens[r])

        pred_name = df_train1['names'][pred_id]

        fig.add_subplot(rows, columns, i)

        plt.title("True class: " + cla_name + ", Predicted class :" + pred_name)

        plt.imshow(img)



plt.show()