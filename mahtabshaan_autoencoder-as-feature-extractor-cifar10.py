import warnings

warnings.filterwarnings('ignore')



import pickle

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import sys



import keras

from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, BatchNormalization

from keras.layers import Conv1D, MaxPool1D, CuDNNLSTM, Reshape

from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate

from keras.datasets import cifar10

from keras import regularizers

from keras.models import Model, Sequential

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.optimizers import SGD, Adam, RMSprop, Adadelta

import keras.backend as K

from keras.objectives import mean_squared_error

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils



from sklearn.utils import class_weight

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, RobustScaler, StandardScaler



from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
dict = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}
x_test_extra = []

y_test_extra = []

x_train_final = []

y_train_final = []

count = [0, 0, 0]

for i, j in zip(x_train, y_train):

    if (j==2):

        if(count[0]<2000):

            x_test_extra.append(i)

            y_test_extra.append(j)

            count[0]+=1

        else:

            x_train_final.append(i)

            y_train_final.append(j)

    elif (j==4):

        if(count[1]<2000):

            x_test_extra.append(i)

            y_test_extra.append(j)

            count[1]+=1

        else:

            x_train_final.append(i)

            y_train_final.append(j)

    elif (j==9):

        if(count[2]<2000):

            x_test_extra.append(i)

            y_test_extra.append(j)

            count[2]+=1

        else:

            x_train_final.append(i)

            y_train_final.append(j)

    else:

        x_train_final.append(i)

        y_train_final.append(j)

        

x_test_extra = np.array(x_test_extra)

y_test_extra = np.array(y_test_extra)

x_train_final = np.array(x_train_final)

y_train_final = np.array(y_train_final)
x_test_final = np.append(x_test_extra, x_test, axis=0)

y_test_final = np.append(y_test_extra, y_test, axis=0)
#x_train_final = x_train    ## These code were used to check model performances with balanced dataset.

#x_test_final = x_test

#y_train_final = y_train

#y_test_final = y_test

x_train_final = x_train_final.astype('float32')

x_test_final = x_test_final.astype('float32')

x_train_final = x_train_final / 255

x_test_final = x_test_final / 255
from sklearn.model_selection import train_test_split



# Split the data

x_train, x_valid, y_trainf, y_validf = train_test_split(x_train_final, y_train_final, test_size=0.2, random_state=42, shuffle= True)
y_train = keras.utils.to_categorical(y_trainf, 10)

y_valid = keras.utils.to_categorical(y_validf, 10)

y_test_one_hot = keras.utils.to_categorical(y_test_final, 10)
def create_block(input, chs): ## Convolution block of 2 layers

    x = input

    for i in range(2):

        x = Conv2D(chs, 3, padding="same")(x)

        x = Activation("relu")(x)

        x = BatchNormalization()(x)

    return x



##############################



## Here, I compute the class weights for using in different models. 

## This is to order our model to emphasize more on classes with less training data.

class_weights = class_weight.compute_class_weight(

               'balanced',

                np.unique(y_trainf), 

                y_trainf.reshape(y_trainf.shape[0]))



class_weights



##############################



def showOrigDec(orig, dec, num=10):  ## function used for visualizing original and reconstructed images of the autoencoder model

    n = num

    plt.figure(figsize=(20, 4))



    for i in range(n):

        # display original

        ax = plt.subplot(2, n, i+1)

        plt.imshow(orig[300*i].reshape(32, 32, 3))

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)



        # display reconstruction

        ax = plt.subplot(2, n, i +1 + n)

        plt.imshow(dec[300*i].reshape(32, 32, 3))

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    plt.show()

        

def show_test(m, d):  ## function used for visualizing the predicted and true labels of test data

    plt.figure(figsize =(40,8))

    for i in range(5):

        ax = plt.subplot(1, 5, i+1)

        test_image = np.expand_dims(d[1810*i+5], axis=0)

        test_result = m.predict(test_image)

        plt.imshow(x_test_final[1810*i+5])

        index = np.argsort(test_result[0,:])

        plt.title("Pred:{}, True:{}".format(dict[index[9]], dict[y_test_final[1810*i+5][0]]))

    plt.show()

    

def show_test2(m, d):  ## function used for visualizing the predicted and true labels of test data

    plt.figure(figsize =(40,8))

    for i in range(5):

        ax = plt.subplot(1, 5, i+1)

        test_image = np.expand_dims(d[1810*i+5], axis=0)

        test_result = m.predict(test_image)[1]

        plt.imshow(x_test_final[1810*i+5])

        index = np.argsort(test_result[0,:])

        plt.title("Pred:{}, True:{}".format(dict[index[9]], dict[y_test_final[1810*i+5][0]]))

    plt.show()

    

def report(predictions): ## function used for creating a classification report and confusion matrix

    cm=confusion_matrix(y_test_one_hot.argmax(axis=1), predictions.argmax(axis=1))

    print("Classification Report:\n")

    cr=classification_report(y_test_one_hot.argmax(axis=1),

                                predictions.argmax(axis=1), 

                                target_names=list(dict.values()))

    print(cr)

    plt.figure(figsize=(12,12))

    sns.heatmap(cm, annot=True, xticklabels = list(dict.values()), yticklabels = list(dict.values()), fmt="d")

    

def loss_function(y_true, y_pred):  ## loss function for using in autoencoder models

    mses = mean_squared_error(y_true, y_pred)

    return K.sum(mses, axis=(1,2))
def full_conv():

    input = Input((32,32,3))

    block1 = create_block(input, 32)

    x = MaxPool2D(2)(block1)

    #x = Dropout(0.2)(x)

    block2 = create_block(x, 64)

    x = MaxPool2D(2)(block2)

    #x = Dropout(0.3)(x)

    block3 = create_block(x, 128)

    #x = MaxPool2D(2)(block3)

    x = Dropout(0.4)(block3)

    x = Flatten()(x)

    output = Dense(10, activation='softmax')(x)

    return Model(input, output)



conv_model = full_conv()

conv_model.summary()
#training

batch_size = 512

epochs=50

opt_rms = Adadelta()

conv_model.compile(loss='categorical_crossentropy',

                   optimizer=opt_rms,

                   metrics=['accuracy'])
def run_conv_model(data_aug):

    er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

    lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)

    callbacks = [er, lr]

    

    if not data_aug:

        history = conv_model.fit(x_train, y_train, batch_size=512,

                                 epochs=epochs,

                                 verbose=1, callbacks=callbacks,

                                 validation_data=(x_valid,y_valid),

                                 class_weight=class_weights)

    else:

        train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

        train_set_ae = train_datagen.flow(x_train, y_train, batch_size=512)



        validation_datagen = ImageDataGenerator()

        validation_set_ae = validation_datagen.flow(x_valid, y_valid, batch_size=512)

        

        history = conv_model.fit_generator(train_set_ae,

                                           epochs=epochs,

                                           steps_per_epoch=np.ceil(x_train.shape[0]/512),

                                           verbose=1, callbacks=callbacks,

                                           validation_data=(validation_set_ae),

                                           validation_steps=np.ceil(x_valid.shape[0]/512),

                                           class_weight=class_weights)

        

        return history
run_conv_model(1)
print('Test accuracy for benchmark model= {}'.format(conv_model.evaluate(x_test_final, y_test_one_hot)[1]))
show_test(conv_model, x_test_final)
predictions = conv_model.predict(x_test_final)

report(predictions)
def unet():  ## I commented several layers of the model for descreasing model complexity as the results were almost same

    input = Input((32,32,3))

    

    # Encoder

    block1 = create_block(input, 32)

    x = MaxPool2D(2)(block1)

    block2 = create_block(x, 64)

    x = MaxPool2D(2)(block2)

    #block3 = create_block(x, 64)

    #x = MaxPool2D(2)(block3)

    #block4 = create_block(x, 128)

    

    # Middle

    #x = MaxPool2D(2)(block2)

    middle = create_block(x, 128)

    

    # Decoder

    #x = Conv2DTranspose(128, kernel_size=2, strides=2)(middle)

    #x = Concatenate()([block4, x])

    #x = create_block(x, 128)

    #x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)

    #x = Concatenate()([block3, x])

    #x = create_block(x, 64)

    x = Conv2DTranspose(64, kernel_size=2, strides=2)(middle)

    x = Concatenate()([block2, x])

    x = create_block(x, 64)

    x = Conv2DTranspose(32, kernel_size=2, strides=2)(x)

    x = Concatenate()([block1, x])

    x = create_block(x, 32)

    

    # output

    x = Conv2D(3, 1)(x)

    output = Activation("sigmoid")(x)

    

    return Model(input, middle), Model(input, output)



def general_ae():

    input = Input((32,32,3))

    

    # Encoder

    block1 = create_block(input, 32)

    x = MaxPool2D(2)(block1)

    block2 = create_block(x, 64)

    x = MaxPool2D(2)(block2)

    

    #Middle

    middle = create_block(x, 128)

    

    # Decoder

    up1 = UpSampling2D((2,2))(middle)

    block3 = create_block(up1, 64)

    #up1 = UpSampling2D((2,2))(block3)

    up2 = UpSampling2D((2,2))(block3)

    block4 = create_block(up2, 32)

    #up2 = UpSampling2D((2,2))(block4)

    

    # output

    x = Conv2D(3, 1)(up2)

    output = Activation("sigmoid")(x)

    return Model(input, middle), Model(input, output)
def run_ae(m):  ## function for choosing unet/general autoencoder

    if m=='unet':

        encoder, model = unet()

    elif m=='ae':

        encoder, model = general_ae()

        

    return encoder, model
encoder_unet, model_unet = run_ae('unet')

model_unet.compile(SGD(1e-3, 0.9), loss=loss_function)

model_unet.summary()
er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)

callbacks = [er, lr]

history = model_unet.fit(x_train, x_train, 

                         batch_size=512,

                         epochs=100,

                         verbose=1,

                         validation_data=(x_valid, x_valid),

                         shuffle=True, callbacks=callbacks,

                         class_weight=class_weights)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
recon_test_unet = model_unet.predict(x_test_final)

recon_valid_unet = model_unet.predict(x_valid)
showOrigDec(x_valid, recon_valid_unet)
showOrigDec(x_test_final, recon_test_unet)
encoder_ae, model_ae = run_ae('ae')

model_ae.compile(SGD(1e-3, 0.9), loss=loss_function)

model_ae.summary()
er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)

callbacks = [er, lr]

history = model_ae.fit(x_train, x_train, 

                       batch_size=512,

                       epochs=100,

                       verbose=1,

                       validation_data=(x_valid, x_valid),

                       shuffle=True, callbacks=callbacks,

                       class_weight=class_weights)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
recon_test_ae = model_ae.predict(x_test_final)

recon_valid_ae = model_ae.predict(x_valid)
showOrigDec(x_valid, recon_valid_ae)
showOrigDec(x_test_final, recon_test_ae)
gist_train_unet = encoder_unet.predict(x_train)

gist_valid_unet = encoder_unet.predict(x_valid)

gist_test_unet = encoder_unet.predict(x_test_final)



gist_train_ae = encoder_ae.predict(x_train)

gist_valid_ae = encoder_ae.predict(x_valid)

gist_test_ae = encoder_ae.predict(x_test_final)
def classifier_dense(inp):

    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))

    #x = MaxPool2D()(input)

    x = Flatten()(input)

    #x = BatchNormalization()(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.64)(x)

    x = Dense(50, activation='relu')(x)

    #x = Reshape((-1, 1))(x)

    #x = Conv1D(128, (3,), activation='relu', padding='same')(x)

    #x = MaxPool1D()(x)

    #x = CuDNNLSTM(64)(x)

    #x = Flatten()(x)

    x = Dropout(0.4)(x)

    output = Dense(10, activation='softmax')(x)

    return Model(input, output)



def classifier_conv(inp):

    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))

    x = Conv2D(1024, 3, padding="same")(input)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = MaxPool2D(2)(x)

    x = Dropout(0.5)(x)

    x = Conv2D(128, 3, padding="same")(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = MaxPool2D(2)(x)

    x = Dropout(0.5)(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.35)(x)

    x = Dense(100, activation='relu')(x)

    x = Dropout(0.69)(x)

    output = Dense(10, activation='softmax')(x)

    return Model(input, output)
def run_cls(m, inp):  ## function for choosing dense/convolutional classifier model

    if m=='dense':

        classifier = classifier_dense(inp)

    elif m=='conv':

        classifier = classifier_conv(inp)

        

    return classifier
decoder_ae_conv = run_cls('conv', gist_train_ae)

decoder_ae_conv.compile(loss='categorical_crossentropy',

                        optimizer=Adadelta(),

                        metrics=['accuracy'])

decoder_ae_conv.summary()
er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)

callbacks = [er, lr]

hist1 = decoder_ae_conv.fit(gist_train_ae, y_train, batch_size=512, epochs=100, 

                            validation_data = (gist_valid_ae, y_valid),

                            shuffle=True, callbacks=callbacks,

                            class_weight=class_weights)
plt.plot(hist1.history['acc'])

plt.plot(hist1.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
print('Test accuracy for AE_conv model= {}'.format(decoder_ae_conv.evaluate(gist_test_ae, y_test_one_hot)[1]))
show_test(decoder_ae_conv, gist_test_ae)
predictions = decoder_ae_conv.predict(gist_test_ae)

report(predictions)
decoder_ae_dense = run_cls('dense', gist_train_ae)

decoder_ae_dense.compile(loss='categorical_crossentropy',

                         optimizer=Adadelta(),

                         metrics=['accuracy'])

decoder_ae_dense.summary()
er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)

callbacks = [er, lr]

hist1 = decoder_ae_dense.fit(gist_train_ae, y_train, batch_size=512, epochs=100, 

                             validation_data = (gist_valid_ae, y_valid),

                             shuffle=True, callbacks=callbacks,

                             class_weight=class_weights)
plt.plot(hist1.history['acc'])

plt.plot(hist1.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
print('Test accuracy for AE_dense model= {}'.format(decoder_ae_dense.evaluate(gist_test_ae, y_test_one_hot)[1]))
show_test(decoder_ae_dense, gist_test_ae)
predictions = decoder_ae_dense.predict(gist_test_ae)

report(predictions)
decoder_un_conv = run_cls('conv', gist_train_unet)

decoder_un_conv.compile(loss='categorical_crossentropy',

                         optimizer=Adadelta(),

                         metrics=['accuracy'])

decoder_un_conv.summary()
er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)

callbacks = [er, lr]

hist1 = decoder_un_conv.fit(gist_train_unet, y_train, batch_size=512, epochs=100, 

                            validation_data = (gist_valid_unet, y_valid),

                            shuffle=True, callbacks=callbacks,

                            class_weight=class_weights)
plt.plot(hist1.history['acc'])

plt.plot(hist1.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
print('Test accuracy for Unet_conv model= {}'.format(decoder_un_conv.evaluate(gist_test_unet, y_test_one_hot)[1]))
show_test(decoder_un_conv, gist_test_unet)
predictions = decoder_un_conv.predict(gist_test_unet)

report(predictions)
decoder_un_dense = run_cls('dense', gist_train_unet)

decoder_un_dense.compile(loss='categorical_crossentropy',

                         optimizer=Adadelta(),

                         metrics=['accuracy'])

decoder_un_dense.summary()
er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)

callbacks = [er, lr]

hist1 = decoder_un_dense.fit(gist_train_unet, y_train, batch_size=512, epochs=100, 

                             validation_data = (gist_valid_unet, y_valid),

                             shuffle=True, callbacks=callbacks,

                             class_weight=class_weights)
plt.plot(hist1.history['acc'])

plt.plot(hist1.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
print('Test accuracy for Unet_dense model= {}'.format(decoder_un_dense.evaluate(gist_test_unet, y_test_one_hot)[1]))
show_test(decoder_un_dense, gist_test_unet)
predictions = decoder_un_dense.predict(gist_test_unet)

report(predictions)
def end_to_end():  ## I commented several layers of the model for descreasing model complexity as the results were almost same

    input = Input((32,32,3))

    

    # Encoder

    block1 = create_block(input, 32)

    x = MaxPool2D(2)(block1)

    block2 = create_block(x, 64)

    x = MaxPool2D(2)(block2)

    #block3 = create_block(x, 64)

    #x = MaxPool2D(2)(block3)

    #block4 = create_block(x, 128)

    

    # Middle

    #x = MaxPool2D(2)(block2)

    middle = create_block(x, 128)

    

    # Decoder

    #x = Conv2DTranspose(128, kernel_size=2, strides=2)(middle)

    #x = Concatenate()([block4, x])

    #x = create_block(x, 128)

    #x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)

    #x = Concatenate()([block3, x])

    #x = create_block(x, 64)

    x = Conv2DTranspose(64, kernel_size=2, strides=2)(middle)

    x = Concatenate()([block2, x])

    x = create_block(x, 64)

    x = Conv2DTranspose(32, kernel_size=2, strides=2)(x)

    x = Concatenate()([block1, x])

    x = create_block(x, 32)

    

    # reconstruction

    x = Conv2D(3, 1)(x)

    recon = Activation("sigmoid", name='autoencoder')(x)

    

    #classification 

    c = Conv2D(1024, 3, padding="same")(middle)

    c = Activation('relu')(c)

    c = BatchNormalization()(c)

    c = MaxPool2D(2)(c)

    c = Dropout(0.5)(c)

    c = Conv2D(128, 3, padding="same")(c)

    c = Activation('relu')(c)

    c = BatchNormalization()(c)

    c = MaxPool2D(2)(c)

    c = Dropout(0.4)(c)

    c = Flatten()(c)

    c = Dense(512, activation='relu')(c)

    c = Dropout(0.35)(c)

    c = Dense(100, activation='relu')(c)

    c = Dropout(0.69)(c)

    classify = Dense(10, activation='softmax', name='classification')(c)

    

    outputs = [recon, classify]

    

    return Model(input, outputs)

multimodel = end_to_end()

multimodel.compile(loss = {'classification': 'categorical_crossentropy', 'autoencoder': loss_function}, 

                  loss_weights = {'classification': 0.9, 'autoencoder': 0.1}, 

                  optimizer = SGD(lr= 0.01, momentum= 0.9),

                  metrics = {'classification': ['accuracy'], 'autoencoder': []})
er = EarlyStopping(monitor='val_classification_acc', patience=10, restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_classification_acc', factor=0.2, patience=5, min_delta=0.0001)

callbacks = [er, lr]

hist_mul = multimodel.fit(x_train, [x_train,y_train], batch_size=512, epochs=100, 

                          validation_data = (x_valid, [x_valid,y_valid]),

                          shuffle=True, callbacks=callbacks)

#                           class_weight=class_weights
recon_test_e2e = multimodel.predict(x_test_final)[0]

recon_valid_e2e = multimodel.predict(x_valid)[0]
showOrigDec(x_valid, recon_valid_e2e)
showOrigDec(x_test_final, recon_test_e2e)
predictions = multimodel.predict(x_test_final)[1]

report(predictions)
show_test2(multimodel, x_test_final)
# def solvers(func):

#     scaler_classifier = MinMaxScaler(feature_range=(0.0, 1.0))

#     pipe = Pipeline(steps=[("scaler_classifier", scaler_classifier),

#                            ("classifier", func)])



#     pipe = pipe.fit(gist_train.reshape(gist_train.shape[0], -1), y_trainf)

#     acc = pipe.score(gist_test.reshape(gist_test.shape[0], -1), y_test_final)

#     predict = pipe.predict(gist_test.reshape(gist_test.shape[0], -1))

    

#     return acc, predict
# lr = LogisticRegression(C=5e-1, random_state=666, solver='lbfgs', multi_class='multinomial')

# rf = RandomForestClassifier(random_state=666)

# knn = KNeighborsClassifier()

# svc = svm.SVC()
# acc_lr, pred_lr = solvers(lr)

# acc_lr
# acc_rf, pred_rf = solvers(rf)

# acc_rf
# acc_knn, pred_knn = solvers(knn)

# acc_knn
# acc_svc, pred_svc = solvers(svc)

# acc_svc
# space = {

#             'units1': hp.choice('units1', [256,512,1024]),

#             'units2': hp.choice('units2', [128,256,512]),

#             'units4': hp.choice('units4', [256,512,1024]),

#             'units5': hp.choice('units5', [50,64,100,128]),

#             'dropout1': hp.uniform('dropout1', .25,.75),

#             'dropout2': hp.uniform('dropout2', .25,.75),

#             'batch_size' : hp.choice('batch_size', [64,128,256,512]),

         

#             'nb_epochs' :  200,

#             'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),

#             'activation': 'relu'

#         }
# def f_nn(params):   

#     from keras.models import Sequential

#     from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, BatchNormalization

#     from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate

#     from keras.optimizers import Adadelta, Adam, rmsprop

#     import sys



#     print ('Params testing: ', params)

#     model = Sequential()

#     model.add(Conv2D(params['units1'], 3, padding="same", activation="relu"))

#     model.add(BatchNormalization())

#     model.add(MaxPool2D())

#     model.add(Conv2D(params['units2'], 3, padding="same", activation="relu"))

#     model.add(BatchNormalization())

#     model.add(MaxPool2D())   



#     model.add(Flatten())

#     model.add(Dense(output_dim=params['units4'], activation="relu"))

#     model.add(Dropout(params['dropout1']))

#     model.add(Dense(output_dim=params['units5'], activation="relu"))

#     model.add(Dropout(params['dropout2']))

#     model.add(Dense(10))

#     model.add(Activation('softmax'))

#     model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])



#     model.fit(gist_train, y_train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)



#     acc = model.evaluate(gist_valid, y_valid)[1]

#     print('Accuracy:', acc)

#     sys.stdout.flush() 

#     return {'loss': -acc, 'status': STATUS_OK}





# trials = Trials()

# best = fmin(f_nn, space, algo=tpe.suggest, max_evals=5, trials=trials)

# print('best: ')

# print(best)