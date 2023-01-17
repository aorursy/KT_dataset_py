#Imports

import pandas as pd

import numpy as np

import tensorflow as tf

print (tf.__version__)

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import *

from tensorflow.keras.optimizers import *

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import itertools

%matplotlib inline

#Constants

BATCHSIZE=256

OUTPUT_SHAPE = 10

#Hyperparameters

DROPOUT=0.1

DROPOUT_FINAL=0.5

ROTATION=0.11

WIDTH_SHIFT=0.23

HEIGHT_SHIFT=0.19

ZOOM=0.39

SHEAR=0.11

INITIAL_LEARNING_RATE=0.001

DECAY=0.99

BASE_FILTERS=64
#Preprocessing

def normalize(array):

    return array / 256.0



def  generate_xy(df):

    #train/val

    if 'label' in df.columns:

        labels = np.array(df[["label"]],dtype=np.int32)

        df.drop('label', axis=1, inplace=True)

        ids = None

    else:

        labels = None

        ids = df["id"].values

        df.drop('id', axis=1, inplace=True)

    

    #data

    data = np.array(df,dtype=np.float32)

    data = normalize(data)

    data = np.reshape(data,(-1,28,28,1))

    return data, labels, ids
#Testing

def validate_dataset(data, expected_shape, labels=None):

    #Test shape

    assert (data.shape == expected_shape)

    #Print out some examples, labels and ids

    if labels is not None:

        print ("Label: " + str(labels[0][0])) 

    plt.imshow(data[0].reshape([28,28]), cmap='gray')



def fit_one_batch(data, model):

    train_x, train_y = data.next()

    history = model.fit(train_x, train_y,

                   epochs=100,

                   shuffle=True,

                    verbose=0)

    test_scores = model.evaluate(train_x, train_y, verbose=2, batch_size=BATCHSIZE)

    test_acc = test_scores[1]

    assert (test_acc > 0.95)
# Create datasets

train_df = pd.read_csv("../input/Kannada-MNIST/train.csv")

test_df = pd.read_csv("../input/Kannada-MNIST/test.csv")

val_df = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

train_x, train_y, _ = generate_xy(train_df)

test_x, _, ids = generate_xy(test_df)

val_x, val_y, _ = generate_xy(val_df)

generator = ImageDataGenerator(

     featurewise_center=False,

     featurewise_std_normalization=False,

     rotation_range=ROTATION,

     width_shift_range=WIDTH_SHIFT,

     height_shift_range=HEIGHT_SHIFT,

     shear_range=SHEAR,

     zoom_range=ZOOM)

train_gen = generator.flow(train_x, train_y, batch_size=BATCHSIZE)

val_gen = generator.flow(val_x, batch_size=BATCHSIZE, shuffle=False)

test_gen = generator.flow(test_x, batch_size=BATCHSIZE, shuffle=False)



#Validate

batch_x, batch_y = train_gen.next()

validate_dataset(batch_x, (BATCHSIZE,28,28,1), labels=batch_y)
#Model

def layer(inputs, features=16, stride=1, dropout=0.1, map=(3,3)):

    out = layers.Conv2D(features, map, strides=stride, padding="same")(inputs)

    out = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(out)

    out = layers.Activation('relu')(out)

    out = layers.Dropout(dropout)(out) 

    return out



def resnet_layer(inputs, features=16, stride=1, dropout=0.1, map=(3,3)):

    out = layers.Conv2D(features, map, strides=stride, padding="same")(inputs)

    out = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(out)

    out = layers.Activation('relu')(out)

    out = layers.Dropout(dropout)(out) 

    

    skip = layers.Conv2D(features, map, strides=stride, padding="same")(inputs)

    out = layers.Add()([skip, out])

    return out



def lr_decay(epoch, lr):

    return INITIAL_LEARNING_RATE * 0.99 ** epoch



def make_model(input_shape, output_shape, dropout=0.2):

    inputs = keras.Input(shape=input_shape)

    filters = BASE_FILTERS

    out = resnet_layer(inputs, features=filters, stride=1, dropout=dropout)

    out = resnet_layer(out, features=filters, stride=2, dropout=dropout)

    filters = filters * 2

    out = resnet_layer(out, features=filters, stride=1, dropout=dropout)

    out = resnet_layer(out, features=filters, stride=2, dropout=dropout)

    filters = filters * 2

    out = resnet_layer(out, features=filters, stride=1, dropout=dropout)

    out = resnet_layer(out, features=filters, stride=2, dropout=dropout)

    filters = filters * 2

    out = resnet_layer(out, features=filters, stride=1, dropout=dropout)

    

    out = layers.GlobalMaxPool2D()(out)

    out = layers.Activation('relu')(out)

    out = layers.Dropout(DROPOUT_FINAL)(out)

    out = layers.Dense(output_shape, activation='softmax')(out)

    

    model = keras.Model(inputs=inputs, outputs=out)

    opt = Adam(learning_rate=INITIAL_LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    return model



callbacks = [ModelCheckpoint('model.pth', monitor='val_acc', save_best_only=True),

             EarlyStopping(monitor='val_acc', patience=10, verbose=2, restore_best_weights=True),

             #ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5),

             LearningRateScheduler(lr_decay)

            ]
#Create and test model

img_shape = train_x.shape[1:]

model = make_model(img_shape, OUTPUT_SHAPE, dropout=DROPOUT)



#Validate model

fit_one_batch(train_gen, model) #Fails if it cannot fit on one batch



#Print model info

print(model.count_params())

keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
def tta(generator, x, y, model, num_passes=5, test=False):

    dfs = []

    

    predictions = model.predict(x, batch_size=BATCHSIZE)

    pred_core = pd.DataFrame(predictions)

    dfs.append(pred_core)

    

    for i in range(num_passes):

        predictions = model.predict_generator(generator)

        pred_new = pd.DataFrame(predictions)

        dfs.append(pred_new)

    

    result = pd.concat(dfs)

    result = result.groupby(result.index).mean()

    if not test:

        result['label_pred'] = result.idxmax('columns')

        result['label'] = y

        confusion_mtx = confusion_matrix(result['label'], result['label_pred']) 

        

        result['score'] = np.where(result['label'] == result['label_pred'], 1, 0)

        acc = result['score'].sum() / float(result['score'].size)

        print ('Acc with TTA: {}'.format(acc))

        return [acc], None, confusion_mtx



    else:

        return None, result, None
def model_average(num_models=2):

    

    all_scores = []

    final_sub = pd.DataFrame()

    for i in range(num_models):

        model = make_model(img_shape, OUTPUT_SHAPE, dropout=DROPOUT)

        history = model.fit_generator(train_gen,

                           epochs=50,

                           validation_data=(val_x, val_y),

                           shuffle=True,

                           verbose=False,

                           callbacks=callbacks)

        train_scores = model.evaluate_generator(train_gen, verbose=2)

        test_scores = model.evaluate(val_x, val_y, verbose=2, batch_size=BATCHSIZE)

        tta_scores, _, confusion_mtx = tta(val_gen, val_x, val_y, model)

        

        

        all_scores.append(test_scores+train_scores+tta_scores)

        

        

        #Gather predictions

        _, sub, _ = tta(test_gen, test_x, None, model, test=True)

        #predictions = model.predict(test_x, batch_size=BATCHSIZE)

        #sub = pd.DataFrame(predictions)

        sub['id'] = ids

        if final_sub.empty:

            final_sub = sub

        else:

            fina_sub = pd.concat([final_sub, sub])

    

    plot_confusion_matrix(confusion_mtx, classes = range(10))

    scores = pd.DataFrame(all_scores, columns=['loss_val', 'acc_val', 'loss_train', 'acc_train', 'acc_val_tta'])

    return scores, sub
#Evaluate and generate submission.csv

scores, sub = model_average(5)

print(scores.describe())

sub = sub.groupby('id').mean()

sub['label'] = sub.idxmax('columns')

sub[['label']].to_csv('submission.csv', index=True)