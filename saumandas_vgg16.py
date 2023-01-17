root_path = '../input/isic-2019'
import os

import numpy as np

import pandas as pd



IMAGE_DIR = os.path.join(root_path, 'ISIC_2019_Training_Input/ISIC_2019_Training_Input')

panda_path = os.path.join(root_path, 'ISIC_2019_Training_GroundTruth.csv')
print(len(os.listdir(IMAGE_DIR)))
print(f'This is the image dir: {IMAGE_DIR}')

print(f'This is the csv filepath: {panda_path}')
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Activation

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
def preprocess(df):

  for index, img in enumerate(df.image):

    img = img+'.jpg'

    df.image[index]=img

  df.drop(['UNK'], axis=1, inplace=True)
def train_val_test_split(df, test_len=1000, val_ratio=0.2):

  test_rows = (np.random.rand(1000)*df.shape[0]).astype(int)

  test_df =  df.iloc[test_rows]

  test_df = test_df.reset_index().drop(['index'], axis=1)

  df.drop(test_rows, axis=0, inplace=True)

  df = df.reset_index().drop(['index'], axis=1)

  val_rows = (np.random.rand(int(val_ratio*df.shape[0]))*df.shape[0]).astype(int)

  val_df = df.iloc[val_rows]

  df.drop(val_rows, axis=0, inplace=True)

  test_df = test_df.reset_index().drop(['index'], axis=1)

  df = df.reset_index().drop(['index'], axis=1)

  return df, val_df, test_df

full_df = pd.read_csv(panda_path)

preprocess(full_df)

train_df, val_df, test_df = train_val_test_split(full_df)

labels=list(train_df.columns[1:])

print(labels)

train_df.head()
def basic_vgg(input_shape=(224, 224, 3), num_classes=8):

  new_input = Input(shape=input_shape)

  model = VGG16(weights=None, input_tensor=new_input, classes=num_classes)

  model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

  return model
base_model = basic_vgg()

base_model.summary()
#sequential API

def vgg_model(input_shape=(224, 224, 3)):

  model = Sequential()

  model.add(VGG16(include_top=False, weights='imagenet', input_shape=input_shape))

  

  model.add(GlobalAveragePooling2D())

  #model.add(Flatten())



  model.add(Dense(512, activation='relu'))

  model.add(Dropout(0.25))



  model.add(Dense(1024))

  model.add(BatchNormalization())

  model.add(Activation('relu')) 

  model.add(Dropout(0.5))



  model.add(Dense(8, activation='sigmoid'))

  model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

  print('Model has compiled')

  return model
vgg16_model = vgg_model(input_shape=(224, 224, 3))
vgg16_model.summary()
def get_train_gen(df, img_path=IMAGE_DIR, target_size=(224, 224)):

  data_gen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                horizontal_flip=True,

                                width_shift_range=0.2,

                                height_shift_range=0.2)

  return data_gen.flow_from_dataframe(dataframe=df, directory=img_path, 

                                      x_col='image', y_col=list(df.columns)[1:],

                                      batch_size=64, shuffle=True, class_mode='raw', 

                                      target_size=target_size)



def get_val_test_gen(val_df, test_df, img_path=IMAGE_DIR, target_size=(224, 224)):

  data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

  val = data_gen.flow_from_dataframe(dataframe=val_df, directory=img_path, 

                                      x_col='image', y_col=list(val_df.columns)[1:],

                                      batch_size=64, shuffle=True, class_mode='raw', 

                                      target_size=target_size)

  test = data_gen.flow_from_dataframe(dataframe=test_df, directory=img_path, 

                                      x_col='image', batch_size=1, shuffle=True, class_mode=None, 

                                      target_size=target_size)

  return val, test
train_generator = get_train_gen(train_df)

valid_generator, test_generator = get_val_test_gen(val_df, test_df)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

history = vgg16_model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator,

                        validation_steps=STEP_SIZE_VALID, epochs=10)
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
preds = vgg16_model.predict_generator(valid_generator)
import matplotlib.pyplot as plt

auc_rocs = get_roc_curve(labels, preds, valid_generator)

#From ONODERA Notebook

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, fontsize=25)

    plt.yticks(tick_marks, fontsize=25)

    plt.xlabel('Predicted label',fontsize=25)

    plt.ylabel('True label', fontsize=25)

    plt.title(title, fontsize=30)

    

    divider = make_axes_locatable(ax)

    cax = divider.append_axes('right', size="5%", pad=0.15)

    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)

    cbar.ax.tick_params(labelsize=20)

    

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

#            title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    fontsize=20,

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
plot_confusion_matrix(valid_generator.labels, preds, labels, title='Skin Disease Confusion Matrix')