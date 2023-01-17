!pip install tensorflow-addons
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split

from tensorflow.keras.metrics import Metric

import tensorflow.keras.backend as K

import tensorflow_addons as tfa
!pip install efficientnet

import efficientnet.tfkeras as efn
train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

print(train_df.shape)

train_df.head()
train_images_path = '../input/panda-resized-train-data-512x512/train_images/train_images/'
plt.imshow(plt.imread(train_images_path + train_df.loc[0]['image_id'] + '.png'))
train_df["image_path"] = train_df["image_id"].apply(lambda x: x + '.png')
xtrain, xval, ytrain, yval = train_test_split(train_df["image_path"], train_df["isup_grade"], test_size = 0.15, stratify = train_df["isup_grade"])



df_train = pd.DataFrame({"image_path":xtrain, "isup_grade":ytrain})

df_val = pd.DataFrame({"image_path":xval, "isup_grade":yval})



df_train["isup_grade"] = df_train["isup_grade"].astype('str')

df_val["isup_grade"] = df_val["isup_grade"].astype('str')
print(df_train.shape) 

print(df_val.shape)
BATCH_SIZE = 4

img_size = 512

EPOCHS = 12

nb_classes = 6
LR_START = 0.00001

LR_MAX = 0.0001 * 8

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 3

LR_SUSTAIN_EPOCHS = 1

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def get_model():

    base_model =  efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))

    x = base_model.output

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.4)(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.3)(x)

    predictions = Dense(nb_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=predictions)
model = get_model()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', tfa.metrics.CohenKappa(num_classes = nb_classes, weightage = 'quadratic')])
train_datagen = ImageDataGenerator(

        rescale = 1./255,          # for normalising the image

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True)  # randomly flip images



valid_datagen = ImageDataGenerator(rescale = 1./255 )
train_generator = train_datagen.flow_from_dataframe(dataframe = df_train,

                                               directory = train_images_path,

                                               x_col = "image_path",

                                               y_col = "isup_grade",

                                               batch_size = BATCH_SIZE,

                                               target_size =  (img_size, img_size),

                                               class_mode = 'categorical')



validation_generator = valid_datagen.flow_from_dataframe(dataframe = df_val,

                                                    directory = train_images_path,

                                                    x_col = "image_path",

                                                    y_col = "isup_grade",

                                                    batch_size = BATCH_SIZE, 

                                                    target_size = (img_size, img_size),

                                                    class_mode = 'categorical')
# %%time

history = model.fit_generator(

            generator = train_generator, 

            steps_per_epoch = (df_train.shape[0] // BATCH_SIZE),

            epochs=EPOCHS,

            validation_data = validation_generator, 

            validation_steps = (df_val.shape[0] // BATCH_SIZE),

            callbacks=[lr_callback]

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



cohens_kappa = history.history['cohen_kappa']

val_cohens_kappa = history.history['val_cohen_kappa']



epochs = range(len(acc))

 

plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.figure()

 

plt.plot(epochs, cohens_kappa, 'b', label='Training Cohen-kappa')

plt.plot(epochs, val_cohens_kappa, 'r', label='Validation Cohen-kappa')

plt.title('Cohen Kappa - Training and validation score')

plt.legend()



plt.show()
model.save('model.h5')