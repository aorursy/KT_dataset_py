import os

import numpy as np

import pandas as pd

from glob import glob

from itertools import chain

from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#tf.debugging.set_log_device_placement(True)
DATA_DIR = '../input/data/'

image_size = 256

batch_size = 32
df = pd.read_csv(f'{DATA_DIR}Data_Entry_2017.csv')
data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(DATA_DIR, 'images*', '*', '*.png'))}
df['path'] = df['Image Index'].map(data_image_paths.get)
df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))

labels = [x for x in labels if len(x) > 0]
labels
for label in labels:

    if len(label) > 1:

        df[label] = df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
df.head()
labels = [label for label in labels if df[label].sum() > 1000]
labels
train_df, valid_df = train_test_split(df, test_size=0.20, random_state=2020, stratify=df['Finding Labels'].map(lambda x: x[:4]))
train_df['Finding Labels']
train_df.loc[:, 'labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

valid_df.loc[:, 'labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
train_df['labels']
core_idg = ImageDataGenerator(rescale=1 / 255,

                                  samplewise_center=True,

                                  samplewise_std_normalization=True,

                                  horizontal_flip=True,

                                  vertical_flip=False,

                                  height_shift_range=0.05,

                                  width_shift_range=0.1,

                                  rotation_range=5,

                                  shear_range=0.1,

                                  fill_mode='nearest',

                                  zoom_range=0.15)



train_gen = core_idg.flow_from_dataframe(dataframe=train_df,

                                             directory=None,

                                             x_col='path',

                                             y_col='labels',

                                             class_mode='categorical',

                                             batch_size=batch_size,

                                             classes=labels,

                                             target_size=(image_size, image_size))



test_gen = core_idg.flow_from_dataframe(dataframe=valid_df,

                                             directory=None,

                                             x_col='path',

                                             y_col='labels',

                                             class_mode='categorical',

                                             batch_size=batch_size,

                                             classes=labels,

                                             target_size=(image_size, image_size))



test_X, test_Y = next(core_idg.flow_from_dataframe(dataframe=valid_df,

                                                       directory=None,

                                                       x_col='path',

                                                       y_col='labels',

                                                       class_mode='categorical',

                                                       batch_size=1024,

                                                       classes=labels,

                                                       target_size=(image_size, image_size)))
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2



# base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

# x = base_model.output

# x = tf.keras.layers.GlobalAveragePooling2D()(x)

# output = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)

# model = tf.keras.Model(base_model.input, output)

# model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
from tensorflow.keras.applications import DenseNet201

rnet = DenseNet201(

        input_shape=(256, 256, 3),

        weights='imagenet',

        include_top=False

    )

# trainable rnet

rnet.trainable = True

model = tf.keras.Sequential([

    rnet,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(len(labels), activation='sigmoid',dtype='float32')])



model.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy'])
# Learning rate schedule for TPU, GPU and CPU.

# Using an LR ramp up because fine-tuning a pre-trained model.

# Starting with a high LR would break the pre-trained weights.



LR_START = 0.00001

LR_MAX = 0.00005

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)



rng = [i for i in range(25)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def get_callbacks(model_name):

    callbacks = []

    checkpoint = tf.keras.callbacks.ModelCheckpoint(

        filepath=f'model.{model_name}.h5',

        verbose=1,

        save_best_only=True)

    erly = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    callbacks.append(checkpoint)

    callbacks.append(erly)

    callbacks.append(lr_callback)

    return callbacks
with tf.device('/GPU:0'):

    callbacks = get_callbacks('DenseNet201')

    history = model.fit(train_gen,

              steps_per_epoch=200,

              validation_data=(test_X, test_Y),

              epochs=30,

              callbacks=callbacks)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left') 

plt.show()

# summarize history for loss 

plt.plot(history.history['loss']) 

plt.plot(history.history['val_loss']) 

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(test_X)
for label, p_count, t_count in zip(labels,

                                     100 * np.mean(y_pred, 0),

                                     100 * np.mean(test_Y, 0)):

    print('%s: actual: %2.2f%%, predicted: %2.2f%%' % (label, t_count, p_count))
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))

for (idx, c_label) in enumerate(labels):

    fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), y_pred[:, idx])

    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('trained_net.png')
print('ROC auc score: {:.3f}'.format(roc_auc_score(test_Y.astype(int), y_pred)))