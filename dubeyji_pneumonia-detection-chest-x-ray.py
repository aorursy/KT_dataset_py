import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.densenet import DenseNet121

from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, BatchNormalization

from keras.models import Sequential

from keras.models import Model

from keras import backend as K



from keras.models import load_model
labels = ['Pneumonia', 'Normal']

batch_size = 16



# this is the augmentation configuration we will use for testing:

# only rescaling

train_datagen = ImageDataGenerator(

        rotation_range = 0.2,

        shear_range=0.2)

test_datagen = ImageDataGenerator()

# this is a generator that will read pictures found in

# subfolers of 'data/train', and indefinitely generate

# batches of augmented image data

train_generator = train_datagen.flow_from_directory(

        '/kaggle/input/chest-xray-pneumonia/chest_xray/train',  # this is the target directory

        target_size=(320, 320),  

        batch_size=batch_size,

        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels



# this is a similar generator, for validation data

validation_generator = test_datagen.flow_from_directory(

        '/kaggle/input/chest-xray-pneumonia/chest_xray/val',

        target_size=(320, 320),

        batch_size=batch_size,

        class_mode='binary')



# this is a similar generator, for validation data

test_generator = test_datagen.flow_from_directory(

        '/kaggle/input/chest-xray-pneumonia/chest_xray/test',

        target_size=(320, 320),

        batch_size=batch_size,

        class_mode='binary')
def plots(ims, figsize=(32,32), rows=1, interp=False, titles=None):

    if type(ims[0]) is np.ndarray:

        ims = np.array(ims).astype(np.uint8)

        

    f = plt.figure(figsize=figsize)

    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1



    for i in range(len(ims)):

        sp = f.add_subplot(cols, rows, i+1)

        sp.axis('Off')

        if titles is not None:

            sp.set_title(titles[i], fontsize=12)

        plt.imshow(ims[i], interpolation=None if interp else 'none')



#########################################################################



#Check the training set (with batch of 10 as defined above

imgs, labels = next(train_generator)



#Images are shown in the output

plots(imgs, titles=labels)
labels = train_generator.labels

N = labels.shape[0]



positive_frequencies = np.sum(labels, axis=0) / N

negative_frequencies = 1 - positive_frequencies
pos_weights = negative_frequencies

neg_weights = positive_frequencies

pos_contribution = positive_frequencies * pos_weights 

neg_contribution = negative_frequencies * neg_weights
dense_net = DenseNet121(weights = '../input/densenet/densenet.hdf5', include_top=False, input_shape=(320,320,3))

for layer in dense_net.layers:

    layer.trainable = False

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(320,320,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))



model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(128, activation="relu"))



model.add(Dense(1))

model.add(Activation("sigmoid"))



model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
history = model.fit_generator(train_generator, 

                              validation_data=validation_generator,

                              steps_per_epoch = len(train_generator.labels)/batch_size,

                              validation_steps= len(validation_generator.labels)/batch_size,

                              epochs=10,

                             verbose=1,

                             class_weight = {0:neg_weights, 1:pos_weights})



plt.plot(history.history['loss'])

plt.ylabel("loss")

plt.xlabel("epoch")

plt.title("Training Loss Curve")

plt.show()
# Accuracy 

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
# Accuracy 

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
predicted_vals = model.predict_generator(test_generator)
from sklearn.metrics import confusion_matrix

tick_labels = ['Normal', 'Pneumonia']



cm = confusion_matrix(test_generator.labels, predicted_vals> 0.5)

ax = sns.heatmap(cm, annot=True, fmt="d")

plt.ylabel('Actual')

plt.xlabel('Predicted')

ax.set_xticklabels(tick_labels)

ax.set_yticklabels(tick_labels)
from sklearn.metrics import roc_curve, roc_auc_score

fp, tp, _ = roc_curve(test_generator.labels, predicted_vals)



plt.plot(fp, tp, label='ROC', linewidth=3)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.plot(

  [0, 1], [0, 1], 

  linestyle='--', 

  linewidth=2, 

  color='r',

  label='Chance', 

  alpha=.8

)

plt.grid(True)

ax = plt.gca()

ax.set_aspect('equal')

plt.legend(loc="lower right")
from sklearn.metrics import precision_recall_curve, auc

precision, recall, threshold = precision_recall_curve(test_generator.labels, predicted_vals)

# plot the model precision-recall curve

plt.plot(recall, precision, label='PR')

# axis labels

plt.xlabel('Recall')

plt.ylabel('Precision')

# show the legend

plt.legend()

# show the plot

plt.show()
auc_score = auc(recall, precision)

auc_score
model = Sequential()

model.add(dense_net)

model.add(Conv2D(32, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))



model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(128, activation="relu"))



model.add(Dense(1))

model.add(Activation("sigmoid"))



model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
history = model.fit_generator(train_generator, 

                              validation_data=validation_generator,

                              steps_per_epoch = len(train_generator.labels)/batch_size,

                              validation_steps= len(validation_generator.labels)/batch_size,

                              epochs=10,

                             verbose=1,

                             class_weight = {0:neg_weights, 1:pos_weights})



plt.plot(history.history['loss'])

plt.ylabel("loss")

plt.xlabel("epoch")

plt.title("Training Loss Curve")

plt.show()
# Accuracy 

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
# Accuracy 

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
predicted_vals = model.predict_generator(test_generator)
from sklearn.metrics import confusion_matrix

tick_labels = ['Normal', 'Pneumonia']



cm = confusion_matrix(test_generator.labels, predicted_vals> 0.5)

ax = sns.heatmap(cm, annot=True, fmt="d")

plt.ylabel('Actual')

plt.xlabel('Predicted')

ax.set_xticklabels(tick_labels)

ax.set_yticklabels(tick_labels)
from sklearn.metrics import roc_curve, roc_auc_score

fp, tp, _ = roc_curve(test_generator.labels, predicted_vals)



plt.plot(fp, tp, label='ROC', linewidth=3)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.plot(

  [0, 1], [0, 1], 

  linestyle='--', 

  linewidth=2, 

  color='r',

  label='Chance', 

  alpha=.8

)

plt.grid(True)

ax = plt.gca()

ax.set_aspect('equal')

plt.legend(loc="lower right")
from sklearn.metrics import precision_recall_curve, auc

precision, recall, threshold = precision_recall_curve(test_generator.labels, predicted_vals)

# plot the model precision-recall curve

plt.plot(recall, precision, label='PR')

# axis labels

plt.xlabel('Recall')

plt.ylabel('Precision')

# show the legend

plt.legend()

# show the plot

plt.show()
auc_score = auc(recall, precision)

auc_score