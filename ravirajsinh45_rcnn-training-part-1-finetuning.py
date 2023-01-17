import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

import os

from tqdm import tqdm

import json

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns
train_path = '/kaggle/input/rcnn-data-preprocessing-part-2/Train/'

test_path = '/kaggle/input/rcnn-data-preprocessing-part-2/Test/'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
BATCH_SIZE = 64

IMAGE_SIZE = (224,224,3)
train_generator = ImageDataGenerator(rescale=1./255,validation_split=0.2)



train_data = train_generator.flow_from_directory(train_path,

                                                 target_size=(224, 224),

                                                 color_mode="rgb",

                                                 class_mode="categorical",

                                                 batch_size=BATCH_SIZE,

                                                 shuffle=True,

                                                 subset='training')



val_data = train_generator.flow_from_directory(train_path,

                                                 target_size=(224, 224),

                                                 color_mode="rgb",

                                                 class_mode="categorical",

                                                 batch_size=BATCH_SIZE,

                                                 shuffle=False,

                                                 subset='validation')



test_generator  = ImageDataGenerator(rescale=1./255)

test_data = test_generator.flow_from_directory(test_path,

                                                 target_size=(224, 224),

                                                 color_mode="rgb",

                                                 class_mode="categorical",shuffle=False,

                                                 batch_size=BATCH_SIZE)
from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import Dense,Flatten,Input,Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping
baseModel = VGG16(weights="imagenet", include_top=False,

    input_tensor=Input(shape=IMAGE_SIZE))



headModel = baseModel.output

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(4096, activation='relu')(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(4096, activation='relu')(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(3, activation='softmax')(headModel)



for layer in baseModel.layers:

    layer.trainable = False



model = Model(inputs=baseModel.input, outputs=headModel)



opt = Adam(lr=0.001)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()
early_stop = EarlyStopping(patience=2,monitor='val_loss')
results = model.fit_generator(train_data,epochs=20,

                              validation_data=val_data,

                             callbacks=[early_stop])
pd.DataFrame(model.history.history)[['accuracy','val_accuracy']].plot()
pd.DataFrame(model.history.history)[['loss','val_loss']].plot()
test_pred = model.predict_generator(test_data)
pred_class = [np.argmax(x) for x in test_pred]
test_data.class_indices
true_class = test_data.classes
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(true_class,pred_class))
sns.heatmap(confusion_matrix(true_class,pred_class),annot=True)
mapping_class = test_data.class_indices
mapping_class = dict([(value, key) for key, value in mapping_class.items()]) 
images, labels = next(iter(test_data))

images = images.reshape(64, 224,224,3)

fig, axes = plt.subplots(4, 4, figsize=(16,16))



for ax, img, label in zip(axes.flat, images[:16], labels[:16]):

    ax.imshow(img)

    true_label = mapping_class[np.argmax(label)]

    

    pred_prob = model.predict(img.reshape(1, 224,224, 3))

    pred_label = mapping_class[np.argmax(pred_prob)]

    

    prob_class = np.max(pred_prob) * 100

    

    ax.set_title(f"TRUE LABEL: {true_label}", fontweight = "bold", fontsize = 12)

    ax.set_xlabel(f"PREDICTED LABEL: {pred_label}\nProb({pred_label}) = {(prob_class):.2f}%",

                 fontweight = "bold", fontsize = 10,

                 color = "blue" if true_label == pred_label else "red")

    

    ax.set_xticks([])

    ax.set_yticks([])

    

plt.tight_layout()

fig.suptitle("PREDICTION for 16 RANDOM TEST IMAGES", size = 30, y = 1.03, fontweight = "bold")

plt.show()


misclassify_pred = np.nonzero(true_class != pred_class)[0]

fig, axes = plt.subplots(4, 4, figsize=(16, 16))



for ax, batch_num, image_num in zip(axes.flat, misclassify_pred // BATCH_SIZE, misclassify_pred % BATCH_SIZE):

    images, labels = test_data[batch_num]

    img = images[image_num]

    ax.imshow(img.reshape(*IMAGE_SIZE))

    

    true_label = mapping_class[np.argmax(label)]

    

    pred_prob = model.predict(img.reshape(1, 224,224, 3))

    pred_label = mapping_class[np.argmax(pred_prob)]

    

    prob_class = np.max(pred_prob)*100

    

    

    ax.set_title(f"TRUE LABEL: {true_label}", fontweight = "bold", fontsize = 12)

    ax.set_xlabel(f"PREDICTED LABEL: {pred_label}\nProb({pred_label}) = {(prob_class):.2f}%",

                 fontweight = "bold", fontsize = 10,

                 color = "blue" if true_label == pred_label else "red")

    

    ax.set_xticks([])

    ax.set_yticks([])

    

plt.tight_layout()

fig.suptitle(f"MISCLASSIFIED TEST IMAGES ({len(misclassify_pred)} out of {len(true_class)})",

             size = 20, y = 1.03, fontweight = "bold")

plt.show()
model.save('RCNN_crop_weed_classification_model.h5')