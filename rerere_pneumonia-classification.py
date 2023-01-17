import random

import os, os.path

import pandas as pd

import numpy as np

import seaborn as sns

import tensorflow as tf

import matplotlib.pyplot as plt



from keras import backend as K

from keras.models import Sequential, model_from_json, Model

from keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint, Callback

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image



from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall



from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, f1_score



from PIL import Image
train_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/"

valid_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/"

test_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/"
DIR = train_path + "NORMAL/"

train_normal = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]

n_train_normal = len(train_normal)



DIR = train_path + "PNEUMONIA/"

train_pneumonia = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]

n_train_pneumonia = len(train_pneumonia)



sns.barplot(["Normal - " + str(n_train_normal), "Pneumonia - " + str(n_train_pneumonia)], [n_train_normal, n_train_pneumonia])
sampling_normal = random.choices(train_normal, k=5)

sampling_pneumonie = random.choices(train_pneumonia, k=5)



images_normal = [Image.open(train_path + "NORMAL/" + sampling_normal[i]) for i in range(0, len(sampling_normal))]

images_pneumo = [Image.open(train_path + "PNEUMONIA/" + sampling_pneumonie[i]) for i in range(0, len(sampling_pneumonie))]



images = images_normal + images_pneumo



plt.figure(figsize=(20,10))

columns = 5

for i, image in enumerate(images):

    plt.subplot(len(images) / columns + 1, columns, i + 1)

    plt.axis('off')

    if i < 5:

        plt.title("NORMAL")

    else:

        plt.title("PNEUMONIA")

    plt.imshow(image, cmap='gray')
batch_size = 32



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        brightness_range=(1.2, 1.5),

        rotation_range=20,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_path,

        shuffle=True,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        valid_path,

        shuffle=True,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='binary')



test_generator = test_datagen.flow_from_directory(

        test_path,

        shuffle=False,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='binary')

threshold = 0.5

metrics = [

      BinaryAccuracy(name='accuracy', threshold = threshold),

      Precision(name='precision', thresholds = threshold),

      Recall(name='recall', thresholds = threshold),

]
base_model = InceptionV3(include_top=False)



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.7)(x)

x = Dense(512, activation='relu')(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation='sigmoid')(x)



model = Model(inputs=base_model.input, outputs=predictions)



# Select the first layer and freeze them

# for layer in base_model.layers[:20]:

#     layer.trainable=False

# for layer in base_model.layers[20:]:

#     layer.trainable=True



# Compile our model

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=metrics)
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
def scheduler(epoch):

  if epoch < 10:

    return 0.001

  else:

    return 0.001 * tf.math.exp(0.1 * (10 - epoch))



learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
mcp = ModelCheckpoint(filepath="best_model.h5", save_best_only=True, save_weights_only=True)
history = model.fit_generator(

        train_generator,

        steps_per_epoch=train_generator.n // train_generator.batch_size,

        epochs=50,

        validation_data=validation_generator,

        validation_steps=validation_generator.n // validation_generator.batch_size,

        callbacks=[mcp, learning_rate_callback])
# serialize weights to HDF5

model.save_weights("model_train.h5")

print("Saved model to disk")
import matplotlib.pyplot as plt

acc = history.history['accuracy']

loss = history.history['loss']

recall = history.history['recall']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, recall, 'g', label='Training recall')

plt.title('Training')

plt.legend(loc=0)

plt.figure()



plt.show()
def evaluate_model(model):

    # evaluate the model

    scores = model.evaluate_generator(test_generator, steps=len(test_generator))

    #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    

    try:

        for name, value in zip(model.metrics_names, scores):

            print(name, ': ', value)

    except: 

        pass

    

    #Confution Matrix and Classification Report

    probabilities = model.predict_generator(test_generator, test_generator.n // test_generator.batch_size + 1)

    y_pred = probabilities > 0.5

    print('Confusion Matrix')

    print(confusion_matrix(test_generator.classes, y_pred))

    

    print('Classification Report')

    target_names = ['Normal', 'Pneumonia']

    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

    

    f1 = f1_score(test_generator.classes, y_pred)

    print("F1 score : " + str(f1))

    

    return y_pred
y_pred = evaluate_model(model)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_generator.classes, y_pred)

auc(fpr_keras, tpr_keras)
# Load json and create model

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("best_model.h5")

print("Loaded model from disk")

loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=metrics)
bestmodel = loaded_model
y_pred = evaluate_model(bestmodel)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_generator.classes, y_pred)

auc(fpr_keras, tpr_keras)
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(test_generator.classes, y_pred)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    plt.xlabel("Threshold")

    plt.legend(loc="upper left")

    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()