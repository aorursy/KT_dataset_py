# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import GlobalAvgPool2D

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import load_model 
from keras.preprocessing.image import load_img,img_to_array

base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(299,299,3))

base_model.summary()
base_model.input
base_model.output
# don't train existing weights
for layer in base_model.layers:
    layer.trainable = False
gap_layer = GlobalAvgPool2D() (base_model.output) # in-place of flatten

dense_layer_1 = Dense(512, activation="relu") (gap_layer)
dense_layer_2 = Dense(256, activation="relu") (dense_layer_1)

output_layer = Dense(10, activation="softmax") (dense_layer_2)
# create a model object
model = Model(inputs=base_model.input, outputs=output_layer)
#compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# summery 
model.summary()
training_path = "../input/10-monkey-species/training/training/"
testing_path = "../input/10-monkey-species/validation/validation/"
#for train datasets
generator = ImageDataGenerator(rescale=1/255,
                               zoom_range = 0.2,
                               width_shift_range=0.15,
                               height_shift_range=0.15,
                               horizontal_flip=True)
training_instances = generator.flow_from_directory(training_path, 
                                                   target_size=(299, 299),
                                                   batch_size=32)
#for test dataset
generator = ImageDataGenerator(rescale=1/255)
test_instances = generator.flow_from_directory(testing_path,
                                               target_size=(299, 299),
                                               batch_size=32)

Epochs=20
Batch_size=32
#early stopping
es=EarlyStopping(monitor="val_loss",
                           patience=5)
#fit
model_fit=model.fit(training_instances, 
                    steps_per_epoch=1098//Batch_size,
                    epochs=Epochs,
                    validation_data=test_instances,
                    callbacks=[es]
                   )

# evaluate model
test_eval = model.evaluate(test_instances , verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#---------------------------------------------------

accuracy = model_fit.history['accuracy']
val_accuracy = model_fit.history['val_accuracy']
loss = model_fit.history['loss']
val_loss = model_fit.history['val_loss']

epochs = range(len(accuracy))

plt.figure(figsize=(15,6))

plt.subplot(121)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(122)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save("model_save.h5")

model_loaded = load_model('model_save.h5')
model_loaded.evaluate(test_instances )
labels_info = []

labels_path = Path("../input/10-monkey-species/monkey_labels.txt")

# Read the file
lines = labels_path.read_text().strip().splitlines()[1:]
for line in lines:
    line = line.split(',')
    line = [x.strip(' \n\t\r') for x in line]
    line[3], line[4] = int(line[3]), int(line[4])
    line = tuple(line)
    labels_info.append(line)
    
# Convert the data into a pandas dataframe
labels_info = pd.DataFrame(labels_info, columns=['Label', 'Latin Name', 'Common Name', 
                                                 'Train Images', 'Validation Images'], index=None)
# Sneak peek 
labels_info


# #load the image 
img_path=training_path+"n1/n1017.jpg"
img = load_img(path=img_path, target_size=(299, 299,3))
# #convert to array
img_arr = img_to_array(img)
# # prepare pixel data
img_arr=img_arr/255
# # reshape into a single sample with 3 channels
img_reshape = img_arr.reshape(1, 299, 299, 3)   #/ 255
# # img_arr
# plt.imshow(img_arr)
Image.open(img_path)
y_pred = model_loaded.predict(img_reshape)
y_pred
y_pred_label = np.argmax(y_pred, axis=1)
display(y_pred_label)
lab=training_instances.class_indices
label=list(lab.keys())
label_name=labels_info['Common Name'].tolist()
print(f"label : {label[y_pred_label[0]]}")
print(f"label_name : {label_name[y_pred_label[0]]}")


