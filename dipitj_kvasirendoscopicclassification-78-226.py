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
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50, MobileNet,InceptionV3
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D,MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Dense, BatchNormalization, Flatten,Activation
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
img_cols, img_rows = 150,150
channels = 3
classes = 8
batch_size = 32
datagen = ImageDataGenerator(rescale=1./255,
                             zoom_range=0.4,
                             rotation_range = 30,
                             height_shift_range = 0.3,
                             width_shift_range = 0.3,
                             shear_range = 0.3,
                             horizontal_flip = True,
                             vertical_flip = True,
                             fill_mode = "nearest",
                             validation_split = 0.25
                            )
data_path = "/kaggle/input/kvasirgastro/kvasir-dataset"
train_generator = datagen.flow_from_directory(directory=data_path,
                                              target_size=(img_rows, img_cols),
                                              batch_size = batch_size,
                                              subset="training",
                                              shuffle= True,
                                              class_mode = "categorical"
                                             )

val_generator = datagen.flow_from_directory(directory=data_path,
                                              target_size=(img_rows, img_cols),
                                              batch_size = batch_size,
                                              subset="validation",
                                              shuffle= True,
                                              class_mode = "categorical"
                                             )
inception = InceptionV3(weights="imagenet",
           include_top = False,
           input_shape = (img_rows, img_cols, channels))

for layer in inception.layers:
    layer.trainable = False
flat1 = Flatten()(inception.layers[-1].output)
class1 = Dense(1024,activation="relu")(flat1)
output = Dense(classes, activation="softmax")(class1)

incep_model = Model(inputs=inception.inputs, outputs = output)
incep_model.summary()
incep_model.compile(loss="categorical_crossentropy",
             optimizer = Adam(lr=0.001),
             metrics=['acc'])
reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                             patience = 3,
                             factor = 0.2,
                             min_delta = 0.001)
callbacks = [reduce_lr]

history = incep_model.fit(train_generator,
                   epochs=10,
                   steps_per_epoch=train_generator.n//train_generator.batch_size,
                   validation_data = val_generator,
                   validation_steps =val_generator.n//val_generator.batch_size,
                   callbacks = callbacks)
scores = incep_model.evaluate(val_generator, steps=val_generator.n//val_generator.batch_size,verbose=1)
print("Validation Accuracy: %.3f Validation Loss : %.3f"%(scores[1]*100, scores[0]))
incep_model.save("KvasirInceptionV3predictions.h5")
test_predictions = incep_model.predict(val_generator,steps=val_generator.n//val_generator.batch_size,verbose=1)
test_labels = np.argmax(test_predictions,axis=1)
test_labels
img = Image.open('/kaggle/input/kvasirgastro/kvasir-dataset/dyed-lifted-polyps/57cd8658-0886-458e-a875-42a74d420e9c.jpg')
plt.imshow(img)
img = img.resize((150,150))
img = np.array(img)
img = np.true_divide(img,255)
img = img.reshape(1,150,150,3)
predictions = incep_model.predict(img)
predicted_label = np.argmax(predictions, axis=1)
print(predictions, predicted_label)
classes= {
    'TRAIN':['dyed-lifted-polyps','dyed-resection-margins','esophagitis','normal-cecum','normal-pylorus',
            'normal-z-line','polyps','ulcerative-colitis'],
    'VALIDATION':['dyed-lifted-polyps','dyed-resection-margins','esophagitis','normal-cecum','normal-pylorus',
            'normal-z-line','polyps','ulcerative-colitis']
}
predicted_labels = classes['VALIDATION'][predicted_label[0]]
print("I think this image is among the {}.".format(predicted_labels.lower()))
