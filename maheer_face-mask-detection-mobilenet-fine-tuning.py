import pandas as pd
import matplotlib.pyplot as plt

img=plt.imread("../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/0007.jpg")
plt.imshow(img)
import os
import cv2
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow
from keras.applications import resnet50, MobileNetV2, Xception
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
anno_dir='/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations/'
images_dir='/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
images=[]
labels=[]
for filename in os.listdir(images_dir):
    num = filename.split('.')[ 0 ]
    print("loading image: {}".format(filename))
    if int(num) > 1800:
        class_name = None
        anno = filename + ".json"
        with open(os.path.join(anno_dir, anno)) as json_file:
            json_data = json.load(json_file)
            no_anno = json_data["NumOfAnno"]
            k = 0
            for i in range(0, no_anno):
                class_nam = json_data['Annotations'][i]['classname']
                if class_nam in ['face_with_mask',"gas_mask", "face_shield", "mask_surgical", "mask_colorful"]:
                    class_name = 'face_with_mask'
                    k = i
                    break
                elif class_nam in ['face_no_mask,"hijab_niqab', 'face_other_covering', "face_with_mask_incorrect", "scarf_bandana", "balaclava_ski_mask", "other" ]:
                    class_name = 'face_no_mask'
                    k = i
                    break
                else:
                    continue
                    
            box = json_data[ 'Annotations' ][k][ 'BoundingBox' ]
            (x1, x2, y1, y2) = box
        if class_name is not None:
            image = cv2.imread(os.path.join(images_dir, filename))
            img = image[x2:y2, x1:y1]
            img = cv2.resize(img, (224, 224))
            img = img[...,::-1].astype(np.float32)
            img = preprocess_input(img)
            images.append(img)
            labels.append(class_name)  
   
images = np.array(images, dtype="float32")
labels = np.array(labels)
print(len(images))
print(len(labels))
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(Xtrain, Xtest, Ytrain, Ytest) = train_test_split(images, labels,test_size=0.20, stratify=labels, random_state=42)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.15,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.15,
                                   horizontal_flip=True,
                                   fill_mode="nearest")
validation_datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=20,
                                        zoom_range=0.15,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.15,
                                        horizontal_flip=True,
                                        fill_mode="nearest")
lr = 1e-4
epochs = 30
BS = 16
baseModel = Xception(weights="../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False,
input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
baseModel.trainable = False
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process


opt = Adam(lr=lr*0.01, decay=0.01)

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
model.summary()
history = model.fit(train_datagen.flow(Xtrain, Ytrain, batch_size=BS),
                    steps_per_epoch=len(Xtrain)//BS,
                    validation_data=(Xtest,Ytest), 
                    validation_steps=len(Xtest)//BS,
                    epochs=7)
baseModel.trainable = True
fine_tune_at = 4

# Freeze all the layers before the `fine_tune_at` layer
for layer in baseModel.layers[:fine_tune_at]:
  layer.trainable =  False
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
model.summary()
history = model.fit(train_datagen.flow(Xtrain, Ytrain, batch_size=BS),
                    steps_per_epoch=len(Xtrain)//BS,
                    validation_data=(Xtest,Ytest), 
                    validation_steps=len(Xtest)//BS,
                    epochs=epochs)
pred = model.predict_classes(Xtest)
from sklearn.metrics import confusion_matrix as cm
c_m = cm(Ytest,pred)
import seaborn as sns
sns.heatmap(c_m, annot= True)