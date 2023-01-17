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
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model,load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.optimizers import Adam
train_data = "/kaggle/input/brain-tumor-classification-mri/Training/"
test_data="/kaggle/input/brain-tumor-classification-mri/Testing/"
img_rows, img_cols = 224, 224
batch_size = 32
classes = 4

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=25,
                                   height_shift_range = 0.3,
                                   width_shift_range = 0.3,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = "nearest")

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 32
train_generator = train_datagen.flow_from_directory(train_data,
                                                    batch_size = batch_size,
                                                    target_size = (img_rows, img_cols),
                                                    class_mode = "categorical",
                                                    shuffle = True)
test_generator = test_datagen.flow_from_directory(test_data,
                                                batch_size = batch_size,
                                                target_size = (img_rows, img_cols),
                                                class_mode = "categorical")
from keras.applications import VGG16
from keras.layers import ZeroPadding2D,GlobalAveragePooling2D
vggmodel = VGG16(weights="imagenet",
                input_shape=(img_rows,img_cols,3),
                include_top=False)
for layer in vggmodel.layers:
    layer.trainable=False
    
def top_model(bottom_model,num_classes):
    top_model = bottom_model.output
    top_model = BatchNormalization()(top_model)
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation="relu")(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(512,activation="relu")(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(num_classes,activation="softmax")(top_model)
    return top_model
    
FC_Head = top_model(vggmodel,classes)
vgg_model = Model(inputs = vggmodel.input, outputs = FC_Head)
vgg_model.summary()
epochs=50
train_samples = 2870
test_samples = 394
vgg_model.compile(loss="categorical_crossentropy",
             optimizer="adam",
             metrics=['acc'])
history_vgg = vgg_model.fit(train_generator,
                   epochs=epochs,
                   steps_per_epoch = train_samples//batch_size,
                   validation_data = test_generator,
                   validation_steps = test_samples//batch_size)
scores=vgg_model.evaluate(test_generator,steps=test_samples//batch_size+1,verbose=1)
print("Test Accuracy: %.3f Test Loss: %.3f"%(scores[1]*100,scores[0]))
vgg_model.save('vgg_model_brainTumor.h5')
classifier = load_model('vgg_model_brainTumor.h5')
test_pred = classifier.predict(test_generator,steps = test_samples//batch_size, verbose=1)
test_labels = np.argmax(test_pred,axis=1)
test_labels
from PIL import Image 
from matplotlib import pyplot as plt 
IMG = Image.open('/kaggle/input/brain-tumor-classification-mri/Testing/pituitary_tumor/image(52).jpg')
plt.imshow(IMG)
IMG = IMG.resize((224,224))
IMG = np.array(IMG)
IMG = np.true_divide(IMG,255)
IMG = IMG.reshape(1, 224,224, 3)
predictions = classifier.predict(IMG)
predicted_classes = np.argmax(predictions,axis=1)
print(predictions, predicted_classes)
classes = {
    'TRAIN':['glioma_tumor', 'meningioma_tumor','no_tumor','pituitary_tumor'],
    'TEST':['glioma_tumor', 'meningioma_tumor','no_tumor','pituitary_tumor']}

predicted_class = classes['TEST'][predicted_classes[0]]
print("I think this image is among the {}.".format(predicted_class.lower()))
