!pip install deepstack
import pandas as pd
import numpy as np 
import os
import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
img_h,img_w= (300,300)
batch_size=128
epochs=10
n_class=48
base_dir = '../input/devnagri-handwritten-character/DEVNAGARI_NEW'
train_dir = os.path.join(base_dir, 'TRAIN')
validation_dir = os.path.join(base_dir, 'TEST')
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

base_model_1=VGG19(include_top=False, weights='imagenet',input_shape=(img_h,img_w,3), pooling='avg')

# Making last layers trainable, because our dataset is much diiferent from the imagenet dataset 
for layer in base_model_1.layers[:-6]:
    layer.trainable=False
    
model_1=Sequential()
model_1.add(base_model_1)

model_1.add(Flatten())
model_1.add(BatchNormalization())
model_1.add(Dropout(0.35))
model_1.add(Dense(n_class,activation='softmax'))
            
model_1.summary()
tf.keras.utils.plot_model(
    model_1,
    to_file="model_1.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=100,
)
from tensorflow.keras.applications.inception_v3 import InceptionV3

base_model_2= InceptionV3(include_top=False, weights='imagenet',
                                        input_tensor=None, input_shape=(img_h,img_w,3), pooling='avg')

for layer in base_model_2.layers[:-30]:
    layer.trainable=False
model_2=Sequential()
model_2.add(base_model_2)
model_2.add(Flatten())
model_2.add(BatchNormalization())
model_2.add(Dense(1024,activation='relu'))
model_2.add(BatchNormalization())

model_2.add(Dense(512,activation='relu'))
model_2.add(Dropout(0.35))
model_2.add(BatchNormalization())

model_2.add(Dense(256,activation='relu'))
model_2.add(Dropout(0.35))
model_2.add(BatchNormalization())

model_2.add(Dense(n_class,activation='softmax'))

model_2.summary()
tf.keras.utils.plot_model(
    model_2,
    to_file="model_2.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=100,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
         rescale=1./255,
         rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

test_datagen= ImageDataGenerator(rescale=1./255)
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=3,
                                         cooldown=2,
                                         min_lr=1e-10,
                                         verbose=1)

callbacks = [reduce_learning_rate]
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model_1.compile( loss='categorical_crossentropy',optimizer= optimizer, metrics=['accuracy'])
model_2.compile( loss='categorical_crossentropy',optimizer= optimizer, metrics=['accuracy'])
train_generator = train_datagen.flow_from_directory(
                    train_dir,                   # This is the source directory for training images
                    target_size=(img_h, img_w),  # All images will be resized to 300x300
                    batch_size=batch_size,
                    class_mode='categorical')


from tensorflow.keras.preprocessing.image import ImageDataGenerator
validation_generator = test_datagen.flow_from_directory(
                        validation_dir,
                        target_size=(img_h, img_w),
                        batch_size=batch_size,
                        class_mode='categorical')
history_1 = model_1.fit(
      train_generator,
      steps_per_epoch=6528//batch_size, 
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=3312//batch_size,  
      callbacks=callbacks,
      verbose=1)
history_2 = model_2.fit(
      train_generator,
      steps_per_epoch=6528//batch_size, 
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=3312//batch_size,  
      callbacks=callbacks,
      verbose=1)
import matplotlib.pyplot as plt
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('VGG-19 model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('VGG-19 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
import matplotlib.pyplot as plt
plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('Inception V3 model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('Inception V3 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from deepstack.base import KerasMember

member1 = KerasMember(name="model1", keras_model=model_1, train_batches=train_generator, val_batches=validation_generator)
member2 = KerasMember(name="model2", keras_model=model_2, train_batches=train_generator, val_batches=validation_generator)
from deepstack.ensemble import DirichletEnsemble
from sklearn.metrics import accuracy_score

wAvgEnsemble = DirichletEnsemble(N=10000, metric=accuracy_score)
wAvgEnsemble.add_members([member1, member2])
wAvgEnsemble.fit()
wAvgEnsemble.describe()
from deepstack.ensemble import StackEnsemble
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#Ensure you have the scikit-learn version >= 0.22 installed
print("sklearn version must be >= 0.22. You have:", sklearn.__version__)

stack = StackEnsemble()

# 2nd Level Meta-Learner
estimators = [
    ('rf', RandomForestClassifier(verbose=0, n_estimators=100, max_depth=15, n_jobs=20, min_samples_split=30)),
    ('etr', ExtraTreesClassifier(verbose=0, n_estimators=100, max_depth=10, n_jobs=20, min_samples_split=20)),
    ('dtc',DecisionTreeClassifier(random_state=0, max_depth=3))
]
# 3rd Level Meta-Learner
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

stack.model = clf
stack.add_members([member1, member2])
stack.fit()
stack.describe(metric=sklearn.metrics.accuracy_score)
stack.save()
stack.load()
model_json = model_1.to_json()
with open("VGG_19.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_1.save_weights("VGG_19_weights.h5")
print("Saved VGG19 to disk")
model_json = model_2.to_json()
with open("Inception_V3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_2.save_weights("Inception_V3_weights.h5")
print("Saved Inception_V3 to disk")