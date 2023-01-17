# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from keras.layers import Dense,Dropout
from keras.models import Model 
### VALIDATION DATA

# CONVERTING 'val_annotations.txt' TO CSV FILE

val_annotations = pd.read_csv('../input/image-detect/val/val_annotations.txt', sep='\t', names=['image','target','a','b','c','d'])
print(type(val_annotations))
val_annotations.head()
### Converting data to required format

# Creating an ImageDataGenerator object
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load and iterate training data
train_generator = datagen.flow_from_directory('../input/image-detect/train', target_size=(128,128), class_mode='categorical', batch_size=128)

# Load and iterate validation data
val_generator = datagen.flow_from_dataframe(
                        val_annotations,
                        directory='../input/image-detect/val/images',
                        x_col="image",
                        y_col="target",
                        target_size=(128,128),
                        color_mode="rgb",
                        class_mode="categorical",
                        batch_size=64,
                        shuffle=True
                        )


# Load and iterate test data
test_generator = datagen.flow_from_directory('../input/image-detect/test', target_size=(128,128), class_mode=None, batch_size= 64, shuffle=False)

test_generator[0].shape
# 1st layer is Xception layer, used through transfer learning
xception = tf.keras.applications.Xception(
                include_top=False,
                weights="imagenet",
                input_shape=(128,128,3),
                pooling='avg'
            )

# Since we are going to use the weights alredy trained on ImageNet data, so we make all the layers except...
# ...the top layer untrainable.

for layer in xception.layers:
    layer.trainable = False
model = tf.keras.Sequential()
model.add(xception)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(200, activation='softmax'))
model.summary()
# Model Compilation

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
len(model.trainable_variables)
trained = model.fit_generator(
                train_generator,
                validation_data=val_generator,
                epochs=20,
                validation_steps=len(val_generator)    
                )
# Ploting accuracies and losses with epochs

import matplotlib.pyplot as plt

plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('No. of epochs')
plt.ylabel('accuracies')
plt.legend(['Train','Validation'])
plt.show()

plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('No. of epochs')
plt.ylabel('Losses')
plt.legend(['Train','Validation'])
plt.show()
model.save('saved_model.keras')
xception.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(xception.layers))
# Fine-tune from this layer onwards
fine_tune_at = 85

# Freeze all the layers before the `fine_tune_at` layer
for layer in xception.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(len(model.trainable_variables))
fine_tune_epochs = 15
initial_epochs = trained.epoch[-1]
total_epochs =  initial_epochs + fine_tune_epochs

trained_fine = model.fit(train_generator,
                         epochs=total_epochs,
                         initial_epoch =  trained.epoch[-1],
                         validation_data=val_generator)
# Ploting accuracies and losses with epochs

import matplotlib.pyplot as plt

plt.plot(trained_fine.history['accuracy'])
plt.plot(trained_fine.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('No. of epochs')
plt.ylabel('accuracies')
plt.legend(['Train','Validation'])
plt.show()

plt.plot(trained_fine.history['loss'])
plt.plot(trained_fine.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('No. of epochs')
plt.ylabel('Losses')
plt.legend(['Train','Validation'])
plt.show()
model.save('saved_model_fine_19.keras')
prediction=model.predict_generator(test_generator, steps=len(test_generator))
prediction.shape
prediction_index= np.argmax(prediction, axis=1)
prediction_index.shape
labels=train_generator.class_indices
print(type(labels), len(labels))
labels = dict((value,key) for key,value in labels.items())
# labels
predicted_class = [labels[k] for k in prediction_index]
len(predicted_class)
# predicted_class
filenames_ = test_generator.filenames
filenames_

filenames=[]

for e in filenames_:
    e = e[7:]
    filenames.append(e)
results = pd.DataFrame({"file_name":filenames, "category":predicted_class})
results.head()
results.to_csv('results.csv', index=False)
