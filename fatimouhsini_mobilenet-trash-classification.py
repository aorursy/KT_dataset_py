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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns

import tensorflow



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam

from tensorflow.keras.regularizers import l1, l2, L1L2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



tensorflow.random.set_seed(0)

np.random.seed(0)
path = '../input/mergeddata/merged_full_split_final'
train_datagen = ImageDataGenerator(

        rescale = 1./255,

        rotation_range = 20,

        width_shift_range = 0.2,

        height_shift_range = 0.2,

        horizontal_flip = True,

        vertical_flip = True,

        fill_mode='nearest'

)

validation_datagen = ImageDataGenerator(

        rescale = 1./255

)

test_datagen = ImageDataGenerator(

        rescale = 1./255

)
img_shape = (224, 224, 3) # default values



train_batch_size = 256

val_batch_size = 32



train_generator = train_datagen.flow_from_directory(

            path + '/train',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = train_batch_size,

            class_mode = 'categorical',)



validation_generator = validation_datagen.flow_from_directory(

            path + '/val',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = val_batch_size,

            class_mode = 'categorical',

            shuffle=False)



test_generator = test_datagen.flow_from_directory(

            path + '/test',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = val_batch_size,

            class_mode = 'categorical',

            shuffle=False,)
from keras.applications import MobileNet

MobileNet_model= MobileNet(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) 
# Freeze the layers except the last 3 layers

for layer in MobileNet_model.layers[:-3]:

    layer.trainable = False
# Create the model

model = Sequential()

 

# Add the vgg convolutional base model

model.add(MobileNet_model)



# Add new layers

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dense(1024, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(6, activation='softmax'))
model.summary()
# Compile the model

model.compile(loss='categorical_crossentropy',

              optimizer=Nadam(lr=1e-4),

              metrics=['acc'])
# Train the model

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

mc = ModelCheckpoint('MobileNet Garbage Classifier1.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)



history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples/train_generator.batch_size ,

    epochs=30,

    validation_data=validation_generator,

    validation_steps=validation_generator.samples/validation_generator.batch_size,

    verbose=0,

    callbacks = [es, mc],)
train_acc = history.history['acc']

val_acc = history.history['val_acc']

train_loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(train_acc) + 1)



plt.plot(epochs, train_acc, 'b*-', label = 'Training accuracy')

plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, train_loss, 'b*-', label = 'Training loss')

plt.plot(epochs, val_loss, 'r', label = 'Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
data = np.load('../input/test-data/test_data.npz')

x_test, y_test = data['x_test'], data['y_test']

y_pred = model.predict(x_test)
acc = np.count_nonzero(np.equal(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)))/x_test.shape[0]

print(acc)
import sklearn.metrics as metrics

from sklearn.metrics import classification_report



LABELS = ['cardboard','glass','metal','paper','plastic','trash']



def show_confusion_matrix(validations, predictions):



    matrix = metrics.confusion_matrix(validations, predictions)

    plt.figure(figsize=(6, 4))

    sns.heatmap(matrix,

                cmap='coolwarm',

                linecolor='white',

                linewidths=1,

                xticklabels=LABELS,

                yticklabels=LABELS,

                annot=True,

                fmt='d')

    plt.title('Confusion Matrix')

    plt.ylabel('True Label')

    plt.xlabel('Predicted Label')

    plt.show()



#y_pred_test = model.predict(x_test)

# Take the class with the highest probability from the test predictions

max_y_pred_test = np.argmax(y_pred, axis=1)

max_y_test = np.argmax(y_test, axis=1)



show_confusion_matrix(max_y_test, max_y_pred_test)



print(classification_report(max_y_test, max_y_pred_test))

labels = (train_generator.class_indices)

print(labels)



labels = dict((v,k) for k,v in labels.items())

print(labels)
from keras.preprocessing import image



img_path = '../input/googleimagestest/GoogleImages/paper.jpg'



img = image.load_img(img_path, target_size=(224, 224))

img = image.img_to_array(img, dtype=np.uint8)

img=np.array(img)/255.0



plt.title("Loaded Image")

plt.axis('off')

plt.imshow(img.squeeze())



p=model.predict(img[np.newaxis, ...])



#print("Predicted shape",p.shape)

print("Maximum Probability: ",np.max(p[0], axis=-1))

predicted_class = labels[np.argmax(p[0], axis=-1)]

print("Classified:",predicted_class)
import keras

file = "MobileNetTrashClassifier1.h5"

keras.models.save_model(model,file)
#Getting files from kernel

from IPython.display import FileLinks

FileLinks('.')
from tensorflow.keras.models import load_model



model = load_model('./MobileNet Garbage Classifier1.h5')

        
!pip install coremltools
import coremltools



model.author = 'fati mouhsini'

input_shape = {

    "image":[None, 224, 224, 3]

}

            





coreml_model = coremltools.converters.tensorflow.convert("./MobileNet Garbage Classifier1.h5",

                                                    input_names='image',

                                                    image_input_names='image',

                                                    input_name_shape_dict=input_shape,

                                                    output_names='probabilities',

                                                    class_labels=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'])



coreml_model.save('TrashClassifierMobileNet-tf.mlmodel')

print('The model was saved')
coreml_model