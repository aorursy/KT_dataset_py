import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shutil

import cv2

from glob import glob

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

print(os.listdir("../input/dogs-vs-cats/train/train"))

base_dir="../input/dogs-vs-cats/train/train"

filenames = os.listdir("../input/dogs-vs-cats/train/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'path': glob(os.path.join(base_dir,'*.jpg')),

    'category': categories

})

df['category']=df['category'].astype(str)

df.head()

df['category'].value_counts()
fig= plt.figure(figsize=(20,8))

index=1

for i in np.random.randint(low=0, high=df.shape[0],size=10):

         file= df.iloc[i]['path']

         img = cv2.imread(file)

         ax = fig.add_subplot(2, 5, index)

         ax.imshow(img)

         index = index + 1

         ax.set_title(df.iloc[i].category, fontsize = 18,color='white')

plt.show()
from keras.models import Sequential

from keras import layers

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.applications import VGG16

from keras.models import Model

from keras.applications.resnet50 import ResNet50

from keras.applications.resnet50 import ResNet50,preprocess_input

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Dense, Flatten, GlobalAveragePooling2D,BatchNormalization,Dropout,Conv2D,MaxPool2D

from keras.optimizers import Adam



image_size = 224

batch_size = 32



pre_trained_model = InceptionV3(include_top=False, weights="imagenet")





for layer in pre_trained_model.layers:

     layer.trainable = False

# for layer in pre_trained_model.layers[:140]:

#     layer.trainable = False



# for layer in pre_trained_model.layers[140:]:

#     layer.trainable = True



last_output = pre_trained_model.output

    

# Flatten the output layer to 1 dimension

x = GlobalMaxPooling2D()(last_output)

# Add a fully connected layer with 512 hidden units and ReLU activation

x = Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5

x = Dropout(0.5)(x)

# Add a final sigmoid layer for classification

x = layers.Dense(1, activation='sigmoid')(x)



model = Model(pre_trained_model.input, x)



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



model.summary()
train_df, validate_df = train_test_split(df, test_size=0.1)

train_df = train_df.reset_index()

validate_df = validate_df.reset_index()



# validate_df = validate_df.sample(n=100).reset_index() # use for fast testing code purpose

# train_df = train_df.sample(n=1800).reset_index() # use for fast testing code purpose



total_train = train_df.shape[0]

total_validate = validate_df.shape[0]




train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest',

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(image_size, image_size),

    batch_size=batch_size

)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(image_size, image_size),

    batch_size=batch_size

)
# fine-tune the model

epochs = 10

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)

reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)



history = model.fit_generator(

    train_generator,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=[reducel, earlystopper])
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('literation')

plt.legend(['Train', 'Test'], loc='best')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='best')

plt.show()
from sklearn.metrics import roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt



# make a prediction

y_pred_keras = model.predict_generator(validation_generator, steps=len(validation_generator), verbose=1)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(validation_generator.classes, y_pred_keras)

auc_keras = auc(fpr_keras, tpr_keras)

auc_keras
plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()
test_filenames = os.listdir("../input/dogs-vs-cats/test1/test1")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]

nb_samples
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/dogs-vs-cats/test1/test1", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    batch_size=batch_size,

    target_size=(image_size, image_size),

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size),verbose=1)

threshold = 0.5

test_df['category'] = np.where(predict > threshold, 1,0)
import seaborn as sns

submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission_13010030.csv', index=False)



plt.figure(figsize=(10,5))

sns.countplot(submission_df['label'])

plt.title("(Test data)")