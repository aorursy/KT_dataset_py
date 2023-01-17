%tensorflow_version 1.x
!pip install -U -q kaggle --force
from google.colab import files
f=files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle competitions download -c nnfl-cnn-lab2
!unzip /content/nnfl-cnn-lab2.zip
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

tr_set = pd.read_csv('/content/upload/train_set.csv')
tr_set.head()
tr_set['label']=tr_set['label'].apply(str)
filenames = os.listdir("/content/upload/train_images/train_images")
sample = random.choice(filenames)
image = load_img("../content/upload/train_images/train_images/"+sample)
plt.imshow(image)
print(image.size)
tr_set['label'].value_counts().plot.bar()
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
train_df, validate_df = train_test_split(tr_set, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
IMAGE_SIZE = (150, 150)
train_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True,
    width_shift_range=0.1, height_shift_range=0.1)

train_generator = train_datagen.flow_from_dataframe(train_df, "../content/upload/train_images/train_images",  x_col='image_name', 
                  y_col='label', target_size=IMAGE_SIZE, class_mode = 'categorical')
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, "../content/upload/train_images/train_images", 
                       x_col='image_name', y_col='label', target_size=IMAGE_SIZE, class_mode='categorical')
from keras.applications import ResNet50
model3 = Sequential([ResNet50(include_top = False, weights = None, input_shape = (150,150,3), classes = 6, pooling = 'avg'),
                    Dropout(0.3),
                    Dense(6, activation = 'softmax')])

model3.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model3.summary()
#callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
earlystop = EarlyStopping(monitor = 'val_acc', patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
mc = ModelCheckpoint('/content/model_resnet50.h5', monitor = 'val_acc', verbose = 1)
history = model3.fit_generator(train_generator, epochs = 20, validation_data=validation_generator,
                               callbacks = [earlystop, learning_rate_reduction, mc])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 20, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 20, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
model3.save_weights("model_resnet50.h5")
#Testing 
test_filenames = os.listdir("/content/upload/test_images/test_images")
test_df = pd.DataFrame({'image_name': test_filenames})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(test_df, "/content/upload/test_images/test_images", x_col='image_name', y_col=None, 
                 class_mode=None, target_size=IMAGE_SIZE, shuffle=False)

predict = model3.predict_generator(test_generator)
test_df['label'] = np.argmax(predict, axis=-1)

test_df.head()
test_df.to_csv('submission_01.csv', index = False)
s = pd.read_csv('submission_01.csv')
s.head()
!kaggle competitions submit -c nnfl-cnn-lab2 -f submission_01.csv -m "Message"
from google.colab import drive
drive.mount('/content/gdrive')
model3.save_weights('weights_resnet50_90.h5')
model3.save('resnet50_90.h5')
from keras.applications import ResNet50V2
model4 = Sequential([ResNet50V2(include_top = False, weights = None, input_shape = (150,150,3), classes = 6, pooling = 'avg'),
                    Dropout(0.3),
                    Dense(6, activation = 'softmax')])

model4.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model4.summary()
#callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
earlystop = EarlyStopping(monitor = 'val_acc', patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
mc = ModelCheckpoint('/content/model_resnet50V2.h5', monitor = 'val_acc', verbose = 1, save_best_only = True)
history = model4.fit_generator(train_generator, epochs = 40, validation_data=validation_generator,
                               callbacks = [earlystop, learning_rate_reduction, mc])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 40, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 40, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
#Testing 
test_filenames = os.listdir("/content/upload/test_images/test_images")
test_df = pd.DataFrame({'image_name': test_filenames})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(test_df, "/content/upload/test_images/test_images", x_col='image_name', y_col=None, 
                 class_mode=None, target_size=IMAGE_SIZE, shuffle=False)

predict = model4.predict_generator(test_generator)
test_df['label'] = np.argmax(predict, axis=-1)
test_df.to_csv('submission_02.csv', index = False)
!kaggle competitions submit -c nnfl-cnn-lab2 -f submission_02.csv -m "Message"
from google.colab import drive
drive.mount('/content/gdrive')
model4.save_weights('weights_resnet50V2_92.h5')
#model4.save('resnet50V2_92.h5')
from keras.applications import InceptionResNetV2
model5 = Sequential([InceptionResNetV2(include_top = False, weights = None, input_shape = (150,150,3), classes = 6, pooling = 'avg'),
                    Dropout(0.3),                                                              
                    Dense(6, activation = 'softmax')])
 #Here 2 models were used, one with the Dropout param 0.3 and one with 0.5. 0.3 performed slightly better on public leaderboard, so
 #I have set the value as 0.3 here. The training cell below shows training for 0.5. I have the weights for both of the models stored. 

model5.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model5.summary()
#callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
earlystop = EarlyStopping(monitor = 'val_acc', patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
mc = ModelCheckpoint('/content/model_incepresnetv2_02.h5', monitor = 'val_acc', verbose = 1, save_best_only = True)
history = model5.fit_generator(train_generator, epochs = 50, validation_data=validation_generator,
                               callbacks = [earlystop, learning_rate_reduction, mc])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 50, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 50, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

#Testing 
test_filenames = os.listdir("/content/upload/test_images/test_images")
test_df = pd.DataFrame({'image_name': test_filenames})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(test_df, "/content/upload/test_images/test_images", x_col='image_name', y_col=None, 
                 class_mode=None, target_size=IMAGE_SIZE, shuffle=False)

predict = model5.predict_generator(test_generator)
test_df['label'] = np.argmax(predict, axis=-1)
test_df.to_csv('submission_04.csv', index = False)
!kaggle competitions submit -c nnfl-cnn-lab2 -f submission_04.csv -m "Message"
from google.colab import drive
drive.mount('/content/gdrive')
model5.save_weights('weights_incep_resnetV2_93_Drop05(prev was Drop03).h5')
#model4.save('resnet50V2_92.h5')
