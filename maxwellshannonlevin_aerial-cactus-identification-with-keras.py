import os

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



sns.set()
cactus_dir = "../input/aerial-cactus-identification/"

train_dir = cactus_dir + "/train/train/"

test_dir = cactus_dir + "/test/test/"



df_train_data = pd.read_csv(cactus_dir + "/train.csv")

df_train_data['has_cactus'] = df_train_data['has_cactus'].astype(str)



df_test = pd.read_csv(cactus_dir + "/sample_submission.csv")
df_train_data.head()
print("The number of training images is: {}".format(len(df_train_data)))



df_train_data['has_cactus'].value_counts()
display, axes = plt.subplots(4, 4, figsize=(16, 16))

imgs = []

labels = []

for i, row in df_train_data.head(16).iterrows():

    fname = os.path.join(train_dir, row['id'])

    imgs.append(load_img(fname))

    labels.append('Cactus' if row['has_cactus'] == '1' else 'No Cactus')



for i, ax in enumerate(axes.reshape(-1)):

    ax.imshow(imgs[i])

    ax.set_title(labels[i])

    ax.axis('off')

plt.show()
has_cactus = df_train_data[df_train_data['has_cactus'] == '1']

not_cactus = df_train_data[df_train_data['has_cactus'] == '0']
# Want 80% of not_cactus to make up 50% of the training set

df_train = not_cactus[:3491]

df_train = df_train.append(has_cactus[:3491])

df_train = df_train.sample(frac=1).reset_index(drop=True)



df_train['has_cactus'].value_counts()
# Want remainder of not_cactus to make up 25% of the validation set

df_valid = not_cactus[3491:]  # Has 873 elements

df_valid = df_valid.append(has_cactus[3491:3491 + 873*3])

df_valid = df_valid.sample(frac=1).reset_index(drop=True)



df_valid['has_cactus'].value_counts()
# Data augmentation for training data

train_datagen = ImageDataGenerator(rotation_range=50, 

                                   width_shift_range=0.2, 

                                   height_shift_range=0.2, 

                                   horizontal_flip=True, 

                                   rescale=1./255)



# No augmentation for validation data

valid_datagen = ImageDataGenerator(rescale=1./255)
# Parameters for resizing the image and number of images in a training/validation batch

new_image_size = 64

batch_size = 128



# Data pipeline: filename - dataframe --> image on disk --> keras model



# Training data pipeline

train_generator = train_datagen.flow_from_dataframe(df_train, 

                                                    directory=train_dir, 

                                                    x_col='id', 

                                                    y_col='has_cactus', 

                                                    target_size=(new_image_size, new_image_size), 

                                                    class_mode='binary', 

                                                    batch_size=batch_size)



# Validation: exactly the same except flows from validation dataframe

valid_generator = valid_datagen.flow_from_dataframe(df_valid, 

                                                    directory=train_dir, 

                                                    x_col='id', 

                                                    y_col='has_cactus', 

                                                    target_size=(new_image_size, new_image_size),

                                                    class_mode='binary',

                                                    batch_size=batch_size)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(new_image_size, new_image_size, 3), activation='relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(512, (3,3), activation='relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer='adam', 

              loss='binary_crossentropy', 

              metrics=['accuracy'])
# Save the model every time its best validation accuracy improves

checkpoint = ModelCheckpoint("model.hdf5", 

                             monitor='val_acc', 

                             verbose=1, 

                             save_best_only=True, 

                             mode='max')



# Reduce the learning rate when the model's performance begins to stagnate 

reduce_lr = ReduceLROnPlateau(monitor='val_acc', 

                              factor=.2,

                              patience=10, 

                              min_lr=1e-9,

                              verbose=1)



# Using early stopping with a high patience value to balance improvement & my own patience

early_stop = EarlyStopping(monitor='val_acc', 

                           patience=100, 

                           verbose=1, 

                           mode='auto', 

                           baseline=None, 

                           restore_best_weights=True)
# hist = model.fit_generator(train_generator, 

#                            steps_per_epoch=len(df_train)//batch_size, 

#                            epochs=1000, 

#                            validation_data=valid_generator, 

#                            validation_steps=len(df_valid)//batch_size, 

#                            callbacks=[checkpoint, reduce_lr, early_stop],

#                            verbose=0)



model = load_model('../input/best-model-1/model.hdf5')
# def plot_from_history(history):

#     history_dict = history.history

#     loss = history_dict['loss']

#     val_loss = history_dict['val_loss']

#     acc = history_dict['acc']

#     val_acc = history_dict['val_acc']

#     epochs = range(1, len(loss) + 1)

    

#     fig, ax1 = plt.subplots(1, figsize=(8, 4), dpi=150)

#     ax1.set_title('Accuracy vs Epoch')

#     ax1.plot(epochs, acc, label='Training Accuracy')

#     ax1.plot(epochs, val_acc, label='Validation Accuracy')

#     ax1.legend(loc='best')

#     plt.show()



# plot_from_history(hist)
model.evaluate_generator(valid_generator, steps=len(df_valid)//batch_size)
test_fnames = []

test_imgs = []

for fname in os.listdir(test_dir):

    test_fnames.append(fname)

    img = load_img(test_dir + fname, target_size=(new_image_size, new_image_size, 3))

    img = img_to_array(img) / 255

    test_imgs.append(img)

test_imgs = np.array(test_imgs)



test_imgs.shape
pred = model.predict(test_imgs)

pred[:5]
submission = pd.DataFrame()

submission['id'] = test_fnames

submission['has_cactus'] = pred

submission.to_csv('submission.csv', index=False)



submission.head()