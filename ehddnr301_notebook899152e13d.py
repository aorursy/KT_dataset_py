import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("../input/dataset-for-mask-detection"))
IMAGE_WIDTH=200
IMAGE_HEIGHT=200
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
BATCH_SIZE=32
EPOCHS=30
filenames_with = os.listdir("../input/dataset-for-mask-detection/dataset/with_mask")
filenames_without = os.listdir("../input/dataset-for-mask-detection/dataset/without_mask")
filenames_list_with = []
filenames_list_without = []
categories_with = []
categories_without = []
for filename in filenames_with:
    filenames_list_with.append("../input/dataset-for-mask-detection/dataset/with_mask/" + filename)
    categories_with.append(1)
for filename in filenames_without:
    filenames_list_without.append("../input/dataset-for-mask-detection/dataset/without_mask/" + filename)
    categories_without.append(0)
    

df_w = pd.DataFrame({
    'image': filenames_list_with,
    'category': categories_with
})
df_wo = pd.DataFrame({
    'image': filenames_list_without,
    'category': categories_without
})
print(df_w.shape, df_wo.shape)
df = df_w.append(df_wo)
print(df)

# create model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, BatchNormalization, MaxPooling2D, Dropout

def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', strides=(2,2), input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model
model1 = create_model()
model1.summary()
df["category"] = df["category"].replace({0: 'unmasked', 1: 'masked'}) 
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=3, stratify = df['category'])
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
print(total_train)
print(total_validate)
train_df.to_csv('train_df.csv')
validate_df.to_csv('validate_df.csv')
a = pd.read_csv('./train_df.csv')
b = pd.read_csv('./validate_df.csv')
print(a)
print(b)
print('train_count')
print(train_df['category'].value_counts())
print('-----------------------------')
print('validation_count')
print(validate_df['category'].value_counts())
train_datagen = ImageDataGenerator(
    rescale=1./255,
#     horizontal_flip=True,
    zoom_range=0.4
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    target_size=IMAGE_SIZE,
    x_col="image",
    y_col="category",
    class_mode='binary',
    batch_size=BATCH_SIZE,
    validate_filenames=False
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    x_col="image",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    validate_filenames=False
)

# 경로를 입력해주어야 밑에서 사진을 그릴수있는데 dataset이 경로가 나뉘어져있어서 가끔 안그려집니다.

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    x_col='image',
    y_col='category',
    classes=['masked','unmasked'],
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    validate_filenames=False
)
print(example_generator)
plt.figure(figsize=(12, 12))
for i in range(0, 2):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

callbacks = [learning_rate_reduction]
history = model1.fit_generator(
    train_generator, 
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=total_validate//BATCH_SIZE,
    steps_per_epoch=total_train//BATCH_SIZE,
    callbacks=callbacks
)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, EPOCHS, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, EPOCHS, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
