# Import The Libraries

import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!unzip /kaggle/input/dogs-vs-cats/train.zip -d train
!unzip /kaggle/input/dogs-vs-cats/test1.zip -d test
## Prepare Traning Data
filenames = os.listdir('train/train')
categories = []
for filename in filenames: 
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else :
        categories.append(0)

df= pd.DataFrame({'filename': filenames,
                  'category': categories})
df.head()
df.tail()
df.category.value_counts().plot.bar()
# See Sample
sample = random.choice(filenames)
image = load_img('train/train/'+sample)
plt.imshow(image)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

cnn = Sequential()

cnn.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (128, 128, 3)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size= (2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64, (3, 3), activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size= (2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, (3, 3), activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size= (2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(512, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))
cnn.add(Dense(2, activation = 'softmax'))
cnn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
df.category = df.category.replace({0 : 'cat', 1 : 'dog'})
train_df, test_df = train_test_split(df, test_size = 0.20, random_state= 42)
train_df = train_df.reset_index(drop = True)
test_df = test_df.reset_index(drop = True)
train_df.category.value_counts().plot.bar()
test_df.category.value_counts().plot.bar()
total_train = train_df.shape[0]
total_test = test_df.shape[0]
batch_size=15
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "dataset/train/", 
    x_col='filename',
    y_col='category',
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=batch_size
)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    "dataset/train/", 
    x_col='filename',
    y_col='category',
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=batch_size
)
sample_df = train_df.sample(n= 1).reset_index(drop = True)
sample_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "train/train/", 
    x_col='filename',
    y_col='category',
    target_size=(128, 128),
    class_mode='categorical',
)
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in sample_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
FAST_RUN = False
epochs=3 if FAST_RUN else 11
history = cnn.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=total_test//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
    
)
!pip install PyYAML
model_yaml = cnn.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
cnn.save_weights("model.h5")
print("Saved model to disk")
test_filenames = os.listdir("dataset/test/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "dataset/test/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(128, 128),
    batch_size=batch_size,
    shuffle=False
)
predict = cnn.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
sample_test = test_df.head(36)
sample_test.head()
plt.figure(figsize=(12, 20))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("dataset/test/"+filename, target_size=(128, 128))
    plt.subplot(9, 4, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()

plt.show()
