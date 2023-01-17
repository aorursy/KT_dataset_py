import numpy as np

import pandas as pd

import os



import keras

keras.__version__
DATA_DIRECTORY = os.path.abspath("../input/data/data")



TEST_SPLIT = 0.2

VALIDATION_SPLIT = 0.2

RANDOM_SEED=42



np.random.seed(RANDOM_SEED)
def categorized_from_directory(path):

    """Returns a Pandas dataframe with the `category` and `path` of each image."""

    rows = []

    for category in os.listdir(path):

        category_path = os.path.join(path, category)

        for image in os.listdir(category_path):

            image_path = os.path.join(category_path, image)

            rows.append({'category': category, 'path': image_path})

    return pd.DataFrame(rows)
all_classes = [f"{i}x{j}" for i in range(7) for j in range(0, i + 1)]
from sklearn.model_selection import train_test_split



full_data = categorized_from_directory(DATA_DIRECTORY)



# Put aside a test set for final evaluation

train_data, test_data = train_test_split(

    full_data, 

    test_size=TEST_SPLIT, 

    stratify=full_data['category'])



# Further decompose training data for training and validation

train_data, validation_data = train_test_split(

    train_data, 

    test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),

    stratify=train_data['category'])
num_categories = len(full_data['category'].unique())



assert num_categories == len(all_classes)



num_categories
train_data.head()
len(train_data), len(validation_data), len(test_data)
train_data.groupby('category').count()
test_data.groupby('category').count()
BATCH_SIZE = 20

IMAGE_SIZE = (100, 100)
from keras.preprocessing.image import ImageDataGenerator



def flow_from_datagenerator(datagen, data, batch_size=BATCH_SIZE, shuffle=True):

    """Returns a generator from an ImageDataGenerator and a dataframe."""

    return datagen.flow_from_dataframe(

        dataframe=data, 

        x_col="path", 

        y_col="category", class_mode='categorical', 

        batch_size=batch_size, 

        target_size=IMAGE_SIZE,

        shuffle=shuffle,

        classes=all_classes)
train_datagen = ImageDataGenerator(

    rescale=1.0 / 255, 

    rotation_range=360, 

    #width_shift_range=0.1, 

    #height_shift_range=0.1,

    brightness_range=(-0.1, 0.1),

    #shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=False,

    vertical_flip=False)



train_generator = flow_from_datagenerator(train_datagen, train_data)



train_steps = train_generator.n // train_generator.batch_size
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = flow_from_datagenerator(validation_datagen, validation_data)



validation_steps = validation_generator.n // validation_generator.batch_size
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = flow_from_datagenerator(test_datagen, test_data)



test_steps = test_generator.n // test_generator.batch_size
from keras import layers

from keras import models

from keras import optimizers

from keras import callbacks
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(100, 100, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.3))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(num_categories, activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
model.summary()
EPOCHS = 1200
# Overwrite best model — we don't want to accidentally fill the storage space

checkpoint = callbacks.ModelCheckpoint(

    "best_model.h5", 

    monitor='val_acc', 

    verbose=1, 

    save_best_only=True, 

    mode='max')



history = model.fit_generator(

    train_generator, 

    steps_per_epoch=train_steps, 

    epochs=EPOCHS, 

    validation_data=validation_generator, 

    validation_steps=validation_steps,

    callbacks=[checkpoint]

    )
model.save('final_model.h5')
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()
plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()
pd.DataFrame({'metric': model.metrics_names, 

              'values': model.evaluate_generator(test_generator, steps=test_steps)})
from keras.models import load_model

best_model = load_model("best_model.h5")



pd.DataFrame({'metric': best_model.metrics_names, 

              'values': best_model.evaluate_generator(test_generator, steps=test_steps)})
full_datagen = ImageDataGenerator(rescale=1.0 / 255)

full_generator = flow_from_datagenerator(full_datagen, full_data)



full_steps = full_generator.n // full_generator.batch_size



pd.DataFrame({'metric': best_model.metrics_names, 

              'values': best_model.evaluate_generator(full_generator, steps=full_steps)})
NUM_SAMPLES = 10

sample_data = test_data.sample(NUM_SAMPLES)



sample_datagen = ImageDataGenerator(rescale=1.0 / 255)

sample_generator = flow_from_datagenerator(sample_datagen, sample_data, 

                                           batch_size=1, shuffle=False)



sample_steps = sample_generator.n // sample_generator.batch_size
from keras.preprocessing import image



sample_predictions = np.argmax(best_model.predict_generator(sample_generator, 

                                                            steps=sample_steps), axis=1)



for i in range(NUM_SAMPLES):

    img = image.load_img(sample_data['path'].iloc[i], target_size=IMAGE_SIZE)

    plt.figure(i)

    plt.imshow(image.array_to_img(img))

    true_category = sample_data['category'].iloc[i]

    predicted_category = all_classes[sample_predictions[i]]

    plt.title(f"Predicted category {predicted_category} (actual {true_category})")