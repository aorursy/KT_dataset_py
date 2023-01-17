# Load packages

import os

import random

import shutil



import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd



from IPython.display import SVG



from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.layers import Add, Dense, Dropout

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model



from time import time
# Define some folders

DATA_DIR = '/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset'

WORKING_DIR = '/kaggle/working'

TRAIN_DIR = os.path.join(WORKING_DIR, 'train')

VAL_DIR = os.path.join(WORKING_DIR, 'validation')
def move_data(input_dir, output_dir, *args):

    """Move images from input_dir to output_dir.

    :param input_dir: Input directory

    :param output_dir: Output directory

    """

    if not os.path.exists(output_dir):

        os.mkdir(output_dir)

    for element in args:

        if not os.path.exists(os.path.join(output_dir, element)):

            os.mkdir(os.path.join(output_dir, element))

        files = os.listdir(os.path.join(input_dir, element))

        for file in files:

            try:

                shutil.copy(os.path.join(input_dir, element, file), 

                            os.path.join(output_dir, element, file))

            except OSError as e:

                raise

                

def rearrange_folders(folder, n_val=500, *args):

    """Rearrange folders to be compliant with Keras requirement.

    :param folder: Folder to rearrange

    """

    # Create train and validation folder

    if not os.path.exists(os.path.join(folder, 'train')):

        os.mkdir(os.path.join(folder, 'train'))

    if not os.path.exists(os.path.join(folder, 'validation')):

        os.mkdir(os.path.join(folder, 'validation'))

    

    for element in args:

        files = os.listdir(os.path.join(folder, element))

        val_files = random.sample(files, k=n_val)

        if not os.path.exists(os.path.join(folder, 'train', element)):

            os.mkdir(os.path.join(folder, 'train', element))

        if not os.path.exists(os.path.join(folder, 'validation', element)):

            os.mkdir(os.path.join(folder, 'validation', element))

        for file in files:

            if file in val_files:

                shutil.copy(os.path.join(folder, element, file), 

                            os.path.join(folder, 'validation', element, file))

            else:

                shutil.copy(os.path.join(folder, element, file), 

                            os.path.join(folder, 'train', element, file))

        shutil.rmtree(os.path.join(WORKING_DIR, element))
# Move data

move_data(DATA_DIR, WORKING_DIR, 'homer_simpson', 'bart_simpson')

# Reorder folder

rearrange_folders(WORKING_DIR, 500, 'homer_simpson', 'bart_simpson')
train_bart = os.listdir(os.path.join(TRAIN_DIR, 'bart_simpson'))

img = load_img(os.path.join(TRAIN_DIR, 'bart_simpson', train_bart[0]))



# Convert image to array

X = img_to_array(img)



print(f'Shape of the image array: {X.shape}.')
# Show the image

plt.imshow(X.astype(np.uint8))

plt.axis('off')

plt.show()
# Define ImageDataGenerator

augmenting_data_gen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    channel_shift_range=9,

    fill_mode='nearest'

)
flow = augmenting_data_gen.flow(X[np.newaxis, :, :, :])



plt.figure(figsize=(11, 5))

for i, x_augmented in zip(range(15), flow):

    plt.subplot(3, 5, i + 1)

    plt.imshow(x_augmented[0])

    plt.axis('off')

plt.show()
flow = augmenting_data_gen.flow_from_directory(TRAIN_DIR,

                                               batch_size=1,

                                               target_size=(224, 224))



plt.figure(figsize=(11, 5))

for i, (X, y) in zip(range(15), flow):

    plt.subplot(3, 5, i + 1)

    plt.imshow(X[0])

    plt.axis('off')

plt.show()
# Load the model

full_imagenet_model = ResNet50(weights='imagenet')
# Visualization of the model

plot_model(full_imagenet_model, show_layer_names=False, show_shapes=True)
# Define a model based on the previous one

output = full_imagenet_model.layers[-2].output

base_model = Model(full_imagenet_model.input, output)
def preprocess_function(x):

    """Preprocess x to be used in ResNet50

    :param x: Input image

    """

    if x.ndim == 3:

        x = x[np.newaxis, :, :, :]

    return preprocess_input(x)
BATCH_SIZE = 50



data_gen = ImageDataGenerator(preprocessing_function=preprocess_function)



train_flow = data_gen.flow_from_directory(TRAIN_DIR, 

                                          batch_size=BATCH_SIZE,

                                          target_size=(224, 224),

                                          class_mode='binary',

                                          shuffle=True)



X, y = next(train_flow)

print(f'Shape of the input batch images: {X.shape}, and shape of the output batch images: {y.shape}.')
%%time



features = []

labels = []



count = 0

for X, y in train_flow:

    labels.append(y)

    features.append(base_model.predict(X))

    count += len(y)

    if count % 100 == 0:

        print(f'Processed {count} images.')

    if count >= 2500:

        break
# Concatenate the results

labels_train = np.concatenate(labels)

features_train = np.vstack(features)
# Define the classification model

n_samples, n_features = features_train.shape



top_model = Sequential()

top_model.add(Dense(1, input_dim=n_features, 

                    activation='sigmoid'))

top_model.compile(optimizer=Adam(lr=1e-4),

                  loss='binary_crossentropy',

                  metrics=['accuracy'])



history = top_model.fit(features_train, labels_train,

                        validation_split=0.1,

                        verbose=2,

                        epochs=15)
# Define the complete model

model = Model(base_model.input, top_model(base_model.output))
flow = ImageDataGenerator().flow_from_directory(VAL_DIR,

                                                batch_size=1,

                                                target_size=(224, 224))



# Predict some of the image in the validation set

plt.figure(figsize=(12, 10))

for i, (X, y) in zip(range(15), flow):

    plt.subplot(3, 5, i + 1)

    plt.imshow(X[0] / 255)

    pred = model.predict(preprocess_input(X))[0]

    label = "Homer" if y[:, 1] > 0.5 else "Bart"

    pred_label = "Homer" if pred > 0.5 else "Bart"

    plt.title(f'Pred label: {pred_label}\nProba: {pred[0]:.3}\nTrue label: {label}')

    plt.axis('off')

plt.show()
val_gen = ImageDataGenerator(preprocessing_function=preprocess_function)

val_flow = val_gen.flow_from_directory(VAL_DIR, batch_size=BATCH_SIZE,

                                       target_size=(224, 224),

                                       shuffle=False,class_mode='binary')



predicted_batches = []

all_correct = []

label_batches = []

for i, (X, y) in zip(range(val_flow.n // BATCH_SIZE), val_flow):

    pred = model.predict(X).ravel()

    predicted_batches.append(pred)

    correct = list((pred > 0.5) == y)

    all_correct.extend(correct)

    label_batches.append(y)

    print(f'Processed {len(all_correct)} images.')
print(f'Accuracy on the validation set: {np.round(100 * np.mean(all_correct), 2)}%.')
predictions = np.concatenate(predicted_batches)

true_labels = np.concatenate(label_batches)
N_MISTAKES = 10

top_mistakes = np.abs(true_labels - predictions).argsort()[::-1][:N_MISTAKES]



images_names = np.array(val_flow.filenames, dtype=np.object)[top_mistakes]



plt.figure(figsize=(15, 10))

for i, (img, pred, y) in enumerate(zip(images_names,

                                    predictions[top_mistakes],

                                    true_labels[top_mistakes])):

    plt.subplot(2, 5, i + 1)

    img_load = load_img(os.path.join(VAL_DIR, img))

    img_arr = img_to_array(img_load)

    label = "Homer" if y > 0.5 else "Bart"

    pred_label = "Homer" if pred > 0.5 else "Bart"

    plt.imshow(img_arr.astype(np.uint8))

    plt.title(f'Pred label: {pred_label}\nProba: {pred:.3}\nTrue label: {label}')

    plt.axis('off')



plt.show()
[(i, l.output_shape) for (i, l) in enumerate(model.layers) if isinstance(l, Add)]
for i, layer in enumerate(model.layers):

    layer.trainable = i >= 151
augmenting_data_gen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest',

    preprocessing_function=preprocess_function

)



train_flow = augmenting_data_gen.flow_from_directory(TRAIN_DIR,

                                                     target_size=(224, 224),

                                                     batch_size=BATCH_SIZE,

                                                     class_mode='binary',

                                                     shuffle=True,

                                                     seed=42)



opt = SGD(lr=1e-4, momentum=0.9)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_flow,

                    epochs=30,

                    steps_per_epoch=train_flow.n // BATCH_SIZE,

                    validation_data=val_flow,

                    validation_steps=val_flow.n // BATCH_SIZE)
plt.subplot(1, 2, 1)

plt.plot(history.history['loss'], label='Train')

plt.plot(history.history['val_loss'], label='Validation')

plt.legend()

plt.title('Loss')



plt.subplot(1, 2, 2)

plt.plot(history.history['accuracy'], label='Train')

plt.plot(history.history['val_accuracy'], label='Validation')

plt.legend()

plt.title('Accuracy')



plt.show()
val_gen = ImageDataGenerator(preprocessing_function=preprocess_function)

val_flow = val_gen.flow_from_directory(VAL_DIR, batch_size=BATCH_SIZE,

                                       target_size=(224, 224),

                                       shuffle=False,class_mode='binary')



predicted_batches = []

all_correct = []

label_batches = []

for i, (X, y) in zip(range(val_flow.n // BATCH_SIZE), val_flow):

    pred = model.predict(X).ravel()

    predicted_batches.append(pred)

    correct = list((pred > 0.5) == y)

    all_correct.extend(correct)

    label_batches.append(y)

    print(f'Processed {len(all_correct)} images.')
print(f'Accuracy on the validation set: {np.round(100 * np.mean(all_correct), 2)}%.')