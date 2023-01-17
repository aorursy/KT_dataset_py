import os
import random
import warnings
import numpy as np
import skimage.io
import skimage.transform

from matplotlib import pyplot as plt
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from yellowbrick.target import ClassBalance
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
np.random.seed(1337)
def generator_error(generator):
    """To catch errors when reading in data.

    :param generator: gernerator ->
    :return one
    """

    while True:
        try:
            x, y = next(generator)
            yield x, y
        except OSError:
            pass


def label_to_categorical(y):
    """Convert text to categorical.

    :param y: # todo
    :return: x, y -> arrays
    """

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y)

    return y


def files_per_directory(base_dir, categories):
    """Counts the files in the directory.

    :param base_dir: string -> The main directory in where the images
        are stored.
    :param categories: list | tuple -> This must contain all folders
        (categories) from where the images are to be read.
    :return: list -> files_per_directory
    """
    files_per_dir = []

    for category in categories:
        files = os.listdir(os.path.join(base_dir, category))
        files_per_dir.append(len(files))

    return files_per_dir


def load_images(base_dir, categories, shuffle=False, resize=False,
                size=(150, 150, 3), output_range=1000):
    """Load the images, [calculates the size of the categories].

    :param base_dir: string -> The main directory in where the images
        are stored.
    :param categories: list | tuple -> This must contain all folders
        (categories) from where the images are to be read.
    :param shuffle: bool -> Shuffles the images. Standard is 'False'.
    :param resize: bool -> should the images all be changed to a
        certain size? Standard is 'False'.
    :param size: list | tuple -> (width, height, channels).
    :param output_range: integer -> shows one output all images.
    :return: x, y  -> arrays (data, label)
    """

    file_list = []
    x = []
    y = []
    images_counter = 0
    errors_counter = 0

    for category in categories:
        files = os.listdir(os.path.join(base_dir, category))
        for file in files:
            file_list.append((os.path.join(base_dir, category, file), category))

    if shuffle:
        random.shuffle(file_list)

    for ix, (data, label) in enumerate(file_list, start=1):
        try:
            image = skimage.io.imread(data)
            images_counter = ix
        except OSError:
            errors_counter += 1
            continue
        if resize:
            image = skimage.transform.resize(image, size, mode='reflect')
        x.append(image)
        y.append(label)

        if ix % output_range == 0:
            print('Images loaded: {} | Errors: {}'.format(ix, errors_counter))

    print('Total images loaded: {} | Total errors: {}'.format(images_counter,
                                                              errors_counter))

    x = np.array(x)
    y = np.array(y)

    return x, y


def statistic_diversification(files_per_dir, categories):
    """Print the diversification of the data in the folders.

    :param files_per_dir: list | tuple ->  Number of files in each
        category.
    :param categories: list | tuple -> This must contain all folders
        (categories) from where the data is to be read.
    :return: None
    """

    for category, files in zip(categories, files_per_dir):
        print('{}: {} files'.format(category, files))


def statistic_samples_size_shape(array):
    """Show Samples | Batch size | Shape.

    :param array: # todo
    :return: None
    """

    print('''Samples {} | Batch size: {} | Shape {}'''.format(
        array.samples, array.batch_size, array.image_shape
    ))


def statistic_val_loss_acc(result):
    """Show val_loss | val_acc

    :param result: # todo
    :return: None
    """
    val_loss = min(result.history.get('val_loss'))
    val_acc = max(result.history.get('val_acc'))

    print('val_loss: {} | val_acc: {}'.format(
        val_loss, val_acc
    ))


def statistic_test_loss_acc(model, x, y=None, error=False):
    """Displays the result of the test data.

    This does not mean the validation data in Deep Learning! Some say
    you should have three sets of data: Training data, validation data
    and test data.

    :param model: model
    :param x: array -> Data
    :param y: array -> Labels (N)
    :param error: bool -> If an error occurs while reading, you should
    try it with 'True'. Maybe the error was already considered in the
    code and intercepted. Standard is 'False'

    :return: None
    """

    if y is not None:
        loss, acc = model.evaluate(x, y)
        print('test_loss: {} | test_acc: {}'.format(loss, acc))
    else:
        steps = x.samples // x.batch_size
        if error:
            loss, acc = model.evaluate_generator(generator_error(x),
                                                 steps=steps)
        else:
            steps = x.samples // x.batch_size
            loss, acc = model.evaluate_generator(x, steps=steps)

        print('test_loss: {} | test_acc: {}'.format(loss, acc))


def plot_diversification(files_per_dir, categories):
    """Plot the diversification of the data in the folders.

    :param files_per_dir: list | tuple ->  Number of files in each
        category.
    :param categories: list | tuple -> This must contain all folders
        (categories) from where the data is to be read.
    :return: None
    """

    plt.bar([x for x in range(len(files_per_dir))],
            files_per_dir, tick_label=categories)
    plt.show()


def plot_pic_per_category(base_dir, categories, size=(150, 150, 3)):
    """Displays one image from each category.

    :param base_dir: string -> The main directory in where the data is
        stored.
    :param categories: list | tuple -> This must contain all folders
        (categories) from where the data is to be read.
    :param size: list | tuple -> (width, height, channels).
    :return: None
    """

    fig, axes = plt.subplots(nrows=1, ncols=len(categories), figsize=(15, 3))

    for ix, category in enumerate(categories):
        directory = os.path.join(base_dir, category)
        first_image = os.listdir(directory)[0]

        try:
            image = skimage.io.imread(os.path.join(directory, first_image))
        except OSError:
            continue

        image = skimage.transform.resize(image, size, mode='reflect')
        axes[ix].imshow(image, resample=True)
        axes[ix].set_title(category)

    plt.show()


def plot_loss_acc(result, linestyle='-', result2=None, linestyle2=':'):
    acc = result.history['acc']
    val_acc = result.history['val_acc']
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    epochs = range(1, len(acc) + 1)

    if result2:
        acc2 = result2.history['acc']
        val_acc2 = result2.history['val_acc']
        loss2 = result2.history['loss']
        val_loss2 = result2.history['val_loss']
        epochs2 = range(1, len(acc2) + 1)
    else:
        acc2 = 1
        val_acc2 = None
        loss2 = None
        val_loss2 = None
        epochs2 = None

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(epochs, loss, color='green', linestyle=linestyle, marker='o',
             markersize=2, label='Training')
    plt.plot(epochs, val_loss, color='red', linestyle=linestyle, marker='o',
             markersize=2, label='Validation')
    if result2:
        plt.plot(epochs2, loss2, color='green', linestyle=linestyle2,
                 marker='o', markersize=2, label='Training')
        plt.plot(epochs2, val_loss2, color='red', linestyle=linestyle2,
                 marker='o', markersize=2, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(epochs, acc, color='green', linestyle=linestyle, marker='o',
             markersize=2, label='Training')
    plt.plot(epochs, val_acc, color='red', linestyle=linestyle, marker='o',
             markersize=2, label='Validation')
    if result2:
        plt.plot(epochs2, acc2, color='green', linestyle=linestyle2,
                 marker='o', markersize=2, label='Training')
        plt.plot(epochs2, val_acc2, color='red', linestyle=linestyle2,
                 marker='o', markersize=2, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
train_dir = '../input/dataset/dataset_updated/training_set/'
test_dir = '../input/dataset/dataset_updated/validation_set/'
train_categories = [x for x in os.listdir(train_dir) if x[0] is not '.']
test_categories = [x for x in os.listdir(test_dir) if x[0] is not '.']
train_x, train_y = load_images(train_dir, train_categories, resize=True)
test_x, test_y = load_images(test_dir, test_categories, resize=True, output_range=100)
plot_pic_per_category(train_dir, train_categories)
train_files_per_dir = files_per_directory(train_dir, train_categories)
statistic_diversification(train_files_per_dir, train_categories)
plot_diversification(train_files_per_dir, train_categories)
test_files_per_dir = files_per_directory(test_dir, train_categories)
statistic_diversification(test_files_per_dir, test_categories)
plot_diversification(test_files_per_dir, test_categories)
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=1337, stratify=train_y
)
train_viz = ClassBalance()
train_viz.fit(train_y)
train_viz.poof()
val_viz = ClassBalance()
val_viz.fit(val_y)
val_viz.poof()
epochs=150
patience=10
train_y = label_to_categorical(train_y)
val_y = label_to_categorical(val_y)
test_y = label_to_categorical(test_y)
train_datagen = ImageDataGenerator(
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    horizontal_flip=True,
)
train_datagen.fit(train_x)

train_generator = train_datagen.flow(train_x, 
                                     train_y, 
                                     batch_size=32, 
                                     seed=1337)
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), 
                        activation='relu', 
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

result_01 = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=(val_x, val_y),
    callbacks=[
        EarlyStopping(
            monitor='val_loss', 
            mode='auto', 
            verbose=1, 
            patience=patience,
        ), 
        ModelCheckpoint(
            'best_model_01.h5', 
            monitor='val_loss', 
            mode='auto', 
            save_best_only=True, 
            verbose=1
        )
    ]
)
statistic_val_loss_acc(result_01)
plot_loss_acc(result_01)
model_01 = load_model('best_model_01.h5')
statistic_test_loss_acc(model_01, test_x, test_y)