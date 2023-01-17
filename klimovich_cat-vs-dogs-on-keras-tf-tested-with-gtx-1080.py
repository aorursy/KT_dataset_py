import sys

import os

from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt



from keras.layers import *

from keras.optimizers import *

from keras.applications import *

from keras.models import Model, model_from_json

from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import backend as k



from sklearn.model_selection import train_test_split

from IPython.display import clear_output

%matplotlib inline
train_dir = "../input/train/"

test_dir = "../input/test/"

model_path = os.path.join("models", "xception")

top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')



train_files = os.listdir(train_dir)

train_paths = list(map(lambda x: os.path.join(train_dir, x), train_files))

test_files = os.listdir(test_dir)

test_paths = list(map(lambda x: os.path.join(test_dir, x), test_files))
cat_example_file = next(filter(lambda x: x.startswith("cat"), train_files))

dog_example_file = next(filter(lambda x: x.startswith("dog"), train_files))

cat_example = plt.imread(os.path.join(train_dir, cat_example_file))

dog_example = plt.imread(os.path.join(train_dir, dog_example_file))

fig = plt.figure(figsize=(12, 6))

fig.add_subplot(1, 2, 1)

plt.title('Cat')

plt.imshow(cat_example)

fig.add_subplot(1, 2, 2)

plt.title('Dog')

plt.imshow(dog_example)

plt.show()
train_part_files, validation_part_files, train_part_paths, validation_part_paths = train_test_split(

    train_files, train_paths, train_size=0.8, random_state=123)

train_part_ys = np.array(list(map(lambda x: 0 if x.startswith('cat') else 1, train_part_files)))

validation_part_ys = np.array(list(map(lambda x: 0 if x.startswith('cat') else 1, validation_part_files)))
class FileListIterator(Iterator):

    """Iterator capable of reading images located on disk by specified pathes.

    Arguments:

            filenames: Paths to the images.

                    Each subdirectory in this directory will be

                    considered to contain images from one class,

                    or alternatively you could specify class subdirectories

                    via the `classes` argument.

            y: Numpy array of targets data.

            image_data_generator: Instance of `ImageDataGenerator`

                    to use for random transformations and normalization.

            target_size: tuple of integers, dimensions to resize input images to.

            color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.            

            batch_size: Integer, size of a batch.

            shuffle: Boolean, whether to shuffle the data between epochs.

            seed: Random seed for data shuffling.

            data_format: String, one of `channels_first`, `channels_last`.

            save_to_dir: Optional directory where to save the pictures

                    being yielded, in a viewable format. This is useful

                    for visualizing the random transformations being

                    applied, for debugging purposes.

            save_prefix: String prefix to use for saving sample

                    images (if `save_to_dir` is set).

            save_format: Format to use for saving sample images

                    (if `save_to_dir` is set).

    """



    def __init__(self,

                 filenames,

                 y,

                 image_data_generator,

                 target_size=(256, 256),

                 color_mode='rgb',                 

                 class_mode='categorical',

                 batch_size=32,

                 shuffle=True,

                 seed=None,

                 data_format=None,

                 save_to_dir=None,

                 save_prefix='',

                 save_format='jpeg'):

        if data_format is None:

            data_format = K.image_data_format()        

        self.image_data_generator = image_data_generator

        self.target_size = tuple(target_size)

        if color_mode not in {'rgb', 'grayscale'}:

            raise ValueError('Invalid color mode:', color_mode,

                             '; expected "rgb" or "grayscale".')

        self.color_mode = color_mode

        self.data_format = data_format        

        if self.color_mode == 'rgb':

            if self.data_format == 'channels_last':

                self.image_shape = self.target_size + (3,)                

            else:

                self.image_shape = (3,) + self.target_size

        else:

            if self.data_format == 'channels_last':

                self.image_shape = self.target_size + (1,)

            else:

                self.image_shape = (1,) + self.target_size

        self.y = y        

        self.save_to_dir = save_to_dir

        self.save_prefix = save_prefix

        self.save_format = save_format



        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        

        self.filenames = filenames     

        self.nb_sample = len(filenames)

        super(FileListIterator, self).__init__(self.nb_sample, batch_size, shuffle,

                                                seed)



        

    def next(self):

        """For python 2.x.

        Returns:

                The next batch.

        """

        with self.lock:

            index_array, current_index, current_batch_size = next(

                    self.index_generator)

        # The transformation of images is not under thread lock

        # so it can be done in parallel

        batch_x = np.zeros(

                (current_batch_size,) + self.image_shape, dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'

        # build batch of image data

        for i, j in enumerate(index_array):

            fname = self.filenames[j]

            img = load_img(fname, grayscale=grayscale, target_size=self.target_size)

            x = img_to_array(img, data_format=self.data_format)

            x = self.image_data_generator.random_transform(x)

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes

        if self.save_to_dir:

            for i in range(current_batch_size):

                img = array_to_img(batch_x[i], self.data_format, scale=True)

                fname = '{prefix}_{index}_{hash}.{format}'.format(

                        prefix=self.save_prefix,

                        index=current_index + i,

                        hash=np.random.randint(1e4),

                        format=self.save_format)

                img.save(os.path.join(self.save_to_dir, fname))

        # build batch of labels

        if self.y is None:

            return batch_x

        batch_y = self.y[index_array]        

        return batch_x, batch_y
try:

    img_width, img_height = 299, 299



    # learning process parameters

    batch_size = 32

    train_epochs = 5



    # sgd parameters

    learn_rate = 1e-4

    momentum = .9



    # take base model with weights pre-trained using imagenet dataset

    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)



    # set model ending to fit current problem (binary classification)

    base_output = base_model.output

    avg_pool_base_output = GlobalAveragePooling2D()(base_output)

    predictions = Dense(1, activation='sigmoid')(avg_pool_base_output)



    # construct keras model object passing input and output layers

    model = Model(base_model.input, predictions)



    # do not train layers of the based model which are already pre-trained

    for layer in base_model.layers:

        layer.trainable = False

except:

    # kaggle doesn't allow loading weights

    pass
try:

    model_json = model.to_json()

    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:

        json_file.write(model_json)

    

    # to avoid shoing error message

    clear_output()

except:

    # won't work because of error in cell 6

    pass
try:

    data_generator = ImageDataGenerator(rescale=1. / 255)



    train_generator = FileListIterator(train_part_paths, train_part_ys, data_generator,

                                       target_size=(img_width, img_height), batch_size=batch_size)



    validation_generator = FileListIterator(validation_part_paths, validation_part_ys, data_generator, 

                                            target_size=(img_width, img_height), batch_size=batch_size)



    model.compile(optimizer='nadam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])



    callbacks_list = [

        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=3, save_best_only=True),

        EarlyStopping(monitor='val_acc', patience=5, verbose=3)

    ]



    model.fit_generator(train_generator,

                        samples_per_epoch=train_generator.nb_sample,

                        nb_epoch=train_epochs,

                        validation_data=validation_generator,

                        nb_val_samples=validation_generator.nb_sample,

                        callbacks=callbacks_list)



    # to avoid shoing error message

    clear_output()    

except:

    # won't work because of error in cell 6

    pass
try:

    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'r') as json_file:

        model_json = json_file.read()



    model = model_from_json(model_json)

    model.load_weights(top_weights_path)

except:

    # won't work because of error in cell 6

    pass
try:

    predictions = []

    batch_size = 128

    for i in range(0, len(test_paths), batch_size):

        batch_paths = test_paths[i:i + batch_size]

        batch_x = np.zeros((len(batch_paths),) + (img_width, img_height) + (3,), dtype=K.floatx())

        for j, img_path in enumerate(batch_paths):

            img = load_img(batch_paths[j], grayscale=False, target_size=(img_width, img_height))

            img_array = img_to_array(img, data_format=None)        

            batch_x[j] = data_generator.standardize(img_array)

        ys = model.predict(batch_x)

        predictions.extend(list(zip(batch_paths, ys[:, 0])))

        clear_output(wait=True)

        print("{}/{}".format(len(predictions), len(test_paths)))    

except:

    # won't work because of error in cell 6

    pass
out_path = str(datetime.now()).replace(":", "_").replace(" ", "_").split('.')[0] + ".csv" 

with open(out_path, "w") as out: 

    out.write('id,label\n')

    for fname, val in predictions:        

        out.write('{},{}\n'.format(fname.split('/')[1].split('.')[0], val))

print("done {}".format(out_path))
k.clear_session()