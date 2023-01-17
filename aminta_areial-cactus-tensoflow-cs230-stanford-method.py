import random

import os



import sklearn.utils

from tqdm import tqdm, tqdm_notebook

import pandas as pd

import cv2 as cv



from tqdm import tqdm



import tensorflow as tf

#tf.enable_eager_execution()

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, DepthwiseConv2D, Flatten

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping



import json

import logging



import numpy as np

import matplotlib.pyplot as plt 
data_dir = r'../input'

dataset_dir = os.path.join(data_dir, r'train/train')

csv_dir = os.path.join(data_dir , 'train.csv')

print(os.getcwd())

print(dataset_dir)
# I set all seed as 1372



from numpy.random import seed

seed(1372)

from tensorflow import set_random_seed

set_random_seed(1372)



#df = sklearn.utils.shuffle(df,random_state=1372)
def resize_and_save(filename, input_dir, output_dir, size=32):

    """Resize the image contained in `filename` and save it to the `output_dir`"""

    image = Image.open(os.path.join(input_dir, filename))

    # No resize Need for this dataset

    # Use bilinear interpolation instead of the default "nearest neighbor" method

    # image = image.resize((size, size), Image.BILINEAR)

    image.save(os.path.join(output_dir, filename)) # linux => / windows => \\
class Params():

    """Class that loads hyperparameters from a json file.



    Example:

    ```

    params = Params(json_path)

    print(params.learning_rate)

    params.learning_rate = 0.5  # change the value of learning_rate in params

    ```

    """



    def __init__(self, json_path):

        self.update(json_path)



    def save(self, json_path):

        """Saves parameters to json file"""

        with open(json_path, 'w') as f:

            json.dump(self.__dict__, f, indent=4)



    def update(self, json_path):

        """Loads parameters from json file"""

        with open(json_path) as f:

            params = json.load(f)

            self.__dict__.update(params)



    @property

    def dict(self):

        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""

        return self.__dict__





def set_logger(log_path):

    """Sets the logger to log info in terminal and file `log_path`.



    In general, it is useful to have a logger so that every output to the terminal is saved

    in a permanent file. Here we save it to `model_dir/train.log`.



    Example:

    ```

    logging.info("Starting training...")

    ```



    Args:

        log_path: (string) where to log

    """

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)



    if not logger.handlers:

        # Logging to a file

        file_handler = logging.FileHandler(log_path)

        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))

        logger.addHandler(file_handler)



        # Logging to console

        stream_handler = logging.StreamHandler()

        stream_handler.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(stream_handler)





def save_dict_to_json(d, json_path):

    """Saves dict of floats in json file



    Args:

        d: (dict) of float-castable values (np.float, int, float, etc.)

        json_path: (string) path to json file

    """

    with open(json_path, 'w') as f:

        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )

        d = {k: float(v) for k, v in d.items()}

        json.dump(d, f, indent=4)

# Get the filenames in each directory (train and test)

df = pd.read_csv(csv_dir)

df['id'] = dataset_dir + '/' + df['id'].astype(str)

filenames = df['id']

labels = df['has_cactus'].astype(np.float32)





# Make sure to always shuffle with a fixed seed so that the split is reproducible

df = sklearn.utils.shuffle(df,random_state=1372)

df = df.reset_index(drop=True)
print('sample filename ',filenames[0])

print('sample label (1 = exsit) , (0 = dosent exist any cactus) ',labels[0],type(labels[0]))
def _parse_function(filename, label, size):

    """Obtain the image from the filename (for both training and validation).



    The following operations are applied:

        - Decode the image from jpeg format

        - Convert to float and to range [0, 1]

    """

    image_string = tf.read_file(filename)



    # Don't use tf.image.decode_image, or the output shape will be undefined

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)



    # This will convert to float values in [0, 1]

    image = tf.image.convert_image_dtype(image_decoded, tf.float32)



    resized_image = tf.image.resize_images(image, [size, size])



    return resized_image, label
def train_preprocess(image, label, use_random_flip):

    """Image preprocessing for training.



    Apply the following operations:

        - Horizontally flip the image with probability 1/2

        - Apply random brightness and saturation

    """

    if use_random_flip:

        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=25.0 / 255.0)

    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)



    # Make sure the image is still in [0, 1]

    image = tf.clip_by_value(image, 0.0, 1.0)



    return image, label
def input_fn(is_training, filenames, labels, params):

    """Input function for the dataset.



        Args:

        is_training: (bool) whether to use the train or test pipeline.

                     At training, we shuffle the data and have multiple epochs

        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]

        labels: (list) corresponding list of labels

        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """

    num_samples = len(filenames)

    assert len(filenames) == len(labels), "Filenames and labels should have same length"



    # Create a Dataset serving batches of images and labels

    # We don't repeat for multiple epochs because we always train and evaluate for one epoch

    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)

    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)



    if is_training:

        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))

            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling

            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)

            .map(train_fn, num_parallel_calls=params.num_parallel_calls)

            .batch(params.batch_size)

            .repeat()

            .prefetch(32)  # make sure you always have one batch ready to serve

        )

    else:

        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))

            .map(parse_fn)

            .batch(params.batch_size)

            .repeat()

            .prefetch(32)  # make sure you always have one batch ready to serve

        )

        

    return dataset
with open("params.json", "w") as text_file:

    text_file.write("{\n"+

    "\"learning_rate\": 1.5e-3,"+

    "\"batch_size\": 64,"+

    "\"num_epochs\": 50,"+

    "\"image_size\": 32,"+

    "\"use_random_flip\": false,"+

    "\"num_labels\": 2,"+

    "\"num_parallel_calls\": 8,"+

    "\"save_summary_steps\": 1"+

    "\n}")

    

paramPath = r'./'

json_path = os.path.join(paramPath , 'params.json')

assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

split = int(len(filenames)*0.15)

train_dataset = input_fn(True, filenames[:split], labels[:split], params)

valid_dataset = input_fn(False , filenames[split:], labels[split:], params)
iterator = train_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:

    one_batch = sess.run(next_element)

    print(one_batch[0].shape,' = 64 batch-size & 32x32x3 image')

    for i in range(3):

        plt.figure()

        sample = one_batch[0]

        label = one_batch[1]

        print(label[i])

        plt.imshow(sample[i])

        plt.grid(False)       
model = Sequential()

        

model.add(Conv2D(3, kernel_size = 3, activation = 'relu', input_shape = (32, 32, 3)))



model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 32, kernel_size = 1, activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 128, kernel_size = 1, activation = 'relu'))

model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



#model.add(GlobalAveragePooling2D())

model.add(Flatten())



model.add(Dense(512, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(128, activation = 'elu'))



model.add(Dense(1, activation = 'sigmoid'))
Adam = tf.keras.optimizers.Adam(lr=params.learning_rate , amsgrad=True)

model.compile(optimizer = Adam, loss = tf.losses.log_loss, metrics = ['accuracy'])

model.summary()
file_path = 'weights-aerial-cactus.h5'



callbacks = [

        ModelCheckpoint(file_path, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max'),

        ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, mode = 'min', min_lr = 0.00001),

        EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 15, verbose = 1, restore_best_weights = True)

        ]
history = model.fit(train_dataset, validation_data=valid_dataset,

          epochs=50,verbose=True,

          steps_per_epoch=int((len(filenames) - split)/params.batch_size),

          validation_steps=int(split/params.batch_size),

           callbacks = callbacks)
model.load_weights(file_path)
def plot_training_curves(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    

    epochs = range(1, len(acc) + 1)

    

    plt.plot(epochs, loss, 'r', label='Training loss')

    plt.plot(epochs, val_loss, 'g', label='Validation loss')

    plt.title('Losses')

    plt.legend()

    plt.figure()

    

    plt.plot(epochs, acc, 'r', label='Training acc')

    plt.plot(epochs, val_acc, 'g', label='Validation acc')

    plt.title('Accuracies')

    plt.legend()

    plt.figure()

    

    plt.show()
plot_training_curves(history)
test_df = pd.read_csv('../input/sample_submission.csv')

X_test = []

images_test = test_df['id'].values



for img_id in tqdm_notebook(images_test):

    X_test.append(cv.imread('../input/test/test/' + img_id))

    

X_test = np.asarray(X_test)

X_test = X_test.astype('float32')

X_test /= 255



y_test_pred = model.predict_proba(X_test)



test_df['has_cactus'] = y_test_pred

test_df.to_csv('aerial-cactus-submission_1.csv', index = False)



for i in range(len(y_test_pred)):

    if y_test_pred[i][0] >= 0.5:

        y_test_pred[i][0] = 1.0

    else:

        y_test_pred[i][0] = 0.0

        

test_df['has_cactus'] = y_test_pred

test_df.to_csv('aerial-cactus-submission_2.csv', index = False)