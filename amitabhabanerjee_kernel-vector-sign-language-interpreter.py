# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
try:

    from PIL import Image

except ImportError:

    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")
try:

    import keras

    from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

    from keras.models import Sequential, model_from_json

    from keras.preprocessing.image import img_to_array

except ImportError as exc:

    sys.exit("No keras")
try:

    import tensorflow as tf

except ImportError:

    sys.exit("Cannot import from tensorflow:")
try:

    from sklearn.model_selection import train_test_split

except ImportError as exc:

    sys.exit("Cannot import scikit")
class NetworkConstants():  # pylint: disable=too-few-public-methods

    """Constant values used as image and network parameters."""



    # Width of images passed to the network

    IMAGE_WIDTH: int = 200



    # Height of images passed to the network

    IMAGE_HEIGHT: int = 200



    # Currently set to 2 alphabet images and 1 background image class

    # Number of classes that the network can categorize

    NUM_CLASSES: int = 27



    # The fraction of images passed to the network during training that should

    # be used as a validation set. Range: 0 to 1

    VALIDATION_SPLIT: float = 0.1



    # The fraction of images passed to the network during training that should

    # be used as a test set. Range: 0 to 1

    TEST_SPLIT: float = 0.2



    # Number of epochs on which to train the network

    EPOCHS: int = 5

class SignLanguageRecognizer():

    """Recognize sign language hand signals using Vector's camera feed.



    A convolutional neural network is used to predict the hand signs.

    The network is built with a Keras Sequential model with a TensorFlow backend.

    """



    def __init__(self):

        self.training_images: np.ndarray = None

        self.training_labels: np.ndarray = None

        self.test_images: np.ndarray = None

        self.test_labels: np.ndarray = None

        self.model: keras.engine.sequential.Sequential = None

        self.graph: tf.python.framework.ops.Graph = tf.compat.v1.get_default_graph()

    

    def load_datasets(self, dataset_root_folder: str) -> None:

        """Load the training and test datasets required to train the model.

        A sample dataset is included in the project ("dataset.zip"). Unzip the

        folder to use it to train the model.



        .. code-block:: python



            recognizer = SignLanguageRecognizer()

            recognizer.load_datasets("/path/to/dataset_root_folder")

        """



        if not dataset_root_folder:

            sys.exit("Cannot load dataset. Provide valid path with `--dataset_root_folder`")



        images = []

        labels = []



        for filename in os.listdir(dataset_root_folder):

            if filename.endswith(".png") and not filename.startswith("."):

                # Read black and white image

                image = Image.open(os.path.join(dataset_root_folder, filename))

                # Convert image to an array with shape (image_width, image_height, 1)

                image_data = img_to_array(image)

                images.append(image_data)



                label = filename[0]

                if filename.startswith("background"):

                    # Use the last class to denote an unknown/background image

                    label = NetworkConstants.NUM_CLASSES - 1

                else:

                    # Use ordinal value offsets to denote labels for all alphabets

                    label = ord(label) - 97

                labels.append(label)



        # Normalize the image data

        images = np.array(images, dtype="float") / 255.0

        # Convert labels to a numpy array

        labels = np.array(labels)



        # Split data read in to training and test segments

        self.training_images, self.test_images, self.training_labels, self.test_labels = train_test_split(images, labels, 

                                                                                                          test_size=NetworkConstants.TEST_SPLIT)



        # Convert array of labels in to binary classs matrix

        self.training_labels = keras.utils.to_categorical(self.training_labels, num_classes=NetworkConstants.NUM_CLASSES)

        self.test_labels = keras.utils.to_categorical(self.test_labels, num_classes=NetworkConstants.NUM_CLASSES)

    

    def create_model(self) -> None:

        """Creates a convolutional neural network model with the following architecture:



        ConvLayer -> MaxPoolLayer -> ConvLayer -> MaxPoolLayer -> ConvLayer ->

        Dropout -> Flatten -> Dense -> Dropout -> Dense



        .. code-block:: python



            recognizer = SignLanguageRecognizer()

            recognizer.load_datasets("/path/to/dataset_root_folder")

            recognizer.create_model()

        """

        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(NetworkConstants.IMAGE_WIDTH, 

                                                                                      NetworkConstants.IMAGE_HEIGHT, 1)))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))



        self.model.add(Dropout(0.25))

        self.model.add(Flatten())



        self.model.add(Dense(128, activation="relu"))

        self.model.add(Dropout(0.5))

        self.model.add(Dense(NetworkConstants.NUM_CLASSES, activation="softmax"))



        self.model.compile(loss=keras.losses.categorical_crossentropy,

                           optimizer=keras.optimizers.Adadelta(),

                           metrics=['accuracy'])

        

    def train_model(self, epochs: int = NetworkConstants.EPOCHS, verbosity: int = 1) -> None:

        """Trains the model off of the training and test data provided



        .. code-block:: python



            recognizer = SignLanguageRecognizer()

            recognizer.load_datasets("/path/to/dataset_root_folder")

            recognizer.create_model()

            recognizer.train_model()

        """

        if self.training_images.size == 0 or self.training_labels.size == 0:

            sys.exit("Training dataset is empty. Build a dataset with `data_gen.py` before training the model.")

        self.model.fit(self.training_images,

                       self.training_labels,

                       epochs=epochs,

                       verbose=verbosity,

                       validation_split=NetworkConstants.VALIDATION_SPLIT)

    

    def load_model(self, model_config_filename: str, model_weights_filename: str) -> None:

        """Loads a saved model's config and weights to rebuild the model rather than create

        a new model and re-train.



        .. code-block:: python



            recognizer = SignLanguageRecognizer()

            recognizer.load_model("/path/to/model_config_filename", "/path/to/model_weights_filename")

        """

        if not model_config_filename or not model_weights_filename:

            sys.exit("Cannot load model. Provide valid paths with --model_config and --model_weights.")

        json_model = None

        with open(model_config_filename, "r") as file:

            json_model = file.read()

        # Load the network architecture

        self.model = model_from_json(json_model)

        # Load the weight information and apply it to the model

        self.model.load_weights(model_weights_filename)



        self.model.compile(loss=keras.losses.categorical_crossentropy,

                           optimizer=keras.optimizers.Adadelta(),

                           metrics=['accuracy'])

    def save_model(self, model_config_filename: str, model_weights_filename: str) -> None:

        """Saves a model's config and weights for latter use.



        .. code-block:: python



            recognizer = SignLanguageRecognizer()

            recognizer.load_datasets(args.dataset_root_folder)

            recognizer.create_model()

            recognizer.train_model()

            recognizer.save_model("/path/to/model_config_filename", "/path/to/model_weights_filename")

        """

        json_model = self.model.to_json()

        # Save the network architecture

        with open(model_config_filename, "w") as file:

            file.write(json_model)

        # Save the model's assigned weights

        self.model.save_weights(model_weights_filename)

        

    
recognizer = SignLanguageRecognizer()

recognizer.load_datasets('../input/training-a-robot-to-understand-sign-language/signlanguage/signlanguage/')

recognizer.create_model()

recognizer.train_model()



test_score = recognizer.model.evaluate(recognizer.test_images, recognizer.test_labels, verbose=1)

print(f"{recognizer.model.metrics_names[1].capitalize()}: {test_score[1] * 100}%")