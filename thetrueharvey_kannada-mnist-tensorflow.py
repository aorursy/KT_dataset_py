import os

import shutil

import numpy  as np

import pandas as pd



from PIL import Image

from cv2 import imread, resize, IMREAD_GRAYSCALE



from tensorflow.keras.utils import Sequence

from tensorflow             import convert_to_tensor as to_T



from tensorflow.keras.models import load_model

# Bounding Box

def bbox(image):

    """

    Determines the bounding boxes for images to remove empty space where possible

    :param image:

    :return:

    """

    HEIGHT = image.shape[0]

    WIDTH  = image.shape[1]



    for i in range(image.shape[1]):

        if (image[:, i] > 0).sum() >= 1:

            x_min = i - 1 if (i > 1) else 0

            break



    for i in reversed(range(image.shape[1])):

        if (image[:, i] > 0).sum() >= 1:

            x_max = i + 2 if (i < WIDTH - 2) else WIDTH

            break



    for i in range(image.shape[0]):

        if (image[i] > 0).sum() >= 1:

            y_min = i - 1 if (i > 1) else 0

            break



    for i in reversed(range(image.shape[0])):

        if (image[i] > 0).sum() >= 1:

            y_max = i + 2 if (i < HEIGHT - 2) else HEIGHT

            break



    return x_min, y_min, x_max, y_max



    return x_min, y_min, x_max, y_max
#%% Dataset class

class Dataset(Sequence):

    def __init__( self

                , image_list

                , batch_size

                , dimensions

                ):

        """

        Creates a Keras Sequence class that serves data to the model

        """



        # Class attributes

        self.image_list = image_list

        self.batch_size = batch_size

        self.dimensions = dimensions



        # Initialize the list

        self.on_epoch_end()



    def __len__(self):

        """

        Number of batches available in the dataset

        """

        return int(np.ceil(len(self.image_list) / self.batch_size))



    def __getitem__(self, index):

        """

        Generate a single batch of data

        """

        

        # Indices of samples in the dataset

        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]



        # Associated images and labels

        images = [self.image_list[k] for k in indices]



        # Data generation

        X = self.__data_generation(images)



        return X



    def __data_generation(self, images):

        """

        Retrieve the appropriate image and process as necessary for training

        """

        # Empty storage

        X = np.empty((len(images), self.dimensions["height"], self.dimensions["width"], self.dimensions["channels"]))



        # Loop through each ID

        for idx, image in enumerate(images):

            # Load the image

            image = imread(image, IMREAD_GRAYSCALE)



            # Load and resize

            image = resize( image

                          , (self.dimensions["width"], self.dimensions["height"])

                          )

            image = np.expand_dims(image, axis=2)



            # Append to storage

            X[idx,] = image



        # Normalize X

        X = X / 255.0



        return to_T(X)



    def on_epoch_end(self):

        """

        Update the dataset at the end of an epoch

        """

        self.indices = np.arange(len(self.image_list))

# Load the dataset

df = pd.read_csv("../input/Kannada-MNIST/test.csv")



# Label and image ID storage

ids        = []

image_list = []



# Make a directory for storage

os.makedirs("images")



# Loop through each row

for i,row in df.iterrows():

    # Get the data components

    id    = "{}".format(i)

    img   = np.reshape(row[1:].values, newshape=(28,28)).astype(np.uint8)



    # Remove empty space

    x_min, y_min, x_max, y_max = bbox(img)

    img = img[y_min:y_max, x_min:x_max]



    # Check again

    x_min, y_min, x_max, y_max = bbox(img)

    img = img[y_min:y_max, x_min:x_max]



    # Convert and reshape

    img = Image.fromarray(img)

    img = img.resize((28,28), Image.ANTIALIAS)



    # Write to storage

    img.save("images/{}.png".format(id))

    ids.append(id)

    image_list.append("images/{}.png".format(id))

    

# Save the IDs to a DataFrame and write to CSV

test_df = pd.DataFrame({"id": ids, "label": 0}) # We will update this later
ds = Dataset( image_list=image_list

            , batch_size=512

            , dimensions={"height":28, "width":28, "channels":1}

            )
net = load_model("../input/kannada-mnist-tensorflow-2/CustomNetV2.hdf5")
predictions = net.predict(ds).argmax(-1)
# Update the submission

test_df["label"] = predictions
test_df.head()
test_df.to_csv("submission.csv",index=False)
shutil.rmtree("images")