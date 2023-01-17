# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Constants for FER2013 dataset

FER2013_PATH = "/kaggle/input/facialexpressionrecognition/fer2013.csv"

FER2013_WIDTH = 48

FER2013_HEIGHT = 48
data = pd.read_csv(FER2013_PATH)

data.head()
data.info()
data["Usage"].value_counts()
# Seperate training and public/private test data

data_publ_test = data[data.Usage=="PublicTest"]

data_priv_test = data[data.Usage=="PrivateTest"]

data = data[data.Usage=="Training"]
Emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  # indices 0 to 6
data["emotion"].value_counts(sort=False)
def fer2013_show_instance(index):

    """Shows the image and the emotion label of the index's instance."""

    image = np.reshape(data.at[index, "pixels"].split(" "), (FER2013_WIDTH, FER2013_HEIGHT)).astype("float")

    image -= np.mean(image)

    image /= np.std(image)

    print(Emotions[data.at[index, "emotion"]])

    plt.imshow(image, cmap="gray")
fer2013_show_instance(np.random.randint(90,len(data)))
def fer2013_to_X():

    """Transforms the (blank separated) pixel strings in the DataFrame to an 3-dimensional array 

    (1st dim: instances, 2nd and 3rd dims represent 2D image)."""

    

    X = []

    pixels_list = data["pixels"].values

    

    for pixels in pixels_list:

        single_image = np.reshape(pixels.split(" "), (FER2013_WIDTH, FER2013_HEIGHT)).astype("float")

        X.append(single_image)

        

    # Convert list to 4D array:

    X = np.expand_dims(np.array(X), -1)

    

    # Normalize image data:

    X -= np.mean(X, axis=0)

    X /= np.std(X, axis=0)

    

    return X

# Get features (image data)

X = fer2013_to_X()

X.shape
# Get labels (one-hot encoded)

y = pd.get_dummies(data['emotion']).values

y.shape
# Save data

np.save("fer2013_X", X)

np.save("fer2013_y", y)