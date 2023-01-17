from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



import pandas as pd

import numpy as np

from tqdm import tqdm



import tensorflow as tf

np.random.seed(0)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

INPUT_SOURCE = '/kaggle/input/dogs-vs-cats'

FAST_RUN = False

IMAGE_WIDTH=224

IMAGE_HEIGHT=224

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3

FILE_PATH = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'
loaded = tf.keras.models.load_model("/kaggle/input/project-ai-ankk/DogVsCatModelv1")
!rm -rf "./train"

!unzip -q "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
# Preparing the data

filenames = os.listdir("./train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df['category'] = df["category"].replace({0: 'cat', 1: 'dog'}) 
# Splitting the data

train_df, val_df = train_test_split(df, test_size=.2, stratify=df["category"], random_state=42)

train_df = train_df.reset_index()

val_df = val_df.reset_index()
batch_size=32

total_train = train_df.shape[0]

total_validate = val_df.shape[0]
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    val_df, 

    "./train/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size,

    shuffle = False

)
Y_pred = loaded.predict_generator(validation_generator)

y_pred_pob = np.max(Y_pred,axis=-1)

y_pred = np.argmax(Y_pred, axis=-1)
acc = sum(y_pred==validation_generator.classes)/total_validate

print('Accuracy: %s' % acc)
wrong_index_array = []

right_index_array = []



for i in range(total_validate):

    if y_pred[i] != validation_generator.classes[i]:

        wrong_index_array.append(i)

    else:

        right_index_array.append(i)
len(wrong_index_array)
for i in range(10):

    ind = np.random.choice(right_index_array)

    image = load_img("./train/" + val_df['filename'][ind])

    plt.imshow(image)

    plt.title(("Cat: " if y_pred[ind] == 0 else "Dog: ") + str(y_pred_pob[ind]))

    plt.axis("off")

    plt.show()
for i in range(10):

    ind = np.random.choice(wrong_index_array)

    image = load_img("./train/" + val_df['filename'][ind])

    plt.imshow(image)

    plt.title(("Cat: " if y_pred[ind] == 0 else "Dog: ") + str(y_pred_pob[ind]))

    plt.axis("off")

    plt.show()