# pip install tensorflow-gpu
import os

import glob

import cv2

from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import shuffle



from PIL import Image



from skimage.io import imread, imsave

from skimage.transform import resize 

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model

from tensorflow.keras import layers as L

from tensorflow.keras.applications import vgg16

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.optimizers import SGD, Adam, RMSprop



import tensorflow as tf

import tensorflow.keras.backend as K



import imgaug as ia

from imgaug import augmenters as iaa



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tf.__version__
def plotImages(artist, directory):

    """Plot 25 images of an artist."""

    multipleImages = glob.glob(directory)

    print(f"{artist} has {len(multipleImages)} images")

    plt.rcParams['figure.figsize'] = (12, 12)

    plt.subplots_adjust(wspace=0, hspace=0)

    i_ = 0

    for l in multipleImages[:25]:

        im = cv2.imread(l)

        

        # Shrink image if too large

        try:

            im = cv2.resize(im, (128, 128))

            plt.subplot(5, 5, i_+1) #.set_title(l)

            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

        except:

            print(l)

            pass

        i_ += 1

    plt.savefig(f"{artist}.jpg")
%config InlineBackend.figure_format="svg"

%matplotlib inline





# Reproducibility is importatnt. Always set the seed!

seed=1234

np.random.seed(seed)

tf.random.set_seed(seed)

ia.seed(seed)
df = pd.read_csv('/kaggle/input/best-artworks-of-all-time/artists.csv', index_col=0)
df.head()
genres = list(df['genre'].unique())

print(len(genres))

sorted(genres)
impressionists = df[(df['genre'] == 'Impressionism') | 

                    (df['genre'] == 'Realism,Impressionism') |

                    (df['genre'] == 'Impressionism,Post-Impressionism')]



impressionists = impressionists.sort_values(by=["paintings"], ascending=False)

impressionists
impressionist_artists = impressionists['name'].values.tolist()

impressionist_artists = [artist.replace(" ", "_") for artist in impressionist_artists] # match convention used in folder names

impressionist_artists
all_images_folders = glob.glob("/kaggle/input/best-artworks-of-all-time/images/images/*")

len(all_images_folders)
impressionist_folders = [dir_ for dir_ in all_images_folders if dir_.split('/')[-1] in  impressionist_artists]

print(len(impressionist_folders))

impressionist_folders
impressionist_images = []

for folder in impressionist_folders:

  impressionist_images.extend(glob.glob(folder + "/*.jpg", recursive=True))



len(impressionist_images)
impressioninst_df = pd.DataFrame(impressionist_images)

impressioninst_df.columns = ['filename']

impressioninst_df['genre'] = 'impressionist'

impressioninst_df.head()
artist_index = 0

folder = impressionist_folders[artist_index]

artist = folder.split("/")[-1]

plotImages(artist,  folder+'/*.jpg')
renaissance = df[(df['genre'] == 'Early Renaissance') | 

                    (df['genre'] == 'High Renaissance') |

                    (df['genre'] == 'Northern Renaissance') |

                    (df['genre'] == 'Proto Renaissance') |

                    (df['genre'] == 'High Renaissance,Mannerism')]



renaissance = renaissance.sort_values(by=["paintings"], ascending=False)

renaissance
renaissance_artists = list(renaissance['name'].unique())

print(len(renaissance_artists))
renaissance_artists = [artist.replace(" ", "_") for artist in renaissance_artists] # match convention used in folder names

renaissance_artists
renaissance_folders = [dir_ for dir_ in all_images_folders if dir_.split('/')[-1] in  renaissance_artists]

print(len(renaissance_folders))

renaissance_folders
renaissance_images = []

for folder in renaissance_folders:

  renaissance_images.extend(glob.glob(folder + "/*.jpg", recursive=True))



len(renaissance_images)
artist_index = 0

folder = renaissance_folders[artist_index]

artist = folder.split("/")[-1]

plotImages(artist,  folder+'/*.jpg')
renaissance_df = pd.DataFrame(renaissance_images)

renaissance_df.columns = ['filename']

renaissance_df['genre'] = 'renaissance'

renaissance_df.head()
train_df = pd.concat([renaissance_df, impressioninst_df.iloc[0:len(renaissance_df)]])

train_df = shuffle(train_df)

train_df.head()
train_df['genre'].describe()
# dimensions to consider for the images

img_rows, img_cols, img_channels = 224,224,3
VALIDATION_SPLIT = 0.4

datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=VALIDATION_SPLIT)
# batch size for training  

batch_size=16



# total number of classes in the dataset

nb_classes=2
train_generator=datagen.flow_from_dataframe(

    dataframe=train_df,

    x_col="filename",

    y_col="genre",

    subset="training",

    batch_size=batch_size,

    seed=42,

    shuffle=True,

    class_mode="categorical",

    target_size=(224,224),

    preprocessing_function=tf.keras.applications.vgg16.preprocess_input

)
valid_generator=datagen.flow_from_dataframe(

    dataframe=train_df,

    x_col="filename",

    y_col="genre",

    subset="validation",

    batch_size=batch_size,

    seed=42,

    shuffle=True,

    class_mode="categorical",

    target_size=(224,224),

    preprocessing_function=tf.keras.applications.vgg16.preprocess_input

)
# Augmentation sequence 

seq = iaa.OneOf([

    iaa.Fliplr(), # horizontal flips

    iaa.Affine(rotate=20), # roatation

    iaa.Multiply((1.2, 1.5))]) #random brightness
def get_base_model():

    base_model =  tf.keras.applications.vgg16.VGG16(input_shape=(img_rows, img_cols, img_channels), 

                       weights='imagenet', 

                       include_top=True)

    return base_model



# get the base model

base_model = get_base_model()
#  get the output of the second last dense layer 

base_model_output = base_model.layers[-2].output



# add new layers 

x = L.Dropout(0.5,name='drop2')(base_model_output)

output = L.Dense(nb_classes, activation='softmax', name='fc3')(x)