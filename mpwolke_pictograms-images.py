# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import os

import cv2 as cv

import matplotlib.pyplot as plt

import seaborn as sns

import skimage

import skimage.io



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
IMAGE_PATH = '..//input//art-pictogram//pictogram//'

IMAGE_WIDTH = 64

IMAGE_HEIGHT = 64

IMAGE_CHANNELS = 1

RANDOM_STATE = 42
os.listdir("..//input//art-pictogram")
image_files = list(os.listdir(IMAGE_PATH))

print("Number of image files: {}".format(len(image_files)))
def create_file_name(x):

    file_name = f"input_{x[0]}_{x[1]}_{x[2]}.png"

    return file_name
#data_df["file"] = data_df.apply(create_file_name, axis=1)
#data_df.head()
def read_image_sizes(file_name):

    image = skimage.io.imread(IMAGE_PATH + file_name)

    return list(image.shape)
#m = np.stack(data_df['file'].apply(read_image_sizes))

#df = pd.DataFrame(m,columns=['w','h'])

#data_df = pd.concat([data_df,df],axis=1, sort=False)
#print(f"Images widths #: {data_df.w.nunique()},  heights #: {data_df.h.nunique()}")

#print(f"Images widths values: {data_df.w.unique()},  heights values: {data_df.h.unique()}")
#data_df.head()
def show_images(df, isTest=False):

    f, ax = plt.subplots(10,15, figsize=(15,10))

    for i,idx in enumerate(df.index):

        dd = df.iloc[idx]

        image_name = dd['file']

        image_path = os.path.join(IMAGE_PATH, image_name)

        img_data = cv.imread(image_path)

        ax[i//15, i%15].imshow(img_data)

        ax[i//15, i%15].axis('off')

    plt.show()
#df = data_df.loc[data_df.suite_id==1].sort_values(by=["sample_id","value"]).reset_index()

#show_images(df)
import numpy as np

import pandas as pd 

import cv2

from fastai.vision import *

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

import os

import shutil

from glob import glob

%matplotlib inline

!pip freeze > '../working/dockerimage_snapshot.txt'
#Codes from Paul Mooney https://www.kaggle.com/paultimothymooney/collections-of-paintings-from-50-artists/data



def makeWordCloud(df,column,numWords):

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=numWords,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def plotImages(artist,directory):

    print(artist)

    multipleImages = glob(directory)

    plt.rcParams['figure.figsize'] = (15, 15)

    plt.subplots_adjust(wspace=0, hspace=0)

    i_ = 0

    for l in multipleImages[:25]:

        im = cv2.imread(l)

        im = cv2.resize(im, (128, 128)) 

        plt.subplot(5, 5, i_+1) #.set_title(l)

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

        i_ += 1



np.random.seed(7)
print(os.listdir("../input/art-pictogram/pictogram/"))
img_dir='../input/art-pictogram/pictogram'

path=Path(img_dir)

data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3),

                                  size=299,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')

data.show_batch(rows=8, figsize=(40,40))