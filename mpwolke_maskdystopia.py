# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

im = Image.open("../input/medical-masks-dataset/medical masks dataset/images/1303078448-China-Coronavirus-Death-Toll-Hits-304.jpg")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/medical-masks-dataset/medical masks dataset/images/in-swine_flu_school.jpg")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
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
from PIL import Image

im = Image.open("../input/medical-masks-dataset/medical masks dataset/images/virus_protection123.jpg")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/medical-masks-dataset/medical masks dataset/images/200130103155-wuhan-virus-0129-hong-kong-train-super-tease.jpg")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
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
from PIL import Image

im = Image.open("../input/medical-masks-dataset/medical masks dataset/images/merlin_167592177_faab52d5-95c1-48a0-b533-934d06fbed8d-superJumbo.jpg")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
print(os.listdir("../input/medical-masks-dataset/medical masks dataset/images/"))
from PIL import Image

im = Image.open("../input/medical-masks-dataset/medical masks dataset/images/129_mask.jpg")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
img_dir='../input/medical-masks-dataset/medical masks dataset/images'

path=Path(img_dir)

data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3),

                                  size=299,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')

data.show_batch(rows=8, figsize=(40,40))
from PIL import Image

im = Image.open("../input/medical-masks-dataset/medical masks dataset/images/hong-kong-declares-emergency-shuts-down-schools.jpg")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/medical-masks-dataset/medical masks dataset/images/haze_malaysia_2_malay_mail (1).jpg")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())