# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
paintings_df = pd.read_csv('/kaggle/input/paintings-detail-2/paitings_detail.csv')
paintings_df.head()
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

overview = pd.read_csv('/kaggle/input/artists-detail/artists.csv')

overviewArtist = overview[['artist_id','name','paintings']]

overviewArtist = overviewArtist.sort_values(by=['paintings'],ascending=False)

overviewArtist = overviewArtist.reset_index()

overviewArtist = overviewArtist[['artist_id', 'name','paintings']]
overviewArtist = pd.merge(overviewArtist, paintings_df, on=['artist_id', 'name'], how='right')
print(os.listdir("/kaggle/input/resized-class-2/resized"))
plt.rcParams['figure.figsize'] = (15, 15)

plt.imshow(cv2.imread("/kaggle/input/resized-class-2/resized/Caravaggio_1.jpg"))

shutil.copyfile("/kaggle/input/resized-class-2/resized/Caravaggio_1.jpg", "/kaggle/working/Caravaggio_1.jpg")
overviewArtist.head()
name = overviewArtist['name'].str.replace(' ', '_')

name = name + '_' + overviewArtist['paiting_id'].map(str) + '.jpg'
label  = pd.DataFrame(data={'name':name, 'label': overviewArtist['class']}) 
label.head()
img_dir='/kaggle/input/resized-class-2/resized/'

path=Path(img_dir)



data = (ImageList.from_df(label, path)

        .split_by_rand_pct()

        .label_from_df(cols='label')

        .transform(get_transforms(), size=128)

        .databunch())



print(f'Classes: \n {data.classes}')

data.show_batch(rows=8, figsize=(40,40))
learn = create_cnn(data,

                   models.resnet50,

                   metrics=accuracy,

                   model_dir=Path("/kaggle/working/"),

                   path=Path("."))

learn.fit_one_cycle(3)
plt.rcParams['figure.figsize'] = (5, 5)

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10)
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)
interp.plot_top_losses(9, figsize=(10,10))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)