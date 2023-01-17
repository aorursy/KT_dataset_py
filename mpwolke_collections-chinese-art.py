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

#overview = pd.read_csv('../input/chinese-fine-art/Dataset/')

#overviewArtist = overview[['name','paintings']]

#overviewArtist = overviewArtist.sort_values(by=['paintings'],ascending=False)

#overviewArtist = overviewArtist.reset_index()

#overviewArtist = overviewArtist[['name','paintings']]    
print(os.listdir("../input/chinese-fine-art/Dataset/"))
plotImages("Ling_Jian_凌健","../input/chinese-fine-art/Dataset/Ling_Jian_凌健/**")
plotImages("Zhao_Kailin_赵开霖","../input/chinese-fine-art/Dataset/Zhao_Kailin_赵开霖/**")
plotImages("Zhang_Xiao_Gang_张晓刚","../input/chinese-fine-art/Dataset/Zhang_Xiao_Gang_张晓刚/**")
plotImages("Luo_Zhong_Li_羅中立","../input/chinese-fine-art/Dataset/Luo_Zhong_Li_羅中立/**")
plotImages("Li_Zijian_李自健","../input/chinese-fine-art/Dataset/Li_Zijian_李自健/**")
plotImages("Ai_Xuan_艾軒","../input/chinese-fine-art/Dataset/Ai_Xuan_艾軒/**")
plotImages("Zhao_Kailin_赵开霖","../input/chinese-fine-art/Dataset/Zhao_Kailin_赵开霖/**")
plotImages("Gu_Wenda_谷文达","../input/chinese-fine-art/Dataset/Gu_Wenda_谷文达/**")
plotImages("He_Baili_何百里","../input/chinese-fine-art/Dataset/He_Baili_何百里/**")
img_dir='../input/chinese-fine-art/Dataset'

path=Path(img_dir)

data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3),

                                  size=299,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')

data.show_batch(rows=8, figsize=(40,40))
#learn = create_cnn(data, models.resnet50, metrics=accuracy,model_dir=Path("/kaggle/working/"),path=Path("."))

#learn.fit_one_cycle(3)
plotImages("Wai_Ming","../input/chinese-fine-art/Dataset/Wai_Ming/**")
plotImages("Liu_Yuan_Shou_劉元壽","../input/chinese-fine-art/Dataset/Liu_Yuan_Shou_劉元壽/**")