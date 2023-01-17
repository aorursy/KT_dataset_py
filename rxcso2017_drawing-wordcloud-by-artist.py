# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import random

#import helpers

import cv2

from PIL import Image

import numpy as np # linear algebra

import pandas as pd # data processing,CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from scipy.misc import imread

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
T=pd.read_csv("../input/lyrics.csv")
T.genre.value_counts()
T[ T["genre"]=='Rock' ].artist.value_counts()
T_byartist = T["artist"].value_counts()

T_byartist.head(50)
def textArtist(s):

    lyrics=""

    for ind,val in T.iterrows():

        if val["artist"]==s:

            lyrics = lyrics + str(val["lyrics"])

    return lyrics
lyrics = textArtist('elvis-presley')
def ofString(s):

    s = s.lower()

    s = s.replace('\n',' ')

    s = s.replace(',',' ')

    return s
def allTextByArtist(T):

    D={}

    for ind,val in T.iterrows():

        art = val["artist"]

        if art in D:

            D[art]= D[art]+ ofString(str(val["lyrics"]))

        else:

            D[art]= ofString(str(val["lyrics"]))

    return D
D = allTextByArtist(T)
english_stopwords =set(stopwords.words('english')) | STOPWORDS | ENGLISH_STOP_WORDS | set(['ya','aah','ye','hey','ba','da','buh','duh','doo','oh','ooh','woo','uh','hoo','ah','yeah','oo','la','chorus','beep','ha'])
def imageArtist(s):# Create a black image

    cnt = len(s)

    img = np.zeros((612,cnt*420,3), np.uint8)

    

    # Write some Text

    font = cv2.FONT_HERSHEY_TRIPLEX

    cv2.putText(img,s,(0,500), font, 18,(255,255,255),85)

    img = 255-img

    #Save image

    cv2.imwrite("out.jpg", img)

    return img

plt.imshow(imageArtist('Adele'))

def drawCloud(s):

    img = Image.open("out.jpg")

    #img = img.resize((1900,2000), Image.ANTIALIAS)

    hcmask = np.array(img)

    wordcloud = WordCloud(background_color="white",max_words=150,stopwords=english_stopwords,

        mask=hcmask,

        #background_color='#000000',

        #font_path='#fafafa'

    ).generate(s)

    fig = plt.figure()

    fig.set_figwidth(17)

    fig.set_figheight(10)



    

    #plt.title('GG', color='#fafafa', size=30, y=1.01)

    #plt.annotate('GG', xy=(0, -.025), xycoords='axes fraction', fontsize=12, color='#fafafa')

    

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.figure()

    #plt.imshow(hcmask, cmap=plt.cm.gray)

    #plt.axis("off")

    #plt.show()

    
drawCloud(D['adele'])
def drawCloudArtist(s):

    imageArtist(s)

    s  = s.lower()

    s = s.replace(' ','-')

    drawCloud(D[s])
drawCloudArtist('Foo Fighters')
drawCloudArtist('Eminem')
drawCloudArtist('Coldplay')