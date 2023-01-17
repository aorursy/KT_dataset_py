# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import random
import cv2
from PIL import Image
from scipy.misc import imread
from wordcloud import WordCloud
import matplotlib.pyplot as plt
def imageArtist(s):# Create a black image
    cnt = len(s)
    img = np.zeros((612,cnt*420,3), np.uint8)
    
    # Write some Text
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(img,s,(0,500), font, 18,(255,255,255),85)
    img = 255-img
    #Save image
    cv2.imwrite("text.jpg", img)
    return img
plt.imshow(imageArtist('  RANJIT'))
def drawCloud(s):
    img = Image.open("text.jpg")
    hcmask = np.array(img)
    wordcloud = WordCloud(background_color="white",max_words=200,
                            mask=hcmask, random_state=75,  max_font_size=120).generate(s)
    fig = plt.figure(figsize=(30.0,25.0))
    #fig.set_figwidth(120)
    #fig.set_figheight(120)

    plt.imshow(wordcloud)
    plt.axis("off")
    plt.figure()        
text = "Database Programming Design Modeling Math Statistics Communication Visualization Artificial \
        Intelligence AI ML Machine Learning Data Science Big Data predictive Analytics NLP Python R \
        C C++ JAVA Business MBA Embedded Android Mobile Data-Scientist Management TechLead Telecom Tablaeu Spark Sqoop Hadoop ETL SQL\
        Agile TeamWork MSc Engineer"
drawCloud(text)
