# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer



german_stop_words = stopwords.words('german')

english_stop_words = stopwords.words('english')



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # to create a Word Cloud

from PIL import Image # Pillow with WordCloud to image manipulation

import re
dfDE = pd.read_csv('/kaggle/input/youtube-new/DEvideos.csv', sep=",")
dfDE.head(1)
dfDE.isnull().any() #check, weather there are Nones
dfDE = dfDE.dropna() #del nones
stopwords = set(german_stop_words) #imported from wordclous

stopwords.update(set(english_stop_words))

stopwords.update(["|", "-", "https","google","facebook","youtube","http",

                  "org","twitter", "goo", "gl", "instagram",

                  "UTF 8","org","de","bit","ly",]) #add my own

de = " ".join(dfDE["title"])

de = de.lower()

de = [word for word in de.split() if word not in stopwords and word.isalpha()]

de = " ".join(de)
de_tags = " ".join(dfDE["tags"])

de_tags = de_tags.lower()

de_tags = [word for word in de_tags.split() if word not in stopwords and word.isalpha()]

de_tags = " ".join(de_tags)
de_des = " ".join(str(word) for word in dfDE["description"])

de_des = de_des.lower()

de_des = re.sub(r"\\n\\n","", de_des) #use regular expression to replace \n with nothing

de_des = re.sub(r"\\n", "", de_des)

de_des = [word for word in de_des.split() if word not in stopwords and word.isalpha()]

de_des = " ".join(de_des)
#Display titles

mask = np.array(Image.open('/kaggle/input/flag-pics/heart.png'))

wordcloud_de = WordCloud (stopwords=stopwords, background_color="white", 

                          mode="RGBA", max_words=1000, mask=mask, 

                          font_path='/kaggle/input/font-new-arabic/AdobeArabic-Bold.otf').generate(de)



# create coloring from image

image_colors = ImageColorGenerator(mask)

plt.figure(figsize=[10,10])

plt.imshow(wordcloud_de.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")



# store to file

#plt.savefig(".../de_heart.png", format="png")



plt.show()
#Display tags

mask = np.array(Image.open('/kaggle/input/flag-pics/flag.png'))

wordcloud_de = WordCloud (stopwords=stopwords, background_color="white"

                          , mode="RGBA", max_words=1000, mask=mask,

                         font_path='/kaggle/input/font-new-arabic/AdobeArabic-Bold.otf').generate(de_tags)



# create coloring from image

image_colors = ImageColorGenerator(mask)

plt.figure(figsize=[10,10])

plt.imshow(wordcloud_de.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")



# store to file

#plt.savefig(".../de_flag.png", format="png")



plt.show()
#Most common words in describtion

mask = np.array(Image.open('/kaggle/input/flag-pics/flag.png'))

wordcloud_de = WordCloud (stopwords=stopwords, background_color="white", 

                          mode="RGBA", max_words=1000, mask=mask,

                         font_path='/kaggle/input/font-new-arabic/AdobeArabic-Bold.otf').generate(de_des)



# create coloring from image

image_colors = ImageColorGenerator(mask)

plt.figure(figsize=[10,10])

plt.imshow(wordcloud_de.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")



# store to file

#plt.savefig(".../de_flag.png", format="png")



plt.show()