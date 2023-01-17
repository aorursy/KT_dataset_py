# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read the data

gone_wind = open('../input/text-mining-gone-with-the-wind/GoneWithTheWind.txt',errors='ignore').read()
import random

import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS

%matplotlib inline
# mask address 

from PIL import Image

mask_img = np.array(Image.open('../input/gone-with-the-wind-images/4.jpg'))
plt.figure(figsize=(8,8))

plt.imshow(mask_img)

plt.axis('off')
plt.figure(figsize=(8,8))

wc = WordCloud(max_words=200,

               background_color = 'white',

               mask = mask_img,

               stopwords=set(STOPWORDS),

               random_state=42, mode = 'RGB', 

               ).generate(gone_wind)

plt.title("Gone With The Wind - Word Cloud",fontsize=20)

plt.axis('off')

plt.imshow(wc)