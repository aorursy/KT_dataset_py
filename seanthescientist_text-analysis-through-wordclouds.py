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
# import package and its set of stopwords

from wordcloud import WordCloud, STOPWORDS



print ('Wordcloud is installed and imported!')
herconman = open('/kaggle/input/hermanconman/confidence-man.txt').read()
stopwords = set(STOPWORDS)
conman_wc = WordCloud(

    background_color='white',

    max_words=5000,

    stopwords=stopwords

)



# generate the word cloud

conman_wc.generate(herconman)
%matplotlib inline



import matplotlib as mpl

import matplotlib.pyplot as plt
plt.imshow(conman_wc, interpolation='bilinear')

plt.axis('off')

plt.show()




conman_wc.generate(herconman)





fig = plt.figure()

fig.set_figwidth(14) 

fig.set_figheight(18) 



plt.imshow(conman_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
Giul_inv=open('/kaggle/input/cnngrab/Giuliani_investigation.txt').read()
Giul_wc = WordCloud(

    background_color='white',

    max_words=5000,

    stopwords=stopwords

)



Giul_wc.generate(Giul_inv)


fig = plt.figure()

fig.set_figwidth(14) 

fig.set_figheight(18) 



plt.imshow(Giul_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
TKurd=open('/kaggle/input/cnngrab/Trump_Kurd.txt').read()
TKurd_wc = WordCloud(

    background_color='black',

    max_words=5000,

    stopwords=stopwords

)



TKurd_wc.generate(TKurd)


fig = plt.figure()

fig.set_figwidth(14) 

fig.set_figheight(18) 



plt.imshow(TKurd_wc, interpolation='bilinear')

plt.axis('off')

plt.show()