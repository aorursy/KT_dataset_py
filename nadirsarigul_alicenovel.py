
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image # converting images into arrays

%matplotlib inline

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # needed for waffle Charts

mpl.style.use('ggplot') # optional: for ggplot-like style

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# install wordcloud
!conda install -c conda-forge wordcloud==1.4.1 --yes

# import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud is installed and imported!')
# download file and save as alice_novel.txt
!wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/alice_novel.txt

# open the file and read it into a variable alice_novel
alice_novel = open('/kaggle/input/alice-novel/alice_novel.txt', 'r').read()
    
print ('File downloaded and saved!')
stopwords = set(STOPWORDS)
# instantiate a word cloud object
alice_wc = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

# generate the word cloud
alice_wc.generate(alice_novel)
# display the word cloud
plt.imshow(alice_wc, interpolation="bilinear")
plt.axis("off")
plt.show()
fig = plt.figure()
fig.set_figwidth(14) #set width
fig.set_figheight(18) #set height

#display the could 
plt.imshow(alice_wc, interpolation="bilinear")
plt.axis('off')
plt.show()

stopwords.add("said") #add the words said tostopwords

#re-generate the word cloud
alice_wc.generate(alice_novel)

#display the cloud 
fig = plt.figure()
fig.set_figwidth(14) #set width
fig.set_figheight(18) #set height

plt.imshow(alice_wc, interpolation="bilinear")
plt.axis("off")
plt.show()
# save mask to alice_mask
alice_mask = np.array(Image.open('/kaggle/input/alice-mask/alice_mask.png'))
    
print('Image downloaded and saved!')
fig = plt.figure()
fig.set_figwidth(14) #set width
fig.set_figheight(18) #set height

plt.imshow(alice_mask, cmap = plt.cm.gray, interpolation = "bilinear")
plt.axis('off')
plt.show()
# instantiate a word cloud object 
alice_wc = WordCloud(background_color='white', max_words=2000,
                      mask=alice_mask, stopwords=stopwords
                      )
#generate the word cloud 
alice_wc.generate(alice_novel)

#display th word cloud 
fig = plt.figure()
fig.set_figwidth(14) # set width
fig.set_figheight(18) # set height

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()
