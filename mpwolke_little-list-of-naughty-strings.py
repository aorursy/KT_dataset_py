#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQrfB857hMUGw9RBMs1mJZnOfCzqug6K09YoBSknC-mPNioss28',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_json('../input/big-list-of-naughty-strings/package.json', encoding='ISO-8859-2')  
df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSZZOzHwvmuZvQe6Topk4XxCmvK9jZIALNd4mY1XDHAA3YEMuVe',width=400,height=400)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.description)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="red").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
naughty_file = '../input/big-list-of-naughty-strings/blns.txt'

with open(naughty_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

def plotWordFrequency(input):

    f = open(naughty_file,'r')

    words = [x for y in [l.split() for l in f.readlines()] for x in y]

    data = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)[:40] 

    most_words = [x[0] for x in data]

    times_used = [int(x[1]) for x in data]

    plt.figure(figsize=(20,10))

    plt.bar(x=sorted(most_words), height=times_used, color = 'grey', edgecolor = 'black',  width=.5)

    plt.xticks(rotation=45, fontsize=18)

    plt.yticks(rotation=0, fontsize=18)

    plt.xlabel('Most Common Words:', fontsize=18)

    plt.ylabel('Number of Occurences:', fontsize=18)

    plt.title('Most Commonly Used Words: %s' % (naughty_file), fontsize=24)

    plt.show()
naughty_file = '../input/big-list-of-naughty-strings/blns.txt'

plotWordFrequency(naughty_file)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQWO581xjqRJzMnYDzQGMKRl36e_0Hy3bbzlNgr81NVm3Icu521',width=400,height=400)

   
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQPxfGnHZgtxZcdae7NKRdHzcKoI1RyJrMucJr7uI7DWG0BQdcX',width=400,height=400)