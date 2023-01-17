#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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
imagenet_file = '../input/imagenette/imagenette/full-size-v2/0.1.0/label.labels.txt'

with open(imagenet_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

def plotWordFrequency(input):

    f = open(imagenet_file,'r')

    words = [x for y in [l.split() for l in f.readlines()] for x in y]

    data = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)[:40] 

    most_words = [x[0] for x in data]

    times_used = [int(x[1]) for x in data]

    plt.figure(figsize=(20,10))

    plt.bar(x=sorted(most_words), height=times_used, color = 'pink', edgecolor = 'black',  width=.5)

    plt.xticks(rotation=45, fontsize=18)

    plt.yticks(rotation=0, fontsize=18)

    plt.xlabel('Most Common Words:', fontsize=18)

    plt.ylabel('Number of Occurences:', fontsize=18)

    plt.title('Most Commonly Used Words: %s' % (imagenet_file), fontsize=24)

    plt.show()
imagenet_file = '../input/imagenette/imagenette/full-size-v2/0.1.0/label.labels.txt'

plotWordFrequency(imagenet_file)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQnmt5L19v0E5uKTg5RpyzROR2YrsU8EB_lPynS0NiHU-pNor-t&usqp=CAU',width=400,height=400)