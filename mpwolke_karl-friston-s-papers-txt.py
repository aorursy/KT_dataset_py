#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTT1f-shl8_q0P_x6BH9OW-Z0CxkejbcgvuLppGY21gz2_OaoN_',width=400,height=400)
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSOFq7dKHp9DKF7U6WRdPBV0BJJAYnQedUTxc1c2WMSrOFPUXCX',width=400,height=400)
friston_file = '../input/open-access-karl-fristons-papers-txt/19641614.txt'

with open(friston_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQP7pzQc2kLiPm19ZKXPIL0uZnHXbPETGrjAmyW880iQiD8GLRX',width=400,height=400)
import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

def plotWordFrequency(input):

    f = open(friston_file,'r')

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

    plt.title('Most Commonly Used Words: %s' % (friston_file), fontsize=24)

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTigwtsROOQa98cF0ExcVgUSTLtrXadv-vB4bOIhv65MMUBYviO',width=400,height=400)
friston_file = '../input/open-access-karl-fristons-papers-txt/19641614.txt'

plotWordFrequency(friston_file)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSdJvHGAEJN0eYtMrjTRS9D57fPPKy5zaKYMwFybHOZNrdFq7dl',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSRQwG_JaoEcWxtRS8dPTn1tK5wClq6eIl6C5DZtcGZBvFPNdXz',width=400,height=400)