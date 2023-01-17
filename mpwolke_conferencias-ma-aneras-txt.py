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
mañaneras_file = '../input/mananeras/articulos/2020/agosto/02--version-estenografica-conferencia-de-prensa-informe-diario-sobre-coronavirus-covid-19-en-mexico-249359.txt'

with open(mañaneras_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(2000))
#Codes from Paul Mooney https://www.kaggle.com/paultimothymooney/poetry-generator-rnn-markov



import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

def plotWordFrequency(input):

    f = open(mañaneras_file,'r')

    words = [x for y in [l.split() for l in f.readlines()] for x in y]

    data = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)[:40] 

    most_words = [x[0] for x in data]

    times_used = [int(x[1]) for x in data]

    plt.figure(figsize=(20,10))

    plt.bar(x=sorted(most_words), height=times_used, color = 'purple', edgecolor = 'black',  width=.5)

    plt.xticks(rotation=45, fontsize=18)

    plt.yticks(rotation=0, fontsize=18)

    plt.xlabel('Most Common Words:', fontsize=18)

    plt.ylabel('Number of Occurences:', fontsize=18)

    plt.title('Most Commonly Used Words: %s' % (mañaneras_file), fontsize=24)

    plt.show()
mañaneras_file = '../input/mananeras/articulos/2020/agosto/02--version-estenografica-conferencia-de-prensa-informe-diario-sobre-coronavirus-covid-19-en-mexico-249359.txt'

plotWordFrequency(mañaneras_file)