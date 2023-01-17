# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as se

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gebdf = pd.read_csv('/kaggle/input/gibberish-text-classification/Gibberish.csv', encoding= 'unicode_escape')



gebdf.head()
gebdf.info()
# Let's explore the number of words each row has

word_count = list()

for row in gebdf.iterrows():

    word_count.append(len(row[1][0].split()))

#     print(row[1][0])

#     break
gebdf['counts'] = word_count



gebdf.head()
#checking the word counts per each row, we will only consider gebrish words of one

se.distplot(word_count)
gebdf = gebdf.loc[gebdf['counts'] ==1] #select only rows with single word
np.random.randint(0, high=len(gebdf), size=500) # sample of generating gebrish statement randomly
# drop old index and set new

gebdf = gebdf.reset_index(drop=True)
gebphrase = list()

gebcolumn = list()

for phrase in range(0,1000): #generating 1000 phrases

    for i in np.random.randint(0, high=len(gebdf), size=500): #size is the number of words per phrase

        gebphrase.append(gebdf['Response'][i])

    gebcolumn.append(' '.join(gebphrase))

    del gebphrase[:] # this is important to avoid commulatively overloading the list 
# let's count the words for each row

word_count = list()

for phrase in gebcolumn:

    word_count.append(len(phrase.split()))

    

# plot the distribution of the number of words that we have generated

se.distplot(word_count)
# have a look at one of the phrases

gebcolumn[0]