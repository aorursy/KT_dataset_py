# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.shape
def count_of_words(text):

    exclude = set(string.punctuation)

    text = ''.join(ch for ch in text if ch not in exclude)

    list_of_words=[]

    for word in text.lower().split():

        if word not in list_of_words:

            list_of_words.append(word)

            

    

    count = 0

    text = ''.join(ch for ch in text if ch not in exclude)

    word_count=[]

    for word_to_count in list_of_words:

        individual_count=[]

        for word in text.lower().split():

            if word == word_to_count:

                count=count+1

        individual_count.append(word_to_count)

        individual_count.append(count)

        word_count.append(individual_count)

        count=0



    word_count = pd.DataFrame(word_count,columns =['word', 'count'])

    return word_count
real = count_of_words(train[train['target']==1]['text']).to_csv('real.csv',index=False)
fake = count_of_words(train[train['target']==0]['text']).to_csv('fake.csv',index=False)