import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import string

from matplotlib import pyplot

import numpy as np

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading the data



data = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')

data.head()
#Removing the columns that are not needed



data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data = data.rename(columns={"v1":"label", "v2":"body_text"})
data.head()
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))



data.head()
def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    word_len= len(text) - text.count(" ")

    percent=(count/word_len)*100

    return round(percent, 3)



data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))



data.head()
bins = np.linspace(0, 200, 40)



pyplot.hist(data[data['label']=='spam']['body_len'], bins, alpha=0.5, label='spam')

pyplot.hist(data[data['label']=='ham']['body_len'], bins, alpha=0.5,  label='ham')

pyplot.legend(loc='upper left')

pyplot.show()

bins = np.linspace(0, 50, 40)



pyplot.hist(data[data['label']=='spam']['punct%'], bins, alpha=0.5,  label='spam')

pyplot.hist(data[data['label']=='ham']['punct%'], bins, alpha=0.5,  label='ham')

pyplot.legend(loc='upper right')

pyplot.show()
bins = np.linspace(0, 50, 40)



pyplot.hist(data['body_len'], bins)

pyplot.title("Body Length Distribution")

pyplot.show()
bins = np.linspace(0, 50, 40)



pyplot.hist(data['punct%'], bins)

pyplot.title("Punctuation Distribution")

pyplot.show()