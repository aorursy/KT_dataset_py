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
#inspect tje dataframe

dir = '../input/sms-spam-collection-dataset/spam.csv'

import pandas as pd

df = pd.read_csv(dir, encoding='ISO-8859-1')

df.head()
import pandas as pd

df['sms'] = df['v2']

df['spam'] = np.where(df['v1'] == 'spam', 1, 0)

df.head()
df = df[['sms', 'spam']]

df.head()
len(df)
sample_df = df.sample(frac=0.25)

len(sample_df)

spam_df = sample_df.loc[df['spam'] == 1]

ham_df = sample_df.loc[df['spam'] == 0]



print(len(spam_df))

print(len(ham_df))
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer_spam = TfidfVectorizer(stop_words ='english',

max_features=30)



vectorizer_spam.fit(spam_df['sms'])

vectorizer_spam.vocabulary_
word = 'win'
word = 'win'

spam_count  = 0

spam_with_word_count = 0



for idx,row in spam_df.iterrows():

    spam_count += 1

    

    if word in row.sms:

        spam_with_word_count += 1



probability_of_word_given_spam = spam_count / spam_with_word_count

print(probability_of_word_given_spam)

probability_of_spam = len(spam_df) / (len(sample_df))

print(probability_of_spam)

sms_count = 0

word_in_sms_count = 0

for idx,row in sample_df.iterrows():

    sms_count += 1

    

    if word in row.sms:

        word_in_sms_count += 1

probability_of_word = word_in_sms_count / sms_count

print(probability_of_word)
(probability_of_word_given_spam * probability_of_spam) / probability_of_word