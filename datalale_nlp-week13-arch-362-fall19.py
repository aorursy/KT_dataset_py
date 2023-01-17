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
import matplotlib.pyplot as plt

import seaborn as sns
yelp_df = pd.read_csv("../input/yelp-data-set/yelp.csv")

yelp_df.head(7)
yelp_df.tail()
yelp_df.describe()
yelp_df['text'][0]
yelp_df['text'][1]
yelp_df['text'][9998]
yelp_df['text'][9999]
yelp_df['length']= yelp_df['text'].apply(len)
##print (yelp_df['length'])
yelp_df
yelp_df['length'].plot(bins = 100, kind = 'hist')
yelp_df.length.describe()
yelp_df[yelp_df['length']==4997]['text'].iloc[0] ##finding the longest text
sns.countplot(y='stars', data = yelp_df)
g= sns.FacetGrid(data=yelp_df, col= 'stars', col_wrap=3)

g.map(plt.hist, 'length', bins= 20, color= 'purple')
yelp_df_1 = yelp_df[yelp_df['stars']== 1] # Listing the reviews with 1 stars

yelp_df_1
yelp_df_5 = yelp_df[yelp_df['stars']== 5] # Listing the reviews with 5 stars

yelp_df_5
yelp_df_1_5=pd.concat([yelp_df_1, yelp_df_5])

yelp_df_1_5
yelp_df_1_5.info()
print('1-Stars Review Percentage - ', (len(yelp_df_1)/len(yelp_df_1_5))*100, '%') # prints the percentage of 1star reviews among the total of 1 and 5star reviews.
sns.countplot(yelp_df_1_5['stars'], label='Count') #prints a bar chart of count versus stars
import string #imports a list of punctuation marks.

string.punctuation # prints the list of punctuation marks
Test= "Hello, Are you enjoying the AI course?!!" # a text to make sure the punctuation marks are removed
Test
Test_punc_removed= [char for char in Test if char not in string.punctuation] #removes the punctuation marks from the test text: print the character if it is not listed among punctuation marks

Test_punc_removed #prints each single item to remove p. marks one by one
Test_punc_removed_join = ''.join(Test_punc_removed)# Joins the text back into its shape without the punctuation marks
Test_punc_removed_join
from nltk.corpus import stopwords

stopwords.words('english')
Test_punc_removed_join_clean = [ word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
Test_punc_removed_join_clean
#Assignment_8 = 'Here is the Assignment#8, for you to think on how to remove stopwords and punctuations!'
#Solution = [ char for char in Assignment_8 if char not in string.punctuation]

#Solution = ''.join(Solution)

#Solution = [ word for word in Solution.split() if word.lower() not in stopwords.words('english')]
#Solution
def message_cleaning(message):

    Test_punc_removed = [char for char in message if char not in string.punctuation]

    Test_punc_removed_join = ''.join(Test_punc_removed)

    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower()]

    return Test_punc_removed_join_clean
yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)
print(yelp_df_clean[0]) # cleaned review
print(yelp_df_1_5['text'][0]) # Original review
yelp_df_1_5['length'].describe()
yelp_df_1_5[yelp_df_1_5['length'] == 662] ['text'].iloc[0]
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = message_cleaning)

yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])
print(vectorizer.get_feature_names())
yelp_countvectorizer.shape
from sklearn.naive_bayes import MultinomialNB



NB_classifier = MultinomialNB()

label = yelp_df_1_5['stars'].values
yelp_df_1_5['stars'].values
NB_classifier.fit(yelp_countvectorizer, label)
testing_sample = ['amazing food! highly recommended']

#testing_sample = ['shit food! made me sick']

testing_sample_countvectorizer = vectorizer.transform(testing_sample)

test_predict = NB_classifier.predict(testing_sample_countvectorizer)



test_predict