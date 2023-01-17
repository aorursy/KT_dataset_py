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
import spacy
nlp = spacy.load('en_core_web_lg')
tokens = nlp(u'cats lion pet')
for token1 in tokens:

    for token2 in tokens:

        print(token1.text, token2.text, token1.similarity(token2))
tokens = nlp(u"dog cat Marcially")

for token in tokens:

    print(token.text,token.has_vector,token.vector_norm,token.is_oov)
from scipy import spatial

cosine_similarity = lambda vec1,vec2: 1 - spatial.distance.cosine(vec1,vec2)
king = nlp.vocab['king'].vector

man = nlp.vocab['man'].vector

woman = nlp.vocab['woman'].vector
new_vector = king - man + woman
computed_similarities = []

for word in nlp.vocab:

    if word.has_vector:

        if word.is_lower:

            if word.is_alpha:

                similarity = cosine_similarity(new_vector,word.vector)

                computed_similarities.append((word,similarity))
computed_similarities = sorted(computed_similarities,key=lambda item:-item[1])
print([t[0].text for t in computed_similarities[:10]])
import nltk
nltk.download('vader')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
a = "This is a good movie"

#b = "This was the BEST MOST AWESOME movie ever!!!!!"
sid.polarity_scores(a)

# score contains positive,negative and compound scores
df = pd.read_csv('../input/movie-reviews/moviereviews.tsv',sep='\t')
#Remove null and whitespace entries

df.dropna(inplace=True)

blanks = []

for i,lb,rv in df.itertuples():

    if type(rv) == str:

        if rv.isspace():

            blanks.append(i)
#remove whitespace

df.drop(blanks,inplace=True)
#Check top 5 data rows

df.head()
df['label'].value_counts()
# Add a new column with scores JSON

df['scores'] = df['review'].apply(lambda review:sid.polarity_scores(review))
#Add a new column with the compound score value extracted from score column

df['compound'] = df['scores'].apply(lambda d:d['compound'])
# my own logic for prediction/ No ml involved

df['comp_score'] = df['compound'].apply(lambda score: 'pos' if score>=0 else 'neg')
df.head()
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(df['label'],df['comp_score'])
print(confusion_matrix(df['label'],df['comp_score']))