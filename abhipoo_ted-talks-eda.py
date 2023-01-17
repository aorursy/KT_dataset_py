import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv('../input/ted_main.csv')
df.head()

df['views'].plot.line()
df[df['duration']< 3000]['duration'].plot.hist()
df['event'].value_counts()

#themes with max occurences

import ast
df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))

s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'theme'

theme_df = df.drop('tags', axis=1).join(s)
theme_df.head(3)

#total number of themes
len(theme_df['theme'].value_counts())

#themes with most occurences
theme_df['theme'].value_counts().head(50)

#themes with most views
theme_df.groupby('theme')['views'].sum().sort_values(ascending=False)


#term frequency analysis

transcript_df = pd.read_csv('../input/transcripts.csv')

transcript_df.head()

stop_words = set(stopwords.words('english'))

text1 = transcript_df["transcript"][0]

word_tokens = word_tokenize(text1)

#filtered_sentence = [w for w in word_tokens if not w in stop_words] #SHORTCUT
 
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words and w.isalnum():
        filtered_sentence.append(w)    

len(filtered_sentence)

#print(word_tokens)
#print(filtered_sentence)

Counter(filtered_sentence).most_common()


#finding related talks using themes
vec = DictVectorizer()

vec.fit_transform(df['tags']).toarray()


#testing the matrix

a = [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
b = [0,0,1,0,1,0,1,1,1,0,0,0,0,0,0]
c = [0,0,1,0,1,0,1,1,1,0,0,0,0,0,0]
d = [0,0,1,0,0,0,0,0,0,0,0,1,1,1,1]

print(pearsonr(a,b))
#print(pearsonr(a,c))
#print(pearsonr(a,d))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.