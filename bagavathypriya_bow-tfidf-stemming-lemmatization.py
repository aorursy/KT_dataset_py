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
import nltk
nltk.download()
paragraph = "I have three visions for India. In 3000 years of our history people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture and their history and tried to enforce our way of life on them. Why? Because we respect the freedom of others. That is why my FIRST VISION is that of FREEDOM. I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us."
sentence = nltk.sent_tokenize(paragraph)
type(sentence)
sentence
words = nltk.word_tokenize(paragraph)
type(words)
words
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stopwords.words('english')
stemmer = PorterStemmer()
for i in range(len(sentence)):
    words=nltk.word_tokenize(sentence[i])
    words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentence[i]=' '.join(words)
sentence
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

for i in range(len(sentence)):
    words=nltk.word_tokenize(sentence[i])
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentence[i]=' '.join(words)
sentence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import re
corpus = []
for i in range(len(sentence)):
    ab = re.sub('[^a-zA-Z]',' ',sentence[i])
    ab=ab.lower()
    ab=ab.split()
    ab=[lemmatizer.lemmatize(word) for word in ab if word not in set(stopwords.words('english'))]
    ab=' '.join(ab)
    corpus.append(ab)
type(corpus)
corpus
len(corpus)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(corpus).toarray()
type(x)
print(x)
x.shape
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
y=tfidf.fit_transform(corpus).toarray()
y
