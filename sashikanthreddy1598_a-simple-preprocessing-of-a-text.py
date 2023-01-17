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
import pandas as pd

text = '''Management is the process of designing and maintaining an environment in which individuals, working together

in groups, effectively accomplish selected aims. As managers, people carry out the managerial functions of planning,

organizing, staffing, leading and controlling. Managers are entrusted with the responsibility of taking actions that

will make it possible for individuals to make their best contribution to group objectives. Management thus applies to

small and large organizations, to profit and not-for-profit enterprises, to manufacturing as well as service industries. 

So, the term “Management” refers to business, government agencies, hospitals, universities, churches, temples, and in 

its broad sense covers all business and non-business organizations. Effective managing is the concern of 

the Corporation President, the hospital administrator, the government first line executives and the heads of all

institutions and enterprises, big or small.'''



print(text)
# !pip install nltk

import nltk

from nltk import word_tokenize,sent_tokenize

from nltk.stem import PorterStemmer,WordNetLemmatizer
sent_tokenize(text)
import re

text=re.sub(r'\d|\.|\,','',text) # Removing numbers and additional spaces

tokens = word_tokenize(text)

print(tokens)
print(nltk.pos_tag(tokens))
from nltk.corpus import stopwords

stopwords = stopwords.words("english")

print(stopwords)
tokens = [token for token in tokens if token not in stopwords]

print(tokens)
stemmer = PorterStemmer()

stemmer_words=[stemmer.stem(token) for token in tokens if token not in stopwords]

print(stemmer_words)
lmtzr = WordNetLemmatizer()

lmtzr_words = [lmtzr.lemmatize(token) for token in tokens if token not in stopwords]

print(lmtzr_words)
!pip install polyglot

!pip install pyicu

!pip install pycld2

!pip install morfessor

import polyglot

from polyglot.text import Text
## Word tokenizaiton using Poly Glot



glot_text = Text(text)

print(glot_text.words)
## Sentence Tokenization using PolyGlot



glot_text.sentences
## Extracting Parts of Speech tags (POS Tags) 



print(glot_text.pos_tags)
!pip install textblob

from textblob import TextBlob

blob_text = TextBlob(text)



## Word tokenization using text blob

print(blob_text.words)
## Sentence tokenization using TEXT BLOB

blob_text.sentences
## Extracting Parts of Speech tags (POS Tags)



print(blob_text.pos_tags)
from nltk.corpus import stopwords

stopwords_german = stopwords.words("german")

stopwords_english = stopwords.words("english")
print(stopwords_german)
### Text 

text = '''Zu meiner Familie gehören vier Personen. Die Mutter bin ich und dann gehört natürlich mein Mann dazu. Wir haben zwei Kinder, einen Sohn, der sechs Jahre alt ist und eine dreijährige Tochter.



Wir wohnen in einem kleinen Haus mit einem Garten. Dort können die Kinder ein bisschen spielen. Unser Sohn kommt bald in die Schule, unsere Tochter geht noch eine Zeit lang in den Kindergarten. Meine Kinder sind am Nachmittag zu Hause. So arbeite ich nur halbtags.



Eigentlich gehören zu unserer Familie auch noch die Großeltern. Sie wohnen nicht bei uns. Sie haben ein Haus in der Nähe. Die Kinder gehen sie oft besuchen.'''
from nltk import word_tokenize,sent_tokenize
tokens = word_tokenize(text)
count_english = 0

count_german = 0

for token in tokens:

    if token in stopwords_english:

        count_english=count_english+1

    if token in stopwords_german:

        count_german = count_german+1

print("There are ",count_english,"english and ",count_german,"german words in the given document.")

if count_english < count_german:

    print("So, This is german language document")

elif count_english > count_german:

    print("So, This is english language document")
print(count_english)

print(count_german)
import os

os.listdir('/Users/ise/nltk_data/corpora/stopwords/')
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

sample_text = ['problem of mathematics','mathematics is base for all','all will have different problems']
cv=CountVectorizer(stop_words='english',lowercase=True)



tdm_train = cv.fit_transform(sample_text)

Mat = tdm_train.todense()



count_vec_data = pd.DataFrame(Mat,columns = sorted(cv.get_feature_names()),index=['doc1','doc2','doc3'])

count_vec_data
tfidf_transformer = TfidfVectorizer(stop_words='english',lowercase=True,norm=None)

tfidf = tfidf_transformer.fit_transform(sample_text)
tfidf = pd.DataFrame(tfidf.todense(),columns = tfidf_transformer.get_feature_names(),index=['doc1','doc2','doc3'])

tfidf