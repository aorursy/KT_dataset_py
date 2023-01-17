# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data=pd.read_csv("../input/train.csv")



# Any results you write to the current directory are saved as output.
data.head()
data.iloc[22] ['question_text'] #iloc is information in that location
import nltk
nltk.download('stopwords')

nltk.download('punkt')

nltk.download('vader_lexicon')
!pip install wordcloud
data['target'].value_counts()
#BAG OF WORD ANALYSIS
from wordcloud import WordCloud

import matplotlib.pyplot as plt



wc=WordCloud(background_color='white').generate('i love india')

plt.imshow(wc)
##joins



x=['a','b','c','d']

' '.join(x)
input_string=' '.join(data['question_text'])

wc=WordCloud().generate(input_string)

plt.imshow(wc)
docs= data['question_text'].str.lower().str.replace('[^a-z ]','')
docs.head(2)
stopwords=nltk.corpus.stopwords.words('english')

stopwords[:5]
stemmer= nltk.stem.PorterStemmer()

stemmer.stem('playing')
docs_clean=[]



for doc in docs:

    words=doc.split(' ')

    words_clean=[]

    

    for word in words:

        if word not in stopwords:

            words_clean.append(stemmer.stem(word))

    doc_clean=(' '.join(words_clean))

    docs_clean.append(doc_clean)



print(docs_clean)

    
### using list comprehension



docs = imdb['review'].str.lower().str.replace('[^a-z ]','')

def clean_sentence(text):

    words = text.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    return ' '.join(words_clean)

docs_clean = docs.apply(clean_sentence)

docs_clean.head()
######Properties of document term matrix



### every row is a document and each document should be represented as vectors

### size of vector can be identified by the no. of unique terms in the corpus(i.e. all the reviews together) 

#--it is the no. of columns in the document term matrix

## column sum will give frequency of a word across all reviews/documents

## row sum will give no. of unique words in a review/document(document length)

## sparse matrix ==> most of the values are 0 ==> sparcity = (no. of zero's)/(no. of rows*no.of columns)

## high dimension data

## every column is a vector representation of a term
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

vectorizer.fit(docs_clean)

dtm = vectorizer.transform(docs_clean)

dtm

## in o/p 748 - no. of rows, 2475 - no. of columns(no. of unique words), 6797 are the non zero values 

## here the output is stored as a compressed matrix, as we might get memory errors

## (748*2475) - 6797 is the no. of zero's
no_of_zeroes = (748*2475) - 6797

sparcity = no_of_zeroes / (748*2475) * 100

sparcity
### to display the document term matrix

df_dtm = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names())

(df_dtm == 0).sum() ## column wise no. of zeros

(df_dtm == 0).sum().sum() ## total zeros in the dataset
df_dtm.sum().sort_values(ascending=False).head(2)## frequency of each word
df_dtm.sum(axis = 1).sort_values(ascending = False).head(1) ##uniques words per row/document
from sklearn.model_selection import train_test_split

train_x, validate_x = train_test_split(df_dtm, test_size = 0.2, random_state = 100)
train_y = data.iloc[train_x.index]['target']

validate_y = dat.iloc[validate_x.index]['target']
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(random_state=100,n_estimators=300)

rf_model.fit(train_x,train_y)

rf_predict = rf_model.predict(validate_x)

accuracy_score(validate_y, rf_predict)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

NB_model = MultinomialNB(alpha=1)

NB_model.fit(train_x,train_y)

NB_predict = NB_model.predict(validate_x)

accuracy_score(validate_y, NB_predict)