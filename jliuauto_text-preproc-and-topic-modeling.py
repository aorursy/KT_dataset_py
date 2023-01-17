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
aliases = pd.read_csv("../input/Aliases.csv")
aliases.head()
email_receivers = pd.read_csv("../input/EmailReceivers.csv")
email_receivers.head()
persons = pd.read_csv("../input/Persons.csv")
persons.head()
emails = pd.read_csv("../input/Emails.csv")
print(emails.shape)
print(emails[pd.isnull(emails['ExtractedBodyText']) == True].shape)

email_doc = emails[pd.isnull(emails['ExtractedBodyText']) == False]['ExtractedBodyText']
email_doc.head()

email_doc[1]

import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(email_doc[1])

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
doc.ents[1]
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import string

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
#stemmer = SnowballStemmer('english')
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #stemmed = " ".join(stemmer.stem(word) for word in punc_free.split())    
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in email_doc] 
doc_clean[3]
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=5, num_words=5))
