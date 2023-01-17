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
!pip install textblob

from textblob import TextBlob     
text = '''                                       
Sentiment analysis is the interpretation and classification of emotions (positive, negative and neutral) within text 
data using text analysis techniques. Sentiment analysis allows 
businesses to identify customer sentiment toward products, brands or services in online conversations and feedback.
'''
blob = TextBlob(text)
blob
blob.sentences
from textblob import TextBlob  
'''Sentiment analysis is the interpretation and classification of emotions (positive, negative and neutral) within text 
data using text analysis techniques. Sentiment analysis allows 
businesses to identify customer sentiment toward products, brands or services in online conversations and feedback.
'''
blob = TextBlob(text)
blob.words
from textblob import TextBlob   
text = '''                                       
Sentiment analysis is the interpretation and classification of emotions (positive, negative and neutral) within text 
data using text analysis techniques. Sentiment analysis allows.
'''
blob = TextBlob(text) 
blob.tags  
from textblob import TextBlob
text = '''                                       
Sentiment analysis is the interpretation and classification of emotions (positive, negative and neutral) within text 
data using text analysis techniques. Sentiment analysis allows 
businesses to identify customer sentiment toward products, brands or services in online conversations and feedback.
'''
blob = TextBlob(text) 
blob.noun_phrases   
sentence = TextBlob("This is really good !")       
sentence.sentiment
sentence = TextBlob("This is really good !") 
#transalate sentence to spanish using language translator 
sentence.translate(to="es")
monty = TextBlob("This is beautiful and you are looking beautiful too")
monty.word_counts['beautiful']
#spelling checker which give probabilities 
from textblob import Word
w = Word('aple')
w.spellcheck()
#Pluralization of a word
from textblob import Word
w = Word("personality")
print (w.pluralize())
blob = TextBlob("Now is better than never. We are even. Good night")
blob.ngrams(n=2)
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
opinion = TextBlob("This is a beautiful place", analyzer=NaiveBayesAnalyzer())
opinion.sentiment
