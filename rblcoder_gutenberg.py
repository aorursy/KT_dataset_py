!apt install libdb-dev -y
!pip install bsddb3
!pip install gutenberg
from gutenberg.acquire import load_etext

from gutenberg.cleanup import strip_headers
text = strip_headers(load_etext(28)).strip()

print(text) # AESOP'S FABLES (82 Fables)
!pip install bert-extractive-summarizer

from summarizer import Summarizer

model = Summarizer()

model(text)
import gensim.summarization.summarizer 
print(gensim.summarization.summarizer.summarize(text))
!pip install newspaper3k

from newspaper import fulltext

import requests

article_url="https://www.gutenberg.org/files/41667/41667-h/41667-h.htm"

#Title: The Emerald City of Oz

#Author: L. Frank Baum

article = fulltext(requests.get(article_url).text)

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_summarization.html

print(gensim.summarization.summarizer.summarize(article, word_count=500))

number_of_unique_words = len(set(article.lower().split()))
lexical_diversity = number_of_unique_words/len(article)
number_of_unique_words
len(article)
lexical_diversity