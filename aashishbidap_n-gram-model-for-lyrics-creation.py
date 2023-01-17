import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import bigrams, trigrams
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter, defaultdict
import nltk
import random
nltk.download('punkt')
with open('/kaggle/input/avicii-songs-lyrics-corpus/avicii_lyrics.txt') as f:
    contents = f.read()
content_updated = contents.replace('\n',' ')
content_updated = content_updated.replace("'",'')
content_updated = content_updated.replace(')','')
content_updated = content_updated.replace('(','')
content_updated = content_updated.replace('[','')
content_updated = content_updated.replace(']','')
avicii_corpus = word_tokenize(content_updated)
model = defaultdict(lambda: defaultdict(lambda: 0))
#creating Trigrams
for w1, w2, w3 in trigrams(avicii_corpus, pad_right=True, pad_left=True):
    model[(w1, w2)][w3] += 1
#counting of 
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
dict(model['I','am'])
text = ["the","nights"]
sentence_finished = False
 
while not sentence_finished:
  r = random.random()
  accumulator = .0
  for word in model[tuple(text[-2:])].keys():
      accumulator += model[tuple(text[-2:])][word]
      # select words that are above the probability threshold
      if accumulator >= r:
          text.append(word)
          break
  if text[-2:] == [None, None]:
      sentence_finished = True

count=1
for t in text:
  if count < 500:
    if count % 9 == 0:
      print(t)
      count = count + 1
    else:
      print(t,end=" ")
      count = count + 1
