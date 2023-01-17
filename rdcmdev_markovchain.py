import pandas as pd
import numpy as np
from numpy.linalg import matrix_power
from nltk.tokenize import word_tokenize
import random
!ls ../input
verses = []
with open('../input/robert_frost.txt') as fl:    
    verse = []
    for line in fl.readlines():                
        if line == "\n": 
            verses.append(verse)
            verse = []
        verse.append(line.replace('\n', ''))        
len(verses)
def tokenize(string, to_lower=True, is_alpha=True):
  if to_lower:
    if is_alpha:
      return [word.lower() for word in word_tokenize(string)] #if word.isalpha()]
    else:
      return [word for word in word_tokenize(string)]
  return [word for word in word_tokenize(string) if word.isalpha()]
tokens = tokenize(' '.join([word for verse in verses for word in verse]))
tokens_unique = np.unique(tokens)
tokens_unique.shape
matrix_order = tokens_unique.shape[0]
matrix_freq = np.zeros((matrix_order, matrix_order))
token_map = {}
for token, i in zip(tokens_unique, range(matrix_order)):
    token_map[token] = i
for verse in verses:
    tokens_verse = tokenize(' '.join([word for word in verse]))
    tokens_verse_size = len(tokens_verse)
    tokens_left = range(0, tokens_verse_size - 1)
    tokens_right = range(1, tokens_verse_size)
    
    for tl, tr in zip(tokens_left, tokens_right):        
        il, ir = token_map[tokens_verse[tl]], token_map[tokens_verse[tr]]
        matrix_freq[il][ir] += 1        
    il, ir = token_map[tokens_verse[-1]], token_map[tokens_verse[0]]
    matrix_freq[il][ir] += 1
sum_rows = np.sum(matrix_freq, axis=1)
sum_rows.shape
matrix_prob = np.zeros((matrix_freq.shape[0], matrix_freq.shape[0]))
for i, line in zip(range(matrix_freq.shape[0]), matrix_freq):
    matrix_prob[i] = line / sum_rows[i]
matrix_prob.shape
work = True
for i in range(matrix_prob.shape[0]):
    s = np.sum(matrix_prob[i, :])
    if abs(s - 1.0) > 1e-10:        
        work = False
        break
print(f'work? {"YES" if work else "NO"}')
def check_converge(array):    
    s = 0
    for i in range(array.shape[0]):
        a = array[:, i] * 100000
        a = np.round(a) / 100000        
        s += len(np.unique(a))        
    return s, int(s) == int(array.shape[0])
%%time
pot = 1
done = False
matrix_eman = matrix_prob
while not done:
    matrix_eman = matrix_power(matrix_eman, 2)
    pot *= 2            
    s, done = check_converge(matrix_eman)    
    print(f'potencia: {pot} sum: {s}')     
check_converge(matrix_eman)
for i in range(10):
    text = random.choices(tokens_unique, matrix_eman[0], k=10)
    print(text)