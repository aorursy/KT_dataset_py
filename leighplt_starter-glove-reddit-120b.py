import numpy as np # linear algebra

import os, sys



from gensim.models import Word2Vec, KeyedVectors

from gensim.test.utils import datapath



from torchtext import vocab, data

import torchtext
from tqdm import tqdm_notebook

torchtext.vocab.tqdm = tqdm_notebook # Replace tqdm to tqdm_notebook in module torchtext
# define root path

PATH = '../input'



# Get path to file

vertor_size = 512

glove_file = os.path.join(PATH, f'glovereddit120b/GloVe.Reddit.120B.{vertor_size}D.txt')



# Load with gensim

model = KeyedVectors.load_word2vec_format(glove_file)
# Typical test

similarities = model.most_similar(positive=['woman', 'king'], negative=['man'])

print(similarities[0])



similarities = model.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])

print(similarities[0])
# Election influence on vectors 

similarities = model.most_similar('Trump')

for i in similarities:

    print(i)
# Check some trends:

similarities = model.most_similar('SJWs')

for i in similarities:

    print(i)
def print_similarities(word, n=5):

    print(f'Similarities for word: {word}')

    similarities = model.most_similar(word)

    for i in similarities[:n]:

        print(f'\t{i}')

    

print_similarities('fuck')

print_similarities('sh*t')

analogy_scores = model.evaluate_word_analogies(datapath('questions-words.txt'))

print(f'Analogy score: {analogy_scores[0]:.4f}')
categories = analogy_scores[1:][0][:-1]

for category in categories:

    print(f"In {category['section']}, correct: {len(category['correct'])}, incorrect: {len(category['incorrect'])}")
# Load with torchtext:

vec = vocab.Vectors(glove_file, cache='./')



# clean tmp file

!rm *.pt
print(f'Words in embedding: {vec.vectors.size(0)}, dim_size: {vec.vectors.size(1)}')
## Add path to file

utils_path = '../input'

sys.path.insert(0, utils_path)
from clean_text import RegExCleaner
cleaner = RegExCleaner.reddits()
# separate 's from words

text = "This's world"

print(f'Before: {text}, After: {cleaner(text)}')



# shrink repeated letters to 2

text = "This's wwwwwwwwwworld"

print(f'Before: {text}, After: {cleaner(text)}')