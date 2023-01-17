import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

import numpy as np

import pandas as pd

import torch

torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)
with open('/kaggle/input/13437.txt', encoding='utf8') as f:

    text = f.read()
lines = text.split('\n')

line = lines[504]

line
letter_t = torch.zeros(len(line), 128)  # 128 hardcoded due to the limits of ASCII

letter_t.shape
for i, letter in enumerate(line.lower().strip()):

    letter_index = ord(letter) if ord(letter) < 128 else 0  # The text uses directional double

                                                            # quotes, which are not valid ASCII,

                                                            # so we screen them out here.

    letter_t[i][letter_index] = 1
def clean_words(input_str):

    punctuation = '.,;:"!?”“_-'

    word_list = input_str.lower().replace('\n',' ').split()

    word_list = [word.strip(punctuation) for word in word_list]

    return word_list



words_in_line = clean_words(line)

line, words_in_line
word_list = sorted(set(clean_words(text)))

word2index_dict = {word: i for (i, word) in enumerate(word_list)}



len(word2index_dict), word2index_dict['grandmother']
word_t = torch.zeros(len(words_in_line), len(word2index_dict))

for i, word in enumerate(words_in_line):

    word_index = word2index_dict[word]

    word_t[i][word_index] = 1

    print('{:2} {:4} {}'.format(i, word_index, word))



print(word_t.shape)