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
! wget https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt
with open('./google-10000-english.txt') as f:

    lines = [line.rstrip() for line in f]
print(lines)
elem_to_be_removed = []

for elem in lines:

    if len(elem) <3:

        elem_to_be_removed.append(elem)

        
print(elem_to_be_removed)
for elem in elem_to_be_removed:

    lines.remove(elem)
print(len(lines))
import random

random_words = random.sample(lines, 1000)
print(len(random_words))
list_to_add = ['junky', 'funky', 'funk', 'flunky', 'funny', 'ritual', 'viral', 'rival', 'vital', 'anger', 'genoa', 'bakery', 'abord', 'criddle', 'fiddle', 'riddle' 'siddall', 'railings' ,'tailings']
dataset = random_words + list_to_add
from fuzzywuzzy import fuzz

from fuzzywuzzy import process
random.shuffle(dataset)
input_word = 'virtual'
fuzz.ratio(input_word, dataset[0])
score_dict = {}

for word in dataset:

    score = fuzz.ratio(input_word, word)

    score_dict[word] = score

    

    
sorted_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)}
output_dict_initial = {}

for key, value in sorted_dict.items():

    if value > 50:

        output_dict_initial[key] = value
print(output_dict_initial)
from nltk.corpus import wordnet as wn
# n    NOUN 

# v    VERB 

# a    ADJECTIVE 

# s    ADJECTIVE SATELLITE 

# r    ADVERB


input_word_pos = wn.synsets(input_word)[0].pos()

matching_pos_dict = {}

non_matching_pos_dict = {}

initial_word_pos = wn.synsets(input_word)[0].pos()

for key, value in output_dict_initial.items():

    try:

        temp_pos = wn.synsets(key)[0].pos()

        if temp_pos == input_word_pos:

            matching_pos_dict[key] = value

        else:

            non_matching_pos_dict[key] = value

    except:

        non_matching_pos_dict[key] = value
print(matching_pos_dict)
print(non_matching_pos_dict)
key_list = []

for key, value in matching_pos_dict.items():

    key_list.append(key)

    

for key, value in non_matching_pos_dict.items():

    key_list.append(key)
print(key_list)
len_diff_list = []

input_word_len = len(input_word)

for key in key_list:

    key_len = len(key)

    diff_len = abs(key_len-input_word_len)

    len_diff_list.append((key, diff_len))

    

    
print(len_diff_list)
# Function to sort the list of tuples by its second item 

def Sort_Tuple(tup):  

      

    # getting length of list of tuples 

    lst = len(tup)  

    for i in range(0, lst):  

          

        for j in range(0, lst-i-1):  

            if (tup[j][1] > tup[j + 1][1]):  

                temp = tup[j]  

                tup[j]= tup[j + 1]  

                tup[j + 1]= temp  

    return tup 
sorted_tuple = Sort_Tuple(len_diff_list)
print(sorted_tuple)
output = []



for diff_tuple in sorted_tuple:

    output.append(diff_tuple[0])
print(output)
!pip install textdistance
import textdistance
textdistance.jaccard(input_word, dataset[0])
score_dict = {}

for word in dataset:

    score = textdistance.jaccard(input_word, word)

    score_dict[word] = score

    

    
print(score_dict)
sorted_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)}
print(sorted_dict)
output_dict_initial = {}

for key, value in sorted_dict.items():

    if value > 0.6:

        output_dict_initial[key] = value
print(output_dict_initial)


input_word_pos = wn.synsets(input_word)[0].pos()

matching_pos_dict = {}

non_matching_pos_dict = {}

initial_word_pos = wn.synsets(input_word)[0].pos()

for key, value in output_dict_initial.items():

    try:

        temp_pos = wn.synsets(key)[0].pos()

        if temp_pos == input_word_pos:

            matching_pos_dict[key] = value

        else:

            non_matching_pos_dict[key] = value

    except:

        non_matching_pos_dict[key] = value
print(matching_pos_dict)
print(non_matching_pos_dict)
key_list = []

for key, value in matching_pos_dict.items():

    key_list.append(key)

    

for key, value in non_matching_pos_dict.items():

    key_list.append(key)
print(key_list)
len_diff_list = []

input_word_len = len(input_word)

for key in key_list:

    key_len = len(key)

    diff_len = abs(key_len-input_word_len)

    len_diff_list.append((key, diff_len))

    

    
print(len_diff_list)
# Function to sort the list of tuples by its second item 

def Sort_Tuple(tup):  

      

    # getting length of list of tuples 

    lst = len(tup)  

    for i in range(0, lst):  

          

        for j in range(0, lst-i-1):  

            if (tup[j][1] > tup[j + 1][1]):  

                temp = tup[j]  

                tup[j]= tup[j + 1]  

                tup[j + 1]= temp  

    return tup 
sorted_tuple = Sort_Tuple(len_diff_list)
print(sorted_tuple)
output = []



for diff_tuple in sorted_tuple:

    output.append(diff_tuple[0])
print(output)
abs(textdistance.needleman_wunsch(input_word, dataset[0]))
score_dict = {}

for word in dataset:

    score = abs(textdistance.needleman_wunsch(input_word, word))

    score_dict[word] = score

    

    
print(score_dict)
sorted_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)}
print(sorted_dict)
textdistance.mra(input_word, dataset[0])
score_dict = {}

for word in dataset:

    score = textdistance.mra(input_word, word)

    score_dict[word] = score

    

    
print(score_dict)
sorted_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)}
print(sorted_dict)
## Not that good result