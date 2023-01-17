# import the libraries

import os

import re

import pandas as pd



# import Python reserved words

python_words = pd.read_json ('../input/list-of-python-31-reserved-words-json/python_words.json')



# declare lists

python_word_list = []

python_word_count = []



# iterate over the dataframe extracting just the words and adding it to the list

for i in range(len(python_words)):

    python_word_list.append (python_words.words[i]['word'].lower())

    python_word_count.append (0) 

# open your code file

code_file = open('../input/python-code-example/this_example.py', 'r')

code_content = code_file.read()



# regex pattern: words

pattern = '[a-zA-Z]+'



match = re.findall(pattern, code_content)



# look for the words used in the code into the words list

for word in match:

    for i in range(len(python_word_list)):

        if word == python_word_list[i]:

            python_word_count[i] = python_word_count[i] +1



# print words and quantities

for i in range(len(python_word_list)):

    print ('{}: {}'.format(python_word_list[i], python_word_count[i]))