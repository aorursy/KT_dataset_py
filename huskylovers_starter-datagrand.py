import matplotlib.pyplot as plt # plotting

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

print(os.listdir('../input/datagrand/datagrand/'))
with open('../input/datagrand/datagrand/train.txt') as f:

    train_num = len(f.readlines())

with open('../input/datagrand/datagrand/test.txt') as f:

    test_num = len(f.readlines())

print("train lines:" + str(train_num))

print("test lines :" + str(test_num))
with open('../input/datagrand/datagrand/train.txt') as f:

    lines = f.readlines()

    for sentence in lines[:5]:

        print(sentence)
tag_a_length = []

tag_b_length = []

tag_c_length = []

a_sentence_length = []

b_sentence_length = []

c_sentence_length = []

word_set = set([])

with open('../input/datagrand/datagrand/train.txt') as f:

    lines = f.readlines()

    for sentence in lines:

        #print(sentence)

        words = sentence.replace('\n','').split('  ')

        for j,word in enumerate(words):

            split_word = word.split('/')

            tag = split_word[1]

            word_meta = split_word[0]

            word_meta_split = word.split('_')

            word_set = word_set | set(word_meta_split)

            meta_len = len(word_meta_split)

            if tag == 'a':

                tag_a_length.append(meta_len)

                a_sentence_length.append(len(sentence))

            if tag == 'b':

                tag_b_length.append(meta_len)

                b_sentence_length.append(len(sentence))

            if tag == 'c':

                tag_c_length.append(meta_len)

                c_sentence_length.append(len(sentence))

            
print('tag_A_num:  ' + str(len(tag_a_length)))

print('tag_B_num:  ' + str(len(tag_b_length)))

print('tag_C_num:  ' + str(len(tag_c_length)))
plt.figure(figsize=(20,3))

plt.subplot(1,3,1)

sns.countplot(tag_a_length)

plt.title('tag_A_length')

plt.subplot(1,3,2)

sns.countplot(tag_b_length)

plt.title('tag_B_length')

plt.subplot(1,3,3)

sns.countplot(tag_c_length)

plt.title('tag_C_length')

plt.show()
plt.figure(figsize=(20,3))

plt.subplot(1,3,1)

sns.countplot(a_sentence_length)

plt.title('tag_A_sentence_length')

plt.subplot(1,3,2)

sns.countplot(b_sentence_length)

plt.title('tag_B_sentence_length')

plt.subplot(1,3,3)

sns.countplot(c_sentence_length)

plt.title('tag_C_sentence_length')

plt.show()
print('A tag situation')

print(pd.Series(a_sentence_length).describe())

print('B tag situation')

print(pd.Series(b_sentence_length).describe())

print('C tag situation')

print(pd.Series(c_sentence_length).describe())
with open('../input/datagrand/datagrand/train.txt') as f:

    with open('../train_label.txt', 'w') as fw: 

        for line in f.readlines():

            delimiter = '\t'

            words = line.replace('\n','').split('  ')

            for j,word in enumerate(words):

                split_word = word.split('/')

                tag = split_word[1]

                word_meta = split_word[0]

                word_meta_split = word_meta.split('_')

                meta_len = len(word_meta_split)

                if tag == 'a':

                    if meta_len == 1:

                        fw.write(word_meta_split[0] + delimiter + 'W_a' + '\n')

                    else:

                        for k, char in enumerate(word_meta_split):



                            if k == 0:

                                fw.write(char + delimiter + 'B_a' + '\n')

                            elif k == meta_len - 1:

                                fw.write(char + delimiter + 'E_a' + '\n')

                                #fw.write(char + delimiter + 'I_a' + '\n')

                            else:

                                fw.write(char + delimiter + 'M_a' + '\n')

                                #fw.write(char + delimiter + 'I_a' + '\n')

                if tag == 'b':

                    if meta_len == 1:

                        fw.write(word_meta_split[0] + delimiter + 'W_b' + '\n')

                    else:

                        for k, char in enumerate(word_meta_split):

                            if k == 0:

                                fw.write(char + delimiter + 'B_b' + '\n')

                            elif k == meta_len - 1:

                                fw.write(char + delimiter + 'E_b' + '\n')

                                #fw.write(char + delimiter + 'I_a' + '\n')

                            else:

                                fw.write(char + delimiter + 'M_b' + '\n')

                                #fw.write(char + delimiter + 'I_a' + '\n')

                if tag == 'c':

                    if meta_len == 1:

                        fw.write(word_meta_split[0] + delimiter + 'W_c' + '\n')

                    else:

                        for k, char in enumerate(word_meta_split):



                            if k == 0:

                                fw.write(char + delimiter + 'B_c' + '\n')

                            elif k == meta_len - 1:

                                fw.write(char + delimiter + 'E_c' + '\n')

                                #fw.write(char + delimiter + 'I_a' + '\n')

                            else:

                                fw.write(char + delimiter + 'M_c' + '\n')

                                #fw.write(char + delimiter + 'I_a' + '\n')

                else:

                    if meta_len == 1:

                        fw.write(word_meta_split[0] + delimiter + 'O' + '\n')

                    else:

                        for k, char in enumerate(word_meta_split):

                            fw.write(char + delimiter + 'O' + '\n')

            fw.write('\n')

    
language_set = set([])
with open('../input/datagrand/datagrand/corpus.txt') as f:

    lines = f.readlines()

    for sentence in tqdm(lines):

        words = sentence.split('_')

        language_set = set(words) | language_set
percent = len(language_set & word_set) / len(word_set)

print(str(percent)+'%in embedding')  