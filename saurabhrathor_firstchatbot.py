# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import re
import tensorflow as tf
tf.__version__
southpark = pd.read_csv('../input/All-seasons.csv')[:100]
southpark.head(10)
for i in range(5):
    print('Line #:', i+1)
    print(southpark['Line'][i])
def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"\n", "",  text)
    text = re.sub(r"[-()]", "", text)
    text = re.sub(r"\.", " .", text)
    text = re.sub(r"\!", " !", text)
    text = re.sub(r"\?", " ?", text)
    text = re.sub(r"\,", " ,", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"ohh", "oh", text)
    text = re.sub(r"ohhh", "oh", text)
    text = re.sub(r"ohhhh", "oh", text)
    text = re.sub(r"ohhhhh", "oh", text)
    text = re.sub(r"ohhhhhh", "oh", text)
    text = re.sub(r"ahh", "ah", text)
    
    return text
# Clean the scripts and add them to the same list.
text = []

for line in southpark.Line:
    text.append(clean_text(line))
print(text[:10])
lengths = []
for i in text:
    lengths.append(len(i.split()))
#covert it into dataframe
lengths = pd.DataFrame(lengths, columns=['counts'])
lengths.describe()
print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))
#another way to check length frequency
lengths.hist(bins=8)
# Limit the text we will use to the shorter 95%.
max_line_length = 30

short_text = []
for line in text:
    if len(line.split())<=max_line_length:
        short_text.append(line)
print(len(short_text))
print(len(text))
print(text[0])
print(len(text[0]))
print(text[0].split())
print(len(text[0].split()))
# finding length without splitting it give charcter wise length
# Create a dictionary for the frequency of the vocabulary
vocab = {}
for line in short_text:
    for word in line.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1