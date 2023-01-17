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
from keras.preprocessing.text import Tokenizer
df = pd.read_csv('../input/train.csv')
max_features = 95000
maxlen = 60

# I'm just going to limit cleaning to lowering the string and putting spaces around stuff for now, you could do far more I guess
def clean_str(x):
    x = str(x)
    x = x.lower()
    
    specials = [',', '?']
    for s in specials:
        x = x.replace(s, f' {s} ')
        
    return x


df['question_text'] = df['question_text'].apply(clean_str)

tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(list(df['question_text'].values))
some_string = "burger king doesn't sell hamberders, does maccy ds?"
some_string = clean_str(some_string)
our_sent = tokenizer.texts_to_sequences([some_string])
our_sent
tokenizer.sequences_to_texts(our_sent)
tokenizer_2 = Tokenizer(num_words=max_features, oov_token='OOV')
tokenizer_2.fit_on_texts(list(df['question_text'].values))
our_sent_2 = tokenizer_2.texts_to_sequences([some_string])
our_sent_2
tokenizer_2.sequences_to_texts(our_sent_2)
tokenizer_3 = Tokenizer(num_words=max_features, oov_token='OOV', filters='!"#$%&()*+,-./:;<=>@[\]^_`{|}~ ')
tokenizer_3.fit_on_texts(list(df['question_text'].values))
our_sent_3 = tokenizer_3.texts_to_sequences([some_string])
our_sent_3
tokenizer_3.sequences_to_texts(our_sent_3)