!pip install fastai

!pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
import numpy as np 

import pandas as pd 

from fastai import *

from fastai.text import *
gita = pd.read_csv('../input/bhagavad-gita.csv')
print(gita.head)

print(gita.keys())
print(gita.shape)
#quote = quote.drop(quote.columns[[0,3,4,5]], axis=1)
gita['devanagari'] = gita['devanagari'].str.lower()

print(gita.head())
from pandas import DataFrame

verse = DataFrame.drop_duplicates(DataFrame(gita, columns= ['devanagari']))

verse.head()
test_per = 0.05        # test percentage

verse = verse.iloc[np.random.permutation(len(verse))]

piece = int(test_per * len(verse)) + 1

train_verse, test_verse = verse[piece:], verse[:piece]
print(len(train_verse),len(test_verse))
# Pre-processing

nan_row = verse[verse['devanagari'].isnull()]

print(nan_row)
data_lm = TextLMDataBunch.from_df('content', train_verse, test_verse, text_cols = 'devanagari')

#data_clas = TextClasDataBunch.from_df('content', train_quote, test_quote, vocab = data_lm.train_ds.vocab, bs = 32)

#data_lm.save('data_lm_export.pkl')

#data_clas.save('data_clas_export.pkl')
learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult = 0.5)

learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(1, 1e-3)
learn.fit(20, lr=1e-4, wd=1e-7)
number_of_quotes = 50

quotes_counter = 0

all_quotes = []



for i in range(1000):

    quote = learn.predict("xxbos ", n_words=20, temperature=0.8)

    quotes = quote.split("xxbos ")

    quotes = quotes[1:-1]

    

    for quote in quotes:

        if ("\r" or "\n" or "॥") in quote:

            quote = quote.replace("\r","")

            quote = quote.replace("\n", "")

            quote = quote.replace("॥","")

        quote = quote.replace("xxbos ","").strip()

        quote = quote.replace("Xxbos ","").strip()

        if(quote):

            all_quotes.append(quote)

            quotes_counter = quotes_counter+1

            

    if quotes_counter > number_of_quotes:

        break
all_quotes
outF = open("verses.txt", "w")

j = 0

for verse in all_quotes:

  # write line to output file

  j=j+1

  outF.write(verse)

  outF.write("\n "+str(j)+" \n\n")

outF.close()