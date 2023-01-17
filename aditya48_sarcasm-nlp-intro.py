# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json
# with open("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json","r") as f:

#     datastore = json.load(f)
data = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json",lines=True)

data.head()
data.shape
# This will be treated as a training data.

sentences = list(data['headline'])

labels = list(data['is_sarcastic'])



# Will not use for this case.

urls = list(data['article_link'])
## Run this cell to check the sentences list:

#sentences
print(len(sentences))

print(len(labels))
## Let's develop a word index using Tokenizer

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences





tokenizer = Tokenizer(oov_token= "<OOV'>")   

## OOV stands for Out of Vocabulary

## Used for labelling words which are not in Word-Index or Vocabulary of the model.



## Fitting the tokenizer on the sentences:

tokenizer.fit_on_texts(sentences)

# Sentences here contains the headlines of sarcasm dataset.



## Now we will create Vocabulary or Word_index.

## This will be a dictionary.

## With words as key and Index as values.

word_index = tokenizer.word_index

## This will numbered all the words present in the headlines

## Also there will be no duplicacy in the Index.





print("Number of Unique Words in the Headlines: ",len(word_index))
print(type(word_index))

# word_index
## Now let's develop the Sequences:



sequences = tokenizer.texts_to_sequences(sentences)

print("The Number of Sequences is Equal to : ",len(sequences))

print()

print("First ten sequences are: ")

print()

for i in range(10):

    print(sequences[i])

    print()
test=[

    "i love my country"

]



test_seq = tokenizer.texts_to_sequences(test)

print("Sequence for :",test[0],test_seq)



print(word_index['i'])

print(word_index['love'])

print(word_index['my'])

print(word_index['country'])
from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt



word = str(sentences[::])

## As sentences here is a list with multiple strings(used above)

## So wordcloud can only be fed strings and not list that have strings.

## So every string in the sentences is put into a single variable word here

## Which can be fed into the wordcloud.





stopwords = set(STOPWORDS) 

wordcloud = WordCloud(width = 1600, height = 1600, 

                background_color ='black', 

                stopwords = stopwords).generate(word) 

  

# plotting the Wordcloud Image                        

plt.figure(figsize = (10, 10)) 

plt.imshow(wordcloud) 

plt.axis("off")

## Used to remove the axis. As by-default matplolib gives Axis in all it's Plots





plt.title("HEADLINES")

plt.show()