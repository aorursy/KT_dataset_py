import io



import pandas as pd

import numpy as np



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

def build_matrix(word_index, word_counts, embedding_dim=300):

    data = {}

    unknown_words = dict()    

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    # Priority for fasttext.

    for f in ['../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

              '../input/glove840b300dtxt/glove.840B.300d.txt']:

        print(f'Load {f} ')

        

        fin = io.open(f, 'r', encoding='utf-8', newline='\n', errors='ignore')    

        for line in fin:

            tokens = line.rstrip().split(' ')

            word, vector = tokens[0], np.asarray(tokens[1:], dtype='float32')

            # if we have this word - save it 

            if word_index.get(word):

                # if we saved it before, skip saving (for example: we find in fasttext and gloves)

                if data.get(word) is None:

                    data[word_index.get(word)] = vector

            # Maybe the word is written only with a capital letter.

            if word_index.get(word.lower()):

                if data.get(word.lower()) is None:

                    data[word_index.get(word.lower())] = vector

    

    print("Generate matrix")

    

    for word, i in word_index.items():

        embedding_vector = data.get(i)

        if embedding_vector is not None:        

            embedding_matrix[i] = embedding_vector

        else:

            unknown_words[word] = word_counts[word]

            

    print(f'Found embeddings for {1-len(unknown_words)/len(word_counts):.2%} of vocablen')

    print(f'Found embeddings for {1-sum(unknown_words.values())/sum(word_counts.values()):.2%} of all text')

        

    return embedding_matrix, sorted(unknown_words.items(), key= lambda x : x[1], reverse=True)
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
%%time

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train.comment_text.tolist() + test.comment_text.tolist())

embedding_matrix, unknown_words = build_matrix(tokenizer.word_index, tokenizer.word_counts)
unknown_words[:10]