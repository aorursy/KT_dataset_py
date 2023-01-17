import pandas as pd

from nltk.tokenize import RegexpTokenizer

import gensim

from gensim.models import Word2Vec

from gensim.models import KeyedVectors
train = pd.read_csv('../input/my-data-set/train.csv')
train['mod'] = train.TITLE+train.ABSTRACT
sentences = train['mod'].astype('str').tolist()
tokenizer = RegexpTokenizer(r'\w+')

sentence_tokens = [w.lower() for w in sentences]

sentence_tokens=[tokenizer.tokenize(i) for i in sentence_tokens]

 
model_2 = Word2Vec(size=300, min_count=1)

model_2.build_vocab(sentence_tokens)

total_examples = model_2.corpus_count

model_2.intersect_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)

model_2.train(sentence_tokens, total_examples=total_examples, epochs=5)
# gensim flavored word2vec model (smaller)

model_2.save('Vidya_word2vec_gensim.model')


# generic word2vec model

model_2.wv.save_word2vec_format("Vidya_generic_word2vec.model")
model_2.wv.word_vec