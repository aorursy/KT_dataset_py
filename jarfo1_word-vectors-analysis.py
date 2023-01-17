import numpy as np

import pickle

import torch



# Get the interactive Tools for Matplotlib

%matplotlib inline

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [9.5, 6]
class Vocabulary(object):

    def __init__(self, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'):

        self.token2idx = {}

        self.idx2token = []

        self.pad_token = pad_token

        self.unk_token = unk_token

        self.eos_token = eos_token

        if pad_token is not None:

            self.pad_index = self.add_token(pad_token)

        if unk_token is not None:

            self.unk_index = self.add_token(unk_token)

        if eos_token is not None:

            self.eos_index = self.add_token(eos_token)



    def add_token(self, token):

        if token not in self.token2idx:

            self.idx2token.append(token)

            self.token2idx[token] = len(self.idx2token) - 1

        return self.token2idx[token]



    def get_index(self, token):

        if isinstance(token, str):

            return self.token2idx.get(token, self.unk_index)

        else:

            return [self.token2idx.get(t, self.unk_index) for t in token]



    def __len__(self):

        return len(self.idx2token)



    def save(self, filename):

        with open(filename, 'wb') as f:

            pickle.dump(self.__dict__, f)



    def load(self, filename):

        with open(filename, 'rb') as f:

            self.__dict__.update(pickle.load(f))
DATASET_VERSION = 'ca-100'

CBOW_VOCABULARY_ROOT = f'../input/cbow-preprocessing/data/{DATASET_VERSION}'

CBOW_VECTORS_ROOT = f'../input/cbow-training/data/{DATASET_VERSION}'
dict = f'{CBOW_VOCABULARY_ROOT}/ca.wiki.train.tokens.nopunct.dic'

counter = pickle.load(open(dict, 'rb'))

words, values = zip(*counter.most_common(5000))

print('Most frequent Catalan words')

print(words[:10])

print(values[:10])
_ = plt.plot(values[:50], 'g', 2*values[0]/np.arange(2,52), 'r')
_ = plt.loglog(values)

plt.show()
from collections import Counter

benford = Counter(int(str(item[1])[0]) for item in counter.most_common(5000))

print(benford)

percentage = np.array(list(benford.values()), dtype=np.float)

percentage /= percentage.sum()

_ = plt.bar(list(benford.keys()), percentage*100)
modelname = f'{CBOW_VECTORS_ROOT}/{DATASET_VERSION}.pt'

state_dict = torch.load(modelname, map_location=torch.device('cpu'))
state_dict.keys()
input_word_vectors = state_dict['emb.weight'].numpy()

output_word_vectors = state_dict['lin.weight'].numpy()
token_vocab = Vocabulary()

token_vocab.load(f'{CBOW_VOCABULARY_ROOT}/ca.wiki.vocab')
class WordVectors:

    def __init__(self, vectors, vocabulary):

        # TODO 

        self.vocabulary = vocabulary

    

    def most_similar(self, word, topn=10):

        # TODO

        return [

            ('valencià', 0.8400525),

            ('basc', 0.75919044),

            ('gallec', 0.7418786),

            ('mallorquí', 0.73923385),

            ('castellà', 0.69002914),

            ('francès', 0.6782388),

            ('espanyol', 0.6611247),

            ('bretó', 0.641976),

            ('aragonès', 0.6250948),

            ('andalús', 0.6203275)

        ]

    

    def analogy(self, x1, x2, y1, topn=5, keep_all=False):

        # If keep_all if False we remove the input words (x1, x2, y1) from the returned closed words

        # TODO

        return [

            ('polonès', 0.9679756),

            ('suec', 0.9589857),

            ('neerlandès', 0.95811903),

            ('rus', 0.95155054),

            ('txec', 0.950968),

            ('basc', 0.94935954),

            ('danès', 0.94827694),

            ('turc', 0.9475782)

        ]
model1 = WordVectors(input_word_vectors, token_vocab)

model2 = WordVectors(output_word_vectors, token_vocab)
model1.most_similar('català')
model2.analogy('França', 'francès', 'Polònia')