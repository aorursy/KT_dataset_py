# importing libs

import torch

import torch.nn as nn

import numpy as np

import nltk

from nltk.tokenize import word_tokenize

import string
# Load file

with open('/kaggle/input/word2vec-data/text8/text8') as f:

    raw_txt = f.read()



# Replace puntuations with corresponding english word

def relace_puntuations(text):

    text = text.lower()

    text = text.replace('.', ' PERIOD ')

    text = text.replace(',', ' COMMA ')

    text = text.replace('"', ' QUOTATION_MARK ')

    text = text.replace(';', ' SEMICOLON ')

    text = text.replace('!', ' EXCLAMATION_MARK ')

    text = text.replace('?', ' QUESTION_MARK ')

    text = text.replace('(', ' LEFT_PAREN ')

    text = text.replace(')', ' RIGHT_PAREN ')

    text = text.replace('--', ' HYPHENS ')

    text = text.replace('?', ' QUESTION_MARK ')

    text = text.replace(':', ' COLON ')

    return text



raw_txt = relace_puntuations(raw_txt)



# split the text into words

words = word_tokenize(raw_txt)



# remove remaining punctuations

words = [ w for w in words if w not in string.punctuation ]



# Remove stopwords ( or frequent words like the, of, a etc ) from words data (Or we could use Milkov Subsampling)

from nltk.corpus import stopwords

stopword_eng = set(stopwords.words('english'))

words = [w for w in words if w not in stopword_eng]



# Remove less frequent words like 

# from collections import Counter

# import random

# word_freq = Counter(words)

# less_freq_words = [ w for w, c in word_freq.items() if c <= 3]

# words = [w for w in words if w not in less_freq_words]



# create lookup dictionary for all words

int2word = dict(enumerate(set(words)))

word2int = {w:i for i, w in int2word.items()}
def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):

    # sim = (a . b) / |a||b|

    

    embed_vectors = embedding.weight

    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))

    valid_examples = np.append(valid_examples,

                               random.sample(range(1000,1000+valid_window), valid_size//2))

    valid_examples = torch.LongTensor(valid_examples).to(device)    

    valid_vectors = embedding(valid_examples)

    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes

        

    return valid_examples, similarities
ecoded_words = [ word2int[w] for w in words]

# get context word for center words

def get_surrounding_words(words, ctx_word_inx, window=5):

    size = np.random.randint(1,window+1)

    start_inx = ctx_word_inx - size if (ctx_word_inx - size) > 0 else 0

    end_inx = start_inx + size

    s_words = words[start_inx:ctx_word_inx] + words[ctx_word_inx+1:end_inx+1]

    return s_words



def get_batch(words, batch_size=50, window=5):

    n_words = len(words)

    total_batch = n_words/batch_size 

    words = words[:int(total_batch*batch_size)]

    for idx in range(0, n_words, batch_size):

        x, y = [], []

        batch_words = words[idx:idx+batch_size]

        for i in range(len(batch_words)):

            s_words = get_surrounding_words(batch_words, i, window)

            c_word = batch_words[i]

            x.extend([c_word]*len(s_words))

            y.extend(s_words)

        yield x, y
class SkipGram(nn.Module):

    def __init__(self, vocabulary_size, embedding_size):

        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        self.fc = nn.Linear(embedding_size, vocabulary_size)

        self.logsoft = nn.LogSoftmax(dim=1)

        

    def forward(self, ctx_word):

        score = self.embedding(ctx_word)

        prob = self.fc(score)

        out = self.logsoft(prob)

        return out
def train(net, words, vocabulary_size, embedding_size,epoch=5 , batch_size=10 ,window_size=5, lr=0.0001):

    net.train()

    criterion = nn.NLLLoss()

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for e in range(epoch):        

        steps = 0

        for x, y in get_batch(words,batch_size,window_size):

            steps +=1

            inp = torch.LongTensor(x)

            label = torch.LongTensor(y)        

            out_prop = net(inp)        

            loss = criterion(out_prop, label)

            net.zero_grad()

            loss.backward()

            optim.step()

            if steps % 500 == 0:

                valid_examples, valid_similarities = cosine_similarity(net.embedding)

                _, closest_idxs = valid_similarities.topk(6)           

                for ii, valid_idx in enumerate(valid_examples):

                    closest_words = [int2word[idx.item()] for idx in closest_idxs[ii]][1:]

                    print(int2word[valid_idx.item()] + " | " + ', '.join(closest_words))

                print("...")
n_vocab = len(int2word)

n_embeddings = 100

model = SkipGram(n_vocab, n_embeddings)

print(model)
train(model, ecoded_words, n_vocab, n_embeddings, epoch=5)
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

embeddings = model.embedding.weight.to('cpu').data.numpy()

viz_words = 100

tsne = TSNE()

embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))

for idx in range(viz_words):

    plt.scatter(*embed_tsne[idx, :], color='steelblue')

    plt.annotate(int2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)