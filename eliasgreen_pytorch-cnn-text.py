import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np





import nltk

from nltk.corpus import stopwords

# nltk.download('stopwords')

from nltk.util import ngrams

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

# nltk.download('punkt')



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



import re

import string

import os

from collections import defaultdict

from collections import Counter



plt.style.use('ggplot')

stop = set(stopwords.words('english'))



import gensim

from tqdm.notebook import tqdm
sample_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

train = pd.read_csv('../input/nlp-getting-started/train.csv')
train.head()
train.loc[1]
def create_corpus(target):

    corpus = []

    

    for x in train.loc[train['target'] == target, 'text'].str.split():

        for i in x:

            corpus.append(i)

            

    return corpus
corpus = create_corpus(0)



dic = defaultdict(int)



for word in corpus:

    if word in stop:

        dic[word] += 1

        

top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]



x, y = zip(*top)

plt.title('With no disaster')

plt.bar(x, y)
corpus = create_corpus(1)



dic = defaultdict(int)



for word in corpus:

    if word in stop:

        dic[word] += 1

        

top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]



x, y = zip(*top)



plt.title('With disaster')

plt.bar(x, y, color='green')
plt.figure(figsize=(10, 5))

corpus = create_corpus(0)



dic = defaultdict(int)



special = string.punctuation

for i in corpus:

    if i in special:

        dic[i] += 1

        

x, y = zip(*dic.items())

plt.bar(x, y)
plt.figure(figsize=(10, 5))

corpus = create_corpus(0)



dic = defaultdict(int)



special = string.punctuation

for i in corpus:

    if i in special:

        dic[i] += 1

        

x, y = zip(*dic.items())

plt.bar(x, y, color='green')
counter = Counter(corpus)

most_common = counter.most_common()



x = list()

y = list()



for word, count in most_common[:40]:

    if word not in stop:

        x.append(word)

        y.append(count)

        

sns.barplot(x=y, y=x, orient='h')
def get_top_tweet_bigrams(corpus, n=10):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    

    return words_freq[:n]
plt.figure(figsize=(10, 5))

top_tweet_bigrams = get_top_tweet_bigrams(train['text'])[:10]



x, y = map(list, zip(*top_tweet_bigrams))



sns.barplot(x=y, y=x)
df = pd.concat([train, test])

df.shape
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    

    return url.sub('', text)



example = 'New competition launched: https://www.kaggle.com/c/nlp-getting-started'



remove_URL(example)
df['text'] = df['text'].apply(lambda x: remove_URL(x))
example = """<div>

<h1>Real or Fake</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""
def remove_html(text):

    html = re.compile(r'<.*?>')

    

    return html.sub('', text)



print(remove_html(example))
df['text'] = df['text'].apply(lambda x: remove_html(x))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    

    return emoji_pattern.sub(r'', text)





remove_emoji("Omg another Earthquake 😔😔")
df['text'] = df['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table = str.maketrans('', '', string.punctuation)

    

    return text.translate(table)



example = "I am #king"

print(remove_punct(example))
df['text'] = df['text'].apply(lambda x: remove_punct(x))
keywords = train.keyword.unique()[1:]

keywords = list(map(lambda x: x.replace('%20', ' '), keywords))



wnl = WordNetLemmatizer()



def lemmatize_sentence(sentence):

    sentence_words = sentence.split(' ')

    new_sentence_words = list()

    

    for sentence_word in sentence_words:

        sentence_word = sentence_word.replace('#', '')

        new_sentence_word = wnl.lemmatize(sentence_word.lower(), wordnet.VERB)

        new_sentence_words.append(new_sentence_word)

        

    new_sentence = ' '.join(new_sentence_words)

    new_sentence = new_sentence.strip()

    

    return new_sentence
df['text'] = df['text'].apply(lambda x: lemmatize_sentence(x))
import torch

from torch.nn import functional as F

from torch.autograd import Variable



from torchtext import data

from torchtext import datasets

from torchtext.vocab import Vectors, GloVe
def prepare_csv(df_train, df_test, seed=27, val_ratio=0.3):

    idx = np.arange(df_train.shape[0])

    

    np.random.seed(seed)

    np.random.shuffle(idx)

    

    val_size = int(len(idx) * val_ratio)

    

    if not os.path.exists('cache'):

        os.makedirs('cache')

    

    df_train.iloc[idx[val_size:], :][['id', 'target', 'text']].to_csv(

        'cache/dataset_train.csv', index=False

    )

    

    df_train.iloc[idx[:val_size], :][['id', 'target', 'text']].to_csv(

        'cache/dataset_val.csv', index=False

    )

    

    df_test[['id', 'text']].to_csv('cache/dataset_test.csv',

                   index=False)
def get_iterator(dataset, batch_size, train=True,

                 shuffle=True, repeat=False):

    

    device = torch.device('cuda:0' if torch.cuda.is_available()

                          else 'cpu')

    

    dataset_iter = data.Iterator(

        dataset, batch_size=batch_size, device=device,

        train=train, shuffle=shuffle, repeat=repeat,

        sort=False

    )

    

    return dataset_iter
import logging

from copy import deepcopy



LOGGER = logging.getLogger('tweets_dataset')



def get_dataset(fix_length=100, lower=False, vectors=None):

    

    if vectors is not None:

        lower=True

        

    LOGGER.debug('Preparing CSV files...')

    prepare_csv(train, test)

    

    TEXT = data.Field(sequential=True, 

#                       tokenize='spacy', 

                      lower=True, 

                      include_lengths=True, 

                      batch_first=True, 

                      fix_length=25)

    LABEL = data.Field(use_vocab=True,

                       sequential=False,

                       dtype=torch.float16)

    ID = data.Field(use_vocab=False,

                    sequential=False,

                    dtype=torch.float16)

    

    

    LOGGER.debug('Reading train csv files...')

    

    train_temp, val_temp = data.TabularDataset.splits(

        path='cache/', format='csv', skip_header=True,

        train='dataset_train.csv', validation='dataset_val.csv',

        fields=[

            ('id', ID),

            ('target', LABEL),

            ('text', TEXT)

        ]

    )

    

    LOGGER.debug('Reading test csv file...')

    

    test_temp = data.TabularDataset(

        path='cache/dataset_test.csv', format='csv',

        skip_header=True,

        fields=[

            ('id', ID),

            ('text', TEXT)

        ]

    )

    

    LOGGER.debug('Building vocabulary...')

    

    TEXT.build_vocab(

        train_temp, val_temp, test_temp,

        max_size=20000,

        min_freq=10,

        vectors=GloVe(name='6B', dim=300)  # We use it for getting vocabulary of words

    )

    LABEL.build_vocab(

        train_temp

    )

    ID.build_vocab(

        train_temp, val_temp, test_temp

    )

    

    word_embeddings = TEXT.vocab.vectors

    vocab_size = len(TEXT.vocab)

    

    train_iter = get_iterator(train_temp, batch_size=32, 

                              train=True, shuffle=True,

                              repeat=False)

    val_iter = get_iterator(val_temp, batch_size=32, 

                            train=True, shuffle=True,

                            repeat=False)

    test_iter = get_iterator(test_temp, batch_size=32, 

                             train=False, shuffle=False,

                             repeat=False)

    

    

    LOGGER.debug('Done preparing the datasets')

    

    return TEXT, vocab_size, word_embeddings, train_iter, val_iter, test_iter
TEXT, vocab_size, word_embeddings, train_iter, val_iter, test_iter = get_dataset()
class StackedConv1d(torch.nn.Module):

    def __init__(self, features_num, layers_n=5, kernel_size=3, conv_layer=torch.nn.Conv1d, dropout=0.2):

        super().__init__()

        layers = []

        for _ in range(layers_n):

            layers.append(torch.nn.Sequential(

                conv_layer(features_num, features_num, kernel_size, padding=kernel_size//2),

                torch.nn.Dropout(dropout),

                torch.nn.LeakyReLU()))

        self.layers = torch.nn.ModuleList(layers)

    

    def forward(self, x):

        """x - BatchSize x FeaturesNum x SequenceLen"""

        for layer in self.layers:

            x = x + layer(x)

        return x
class СNNClassifier(torch.nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, n_layers, weights):

        super(СNNClassifier, self).__init__()

        

        self.output_size = output_size

        self.n_layers = n_layers

        

        self.word_embeddings = torch.nn.Embedding(vocab_size,

                                                  embedding_dim)

        self.word_embeddings.weight = torch.nn.Parameter(weights,

                                                         requires_grad=False)

        

        self.backbone = StackedConv1d(embedding_dim)

        self.global_pooling = torch.nn.AdaptiveMaxPool1d(1)

        

        #self.dropout_1 = torch.nn.Dropout(0.05)

        

        #self.conv_1 = torch.nn.Conv1d(in_channels=25, out_channels=25, kernel_size= 3, stride=1, padding=1)

        #self.maxPool_1 = torch.nn.MaxPool1d(2, 2)

        #self.relu_1 = torch.nn.ReLU()

        

        #self.dropout_2 = torch.nn.Dropout(0.05)

        

        #self.conv_2 = torch.nn.Conv1d(in_channels=25, out_channels=25, kernel_size= 3, stride=1, padding=1)

        #self.maxPool_2 = torch.nn.MaxPool1d(2, 2)

        #self.relu_2 = torch.nn.ReLU()

        

        self.label_layer = torch.nn.Linear(embedding_dim, output_size)

        self.act = torch.nn.Sigmoid()

        

    def forward(self, x):

        #batch_size = x.size(0)

        

        sent_embeddings = self.word_embeddings(x)

        sent_embeddings = sent_embeddings.permute(0, 2, 1)

        features = self.backbone(sent_embeddings)

        global_features = self.global_pooling(features).squeeze(-1)

        #x = self.dropout_1(x)



        #out_from_conv = self.conv_1(x)

        #out_from_conv = self.maxPool_1(out_from_conv)

        #out_from_conv = self.relu_1(out_from_conv)

        

        #out_from_conv = self.dropout_2(out_from_conv)

        

        #out_from_conv = self.conv_2(out_from_conv)

        #out_from_conv = self.maxPool_2(out_from_conv)

        #out_from_conv = self.relu_2(out_from_conv)

        

        #out_from_conv = out_from_conv.view(-1, 25*75)

        

        out = self.label_layer(global_features)

        out = self.act(out)

        

        return out
def train_model(model, train_iter, val_iter, optim, loss, num_epochs, batch_size=32):

    clip = 5

    val_loss_min = np.Inf

    

    total_train_epoch_loss = list()

    total_train_epoch_acc = list()

        

    total_val_epoch_loss = list()

    total_val_epoch_acc = list()

        

    

    device = torch.device('cuda:0' if torch.cuda.is_available()

                           else 'cpu')

    

    for epoch in range(num_epochs):



        model.train()

        

        train_epoch_loss = list()

        train_epoch_acc = list()

        

        val_epoch_loss = list()

        val_epoch_acc = list()

        

        for idx, batch in enumerate(tqdm(train_iter, position=0, leave=True)):

            text = batch.text[0]

            target = batch.target

            target = target - 1

            target = target.type(torch.LongTensor)



            text = text.to(device)

            target = target.to(device)



            optim.zero_grad()

            

            if text.size()[0] is not batch_size:

                continue

            

            prediction = model(text)

            #print(prediction.squeeze())

            #print(target)

            loss_train = loss(prediction.squeeze(), target)

            loss_train.backward()



            num_corrects = (torch.max(prediction, 1)[1].

                                view(target.size()).data == target.data).float().sum()



            acc = 100.0 * num_corrects / len(batch)



            train_epoch_loss.append(loss_train.item())

            train_epoch_acc.append(acc.item())

            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            

            optim.step()

    

        print(f'Train Epoch: {epoch}, Training Loss: {np.mean(train_epoch_loss):.4f}, Training Accuracy: {np.mean(train_epoch_acc): .2f}%')



        model.eval()



        with torch.no_grad():

            for idx, batch in enumerate(tqdm(val_iter, position=0, leave=True)):



                text = batch.text[0]

                target = batch.target

                target = target - 1

                target = target.type(torch.LongTensor)

                

                text = text.to(device)

                target = target.to(device)

                

                if text.size()[0] is not batch_size:

                    continue



                prediction = model(text)

                loss_val = loss(prediction.squeeze(), target)



                num_corrects = (torch.max(prediction, 1)[1].

                                view(target.size()).data == target.data).float().sum()



                acc = 100.0 * num_corrects / len(batch)



                val_epoch_loss.append(loss_val.item())

                val_epoch_acc.append(acc.item())

                

            print(f'Vadlidation Epoch: {epoch}, Training Loss: {np.mean(val_epoch_loss):.4f}, Training Accuracy: {np.mean(val_epoch_acc): .2f}%')

                

            if np.mean(val_epoch_loss) <= val_loss_min:

#                 torch.save(model.state_dict(), 'state_dict.pth')

                print('Validation loss decreased ({:.6f} --> {:.6f})'.

                      format(val_loss_min, np.mean(val_epoch_loss)))

                

                val_loss_min = np.mean(val_epoch_loss)

                

        total_train_epoch_loss.append(np.mean(train_epoch_loss))

        total_train_epoch_acc.append(np.mean(train_epoch_acc))

    

        total_val_epoch_loss.append(np.mean(val_epoch_loss))

        total_val_epoch_acc.append(np.mean(val_epoch_acc))

    

    return (total_train_epoch_loss, total_train_epoch_acc,

            total_val_epoch_loss, total_val_epoch_acc)
lr = 1e-4

batch_size = 32

output_size = 2

embedding_length = 300

num_epochs = 6



model = СNNClassifier(vocab_size=vocab_size, 

                       output_size=output_size, 

                       embedding_dim=embedding_length,

                       n_layers=2,

                       weights=word_embeddings

)



device = torch.device('cuda:0' if torch.cuda.is_available()

                      else 'cpu')

    

model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=lr)

loss = torch.nn.CrossEntropyLoss()

    

train_loss, train_acc, val_loss, val_acc = train_model(model=model,

                                                       train_iter=train_iter,

                                                       val_iter=val_iter,

                                                       optim=optim,

                                                       loss=loss,

                                                       num_epochs=num_epochs,

                                                       batch_size=batch_size)

    
plt.figure(figsize=(10, 6))

plt.title('Loss')

sns.lineplot(range(len(train_loss)), train_loss, label='train')

sns.lineplot(range(len(val_loss)), val_loss, label='test')
plt.figure(figsize=(10, 6))

plt.title('Accuracy')

sns.lineplot(range(len(train_acc)), train_acc, label='train')

sns.lineplot(range(len(val_acc)), val_acc, label='test')
results_target = list()



with torch.no_grad():

    for batch in tqdm(test_iter):

        for text, idx in zip(batch.text[0], batch.id):

            text = text.unsqueeze(0)

            res = model(text)



            target = np.round(res.cpu().numpy())

            

            results_target.append(target[0][1])
sample_submission['target'] = list(map(int, results_target))
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)