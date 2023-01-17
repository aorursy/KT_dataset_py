import numpy as np
import os
import pandas as pd
import preprocessor as pp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
df = pd.read_excel('./text_classification_dataset.csv')
df.head()
df['type'].value_counts()
# removing emoticons and urls
pp.set_options(pp.OPT.URL, pp.OPT.EMOJI, pp.OPT.NUMBER)  # to remove urls, numbers and emoticons while cleaning tweets
stop_words = stopwords.words("english")
stop_words.append('@')
stop_words.append('#')
list_of_tweets = []
for row in range(len(df)):
    cleaned_tweet = pp.clean(df.iloc[row,0])
    tweet = word_tokenize(cleaned_tweet.lower())
    list_of_tweets.append(tweet)
    df.iloc[row,0] = " ".join([w for w in tweet if w not in stop_words])
    
df.head()
# create tf-idf values of each word as features
vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
tf_idf_vectors = vectorizer.fit_transform(df['text'])
# encoding the label and splitting it into train_test_set
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
X_train, X_test, y_train, y_test = train_test_split(tf_idf_vectors.toarray(),df['type'].to_numpy(), test_size=0.3, random_state=42)
# applying pca on high dimensional tfidf maxtrix
pca = PCA(n_components=768)
pca_tfidf = pca.fit_transform(tf_idf_vectors.toarray())
pca_X_train, pca_X_test, y_train, y_test = train_test_split(pca_tfidf,df['type'].to_numpy(), test_size=0.3, random_state=42)
# bert embeddings as features

def embed_text(text):
    return np.array(embed.encode(text))

embed = SentenceTransformer('bert-base-nli-mean-tokens')
bert_embeds = embed_text(df['text'])
bert_X_train, bert_X_test, y_train, y_test = train_test_split(bert_embeds,df['type'].to_numpy(), test_size=0.3, random_state=42)
# fitting gaussian models with bert embeddings

gnb = GaussianNB()
gnb.fit(bert_X_train, y_train)
score_train = gnb.score(bert_X_train, y_train)
score_test = gnb.score(bert_X_test, y_test)
print("\nTrain set score for GaussianNB:", score_train)
print("\nTest set score for GaussianNB:", score_test)
# fitting gaussian models with pca tf-idf features

gnb = GaussianNB()
gnb.fit(pca_X_train, y_train)
score_train = gnb.score(pca_X_train, y_train)
score_test = gnb.score(pca_X_test, y_test)
print("\nTrain set score for GaussianNB:", score_train)
print("\nTest set score for GaussianNB:", score_test)
# fitting gaussian models with tf-idf features

gnb = GaussianNB()
gnb.fit(X_train, y_train)
score_train = gnb.score(X_train, y_train)
score_test = gnb.score(X_test, y_test)
print("\nTrain set score for GaussianNB:", score_train)
print("\nTest set score for GaussianNB:", score_test)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
score_train = gnb.score(X_train, y_train)
score_test = gnb.score(X_test, y_test)
print("\nTrain set score for MultinomialNB:", score_train)
print("\nTest set score for MultinomialNB:", score_test)
# fitting logistic regression model with bert embeddings

lr = LogisticRegression()
lr.fit(bert_X_train, y_train)
score_train = lr.score(bert_X_train, y_train)
score_test = lr.score(bert_X_test, y_test)
print("\nTrain set score for LogisticRegression:", score_train)
print("\nTest set score for LogisticRegression:", score_test)
# fitting logistic regression model with pca tf-df features

lr = LogisticRegression()
lr.fit(pca_X_train, y_train)
score_train = lr.score(pca_X_train, y_train)
score_test = lr.score(pca_X_test, y_test)
print("\nTrain set score for LogisticRegression:", score_train)
print("\nTest set score for LogisticRegression:", score_test)
# fitting logistic regression model with tf-idf features

lr = LogisticRegression()
lr.fit(X_train, y_train)
score_train = lr.score(X_train, y_train)
score_test = lr.score(X_test, y_test)
print("\nTrain set score for LogisticRegression:", score_train)
print("\nTest set score for LogisticRegression:", score_test)
# fitting svm classification model with bert embeddings

svc = SVC(gamma='auto')
svc.fit(bert_X_train, y_train)
score_train = svc.score(bert_X_train, y_train)
score_test = svc.score(bert_X_test, y_test)
print("\nTrain set score for SVM:", score_train)
print("\nTest set score for SVM:", score_test)
# fitting svm classification model with pca tf-idf features

svc = SVC(gamma='auto')
svc.fit(pca_X_train, y_train)
score_train = svc.score(pca_X_train, y_train)
score_test = svc.score(pca_X_test, y_test)
print("\nTrain set score for SVM:", score_train)
print("\nTest set score for SVM:", score_test)
# fitting svm classification model with tf-idf features

svc = SVC(gamma='auto')
svc.fit(X_train, y_train)
score_train = svc.score(X_train, y_train)
score_test = svc.score(X_test, y_test)
print("\nTrain set score for SVM:", score_train)
print("\nTest set score for SVM:", score_test)
# fitting RandomForest classification model with bert embeddings

rfc = RandomForestClassifier()
rfc.fit(bert_X_train, y_train)
score_train = rfc.score(bert_X_train, y_train)
score_test = rfc.score(bert_X_test, y_test)
print("\nTrain set score for RandomForestClassifier:", score_train)
print("\nTest set score for RandomForestClassifier:", score_test)
# fitting RandomForest classification model with with pca tf-idf features

rfc = RandomForestClassifier()
rfc.fit(pca_X_train, y_train)
score_train = rfc.score(pca_X_train, y_train)
score_test = rfc.score(pca_X_test, y_test)
print("\nTrain set score for RandomForestClassifier:", score_train)
print("\nTest set score for RandomForestClassifier:", score_test)
# fitting RandomForest classification model with tf-idf features

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
score_train = rfc.score(X_train, y_train)
score_test = rfc.score(X_test, y_test)
print("\nTrain set score for RandomForestClassifier:", score_train)
print("\nTest set score for RandomForestClassifier:", score_test)
# creating LSTM network
class LSTM(nn.Module):
    def __init__(self, embedding_dim, n_hidden, n_outputs, n_hlayers=1, vocab_size=3000):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_hlayers = n_hlayers
        self.embed = nn.Embedding(vocab_size+1,embedding_dim)
        self.layer1 = nn.LSTM(embedding_dim, n_hidden, n_hlayers, dropout=0.5, batch_first=True)
        self.layer2 = nn.Linear(n_hidden, 15)
        self.layer3 = nn.Linear(15, n_outputs)
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x = x.long()
        x = self.embed(x)
        batch_size, hidden_dim = x.shape[0], self.n_hidden
        h,c = torch.zeros((self.n_hlayers,batch_size, hidden_dim)), torch.zeros((self.n_hlayers,batch_size, hidden_dim))
        x, hc = self.layer1(x,(h,c))
        out = self.layer3(self.layer2(x))
        return out[:,-1]      
# function to pad featues to make them equal size
def pad_sequence(reviews_int, seq_length=200):
    features = np.zeros((len(reviews_int), seq_length), dtype=int)
    for ii, review in enumerate(reviews_int):
        features[ii,-len(review):] = np.array(review)[:seq_length]
    return features
# our evaluate func
def evaluate(model, dl, criterion):
    total_loss = 0
    num_correct = 0
    for x,y in dl:
        output = model(x)
        loss = criterion(output,y)
        total_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(y.view_as(pred))
        #correct = np.squeeze(correct.numpy())
        num_correct += np.sum(correct.numpy())
    return total_loss, num_correct/len(dl.dataset)
# our train func
def train(model, train_dl, val_dl, epochs, optimizer, criterion):
    train_loss_list = []
    for e in range(epochs):
        tot_train_loss = 0
        for x,y in train_dl:
            output = model(x)
            loss = criterion(output,y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            tot_train_loss += loss.item()
        print("After {} epcoh:".format(e+1), "Train_Loss:", tot_train_loss,"Val_Loss and Accuracy:", evaluate(model, val_dl, criterion))
# creating vocab2int and mapping tokens to int
words = vectorizer.get_feature_names()
vocab2int ={word:ii for ii, word in enumerate(words,1)}
tweets = []
for row in df.iterrows():
    tweets.append([vocab2int[word] for word in word_tokenize(row[1]['text'].lower())])
# pad tweets to keep the lenght of all tweets to 35
seq_length=35
features = pad_sequence(tweets, seq_length)
assert len(features) == len(tweets)
assert len(features[0]) == seq_length
# loading glove embeddings
embedding_dict = {}
with open('./glove.6B.100d.txt',encoding="utf8")as f:
    for line in f:
        split = line.split()
        word = split[0]
        embeddings = np.array(split[1:], dtype='float32')
        embedding_dict[word] = embeddings
# creating embedding matrix to initialize embedding layer
embed_dim = 100
embedding_matrix = np.zeros((len(words)+1, embed_dim))
embedding_matrix[0] = embedding_dict['pad']
 
for word, index in vocab2int.items():
    try:
        embedding_matrix[index] = embedding_dict[word]
    except:
        embedding_matrix[index] = embedding_dict['unk']
# prepaing dataset and dataloader to feed data to our lstm network
lstm_X_train, lstm_X_test, y_train, y_test = train_test_split(features,df['type'].to_numpy(), test_size=0.3, random_state=42)
train_dataset = TensorDataset(torch.from_numpy(lstm_X_train), torch.from_numpy(y_train).type(torch.LongTensor))
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
val_dataset = TensorDataset(torch.from_numpy(lstm_X_test), torch.from_numpy(y_test).type(torch.LongTensor))
val_loader = DataLoader(val_dataset, batch_size = 32)
"embed_dim" in dir()
if "embed_dim" not in dir():
    embed_dim = 300

n_hidden = 50 # no of hidden dim
n_output = len(le.classes_) # no of class labels
n_hlayer = 1 # no of lstm layers
vocab_size = len(words) # size of vocabulary
model = LSTM(embed_dim, n_hidden, n_output, n_hlayer, vocab_size)
try:
    model.embed.weight.data.copy_(torch.from_numpy(embedding_matrix))
except:
    pass
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001, weight_decay=0.001)
train(model, train_loader, val_loader, 25, optimizer, criterion)