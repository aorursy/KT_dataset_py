import os

import re

import random



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchtext



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import spacy

from collections import namedtuple

from pprint import pprint





from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import ParameterGrid

from tqdm.notebook import tqdm

from torchtext import datasets

from torchtext import data



# Create a Tokenizer with the default settings for English

# including punctuation rules and exceptions

spacy_en = spacy.load('en')

%matplotlib inline
SEED = 1234



torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

np.random.seed(SEED)

torch.backends.cudnn.deterministic = True

random.seed(SEED)
def decontracted(text):

    # specific

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"can\'t", "can not", text)



    # general

    text = re.sub(r"n\'t", " not", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'s", " is", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'t", " not", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'m", " am", text)

    return text
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")



stopwords = set(stopwords.words("english")) | set(["br"])
from string import punctuation



def remove_punctuation(text):

    text = text.translate(str.maketrans('', '', punctuation))

    return text
def tokenizer(text):

    """Tokenize and do an Early PreProcessing"""

    text = decontracted(text)

    text = remove_punctuation(text)

    return [word.text.lower() for word in spacy_en.tokenizer(text) if word.text.lower() not in stopwords]

TEXT = data.Field(tokenize=tokenizer, include_lengths=True)

LABEL = data.LabelField(dtype = torch.float)
%%time

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print(len(train_data), len(test_data))

print(vars(train_data.examples[0]))
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))
print(f'Train Data: \t\t {len(train_data):,}')

print(f'Validation Dataset: \t {len(valid_data):,}')

print(f'Test Data: \t\t {len(test_data):,}')
type(train_data.examples[0].text)
def get_dataframe_from_dataset(dataset, labels = {'X': 'text', 'y': 'label'}):

    """Utility Method to convert torchext.data.Dataset to numpy array of text and label"""

    i = 0

    data = {'X' : [], 'y' : []}

    for example in tqdm(dataset):

        data['X'].append(' '.join(example.text))

        data['y'].append(example.label)

    

    assert len(data['X']) == len(data['y'])



    return pd.DataFrame(data).rename(columns=labels)

train_df = get_dataframe_from_dataset(train_data)

val_df = get_dataframe_from_dataset(valid_data)

test_df = get_dataframe_from_dataset(test_data)
test_df.head()
dataframes = [train_df, val_df, test_df]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for df in tqdm(dataframes):

    df['label'] = le.fit_transform(df['label'])
train_df.head()
split_df = lambda df: (np.array(df['text']), np.array(df['label']))
X_train, y_train = split_df(train_df)

X_valid, y_valid = split_df(val_df)

X_test, y_test = split_df(test_df)
def get_predictions_and_accuracy(X_test):

    y_pred = best_model.pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    print('Test Accuracy: {:.4f}%'.format(test_accuracy * 100))

    report = pd.DataFrame(classification_report(y_pred, y_test, target_names=['neg', 'pos'], output_dict=True))

    print(report)

    sns.heatmap(report.iloc[:-1:].T, annot=True)

    plt.title('Classification Report')

    plt.show()
from sklearn.pipeline import Pipeline
Model = namedtuple('Model', ['pipeline', 'predictions', 'accuracy'])
from sklearn.naive_bayes import MultinomialNB
naive_bais1 = Pipeline(

    [('cv', CountVectorizer()),

      ('nb', MultinomialNB())])



naive_bais2 = Pipeline(

    [('tfidf', TfidfVectorizer()),

      ('nb', MultinomialNB())])
best_accuracy = 0.0

best_model = Model(None, None, None)

for pipeline in tqdm([naive_bais1, naive_bais2]):

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = Model(pipeline, y_pred, accuracy)



print('Best Pipeline found: {} with accuracy: {:.2f}%'.format([step[0] for step in best_model.pipeline.steps], best_model.accuracy*100))
get_predictions_and_accuracy(X_test)
parameters = {

    'ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)]

}

parameters = ParameterGrid(parameters)
best_model = Model(None, None, None)

best_accuracy = 0

for parameter in tqdm(list(parameters)):

    pipeline = Pipeline(

        [('tfidf', TfidfVectorizer(ngram_range=parameter['ngram_range'])),

        ('nb', MultinomialNB())])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    print('Accuracy: {:.2f} Parameter: {}'.format(accuracy * 100, parameter['ngram_range']))

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = Model(pipeline, y_pred, accuracy)



print('Best Pipeline found: {} with accuracy: {:.2f}%'.format([step[1] for step in best_model.pipeline.steps], best_model.accuracy*100))
get_predictions_and_accuracy(X_test)
from sklearn.ensemble import RandomForestClassifier
tf = TfidfVectorizer()

tf_x_train = tf.fit_transform(X_train)

accuracy = accuracy_score(RandomForestClassifier().fit(tf_x_train, y_train).predict(tf.transform(X_valid)), y_valid)

print('Base Accuracy: {:.4f}'.format(accuracy * 100))
parameters = {

    'ngram_range': [(1, 2)],

    'max_depth': [None],

    'n_estimators': [300, 1000],

    'n_jobs' : [-1]

}



parameters = ParameterGrid(parameters)
best_model = Model(None, None, None)

best_accuracy = 0

for parameter in tqdm(list(parameters)):

    pipeline = Pipeline(

        [('tfidf', TfidfVectorizer(ngram_range=parameter['ngram_range'])),

        ('rfc', RandomForestClassifier(

                    max_depth=parameter['max_depth'],

                    n_estimators=parameter['n_estimators']

                                      ))])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    print('Accuracy: {:.2f} Parameters: {}'.format(accuracy * 100, parameter))

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = Model(pipeline, y_pred, accuracy)



print('Best Pipeline found: {} with accuracy: {:.2f}%'.format([step[1] for step in best_model.pipeline.steps], best_model.accuracy*100))
get_predictions_and_accuracy(X_test)
from sklearn.svm import LinearSVC
parameters = {

    'ngram_range': [(1, 2)],

    'penalty': ['l2'],

    'C' : [1.0, 2.0, 3.0]

}



parameters = ParameterGrid(parameters)
best_model = Model(None, None, None)

best_accuracy = 0

for parameter in tqdm(list(parameters)):

    pipeline = Pipeline(

        [('tfidf', TfidfVectorizer(ngram_range=parameter['ngram_range'])),

        ('svc', LinearSVC(penalty=parameter['penalty'],

                          C=parameter['C']))])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    print('Accuracy: {:.2f} Parameters: {}'.format(accuracy * 100, parameter))

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = Model(pipeline, y_pred, accuracy)



print('Best Pipeline found: {} with accuracy: {:.2f}%'.format([step[1] for step in best_model.pipeline.steps], best_model.accuracy*100))
get_predictions_and_accuracy(X_test)
from sklearn.linear_model import SGDClassifier
parameters = {

    'ngram_range': [(1,1), (1, 2)],

    'penalty': ['l2', 'elasticnet'],

    'alpha': [1e-3, 1e-4],

    'n_jobs' : [-1]

}



parameters = ParameterGrid(parameters)
best_model = Model(None, None, None)

best_accuracy = 0

for parameter in tqdm(list(parameters)):

    pipeline = Pipeline(

        [('tfidf', TfidfVectorizer(ngram_range=parameter['ngram_range'])),

        ('sgd', SGDClassifier(penalty=parameter['penalty'],

                          alpha=parameter['alpha'],

                          n_jobs=parameter['n_jobs']))])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    print('Accuracy: {:.2f} Parameters: {}'.format(accuracy * 100, parameter))

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = Model(pipeline, y_pred, accuracy)



print('Best Pipeline found: {} with accuracy: {:.2f}%'.format([step[1] for step in best_model.pipeline.steps], best_model.accuracy*100))
get_predictions_and_accuracy(X_test)
import xgboost as xgb
parameters = {

    'ngram_range': [(1, 2)],

    'max_depth' : [3, 4, 5],

    'n_estimators' : [100, 150]

}



parameters = ParameterGrid(parameters)
best_model = Model(None, None, None)

best_accuracy = 0

for parameter in tqdm(list(parameters)):

    pipeline = Pipeline(

        [('tfidf', TfidfVectorizer(ngram_range=parameter['ngram_range'])),

        ('xgb', xgb.XGBClassifier(objective="binary:logistic",

            max_depth=parameter['max_depth'],

            n_estimators=parameter['n_estimators']

        ))])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    print('Accuracy: {:.2f} Parameters: {}'.format(accuracy * 100, parameter))

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = Model(pipeline, y_pred, accuracy)



print('Best Pipeline found: {} with accuracy: {:.2f}%'.format([step[1] for step in best_model.pipeline.steps], best_model.accuracy*100))
get_predictions_and_accuracy(X_test)
import lightgbm as lgbm
parameters = {

    'ngram_range': [(1, 2)],

    "objective": ["binary"],

    "metric": ["binary_logloss"],

    "max_depth" : [-1, 7],

    "num_leaves": [31, 70],

    "verbose": [-1],

    "n_jobs": [-1]

    }

parameters = ParameterGrid(parameters)
best_model = Model(None, None, None)

best_accuracy = 0

for parameter in tqdm(list(parameters)):

    pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=parameter['ngram_range'])),

        ('lgbm', lgbm.LGBMClassifier(

            objective=parameter['objective'],

            metric=parameter['metric'],

            max_depth=parameter['max_depth'],

            num_leaves=parameter['num_leaves'],

            verbose=parameter['verbose']

        ))])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    print('Accuracy: {:.2f} Parameters: {}'.format(accuracy * 100, parameter))

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = Model(pipeline, y_pred, accuracy)



print('Best Pipeline found: {} with accuracy: {:.2f}%'.format([step[1] for step in best_model.pipeline.steps], best_model.accuracy*100))
get_predictions_and_accuracy(X_test)
from sklearn.ensemble import VotingClassifier
voting_classifier = VotingClassifier(estimators=[

                        ('nb' , MultinomialNB()),

                        ('rfc', RandomForestClassifier()),

                        ('svc', LinearSVC(penalty='l2', C=0.2)),

                        ('sgd', SGDClassifier(penalty='l2', alpha=0.0001))

                    ])
pipeline = Pipeline([

    ('tfid', TfidfVectorizer()),

    ('vc', voting_classifier)

])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)

print('Accuracy: {:.2f}'.format(accuracy * 100))

best_model = Model(pipeline, y_pred, accuracy)
get_predictions_and_accuracy(X_test)
# from torchtext.data import Pipeline

# def remove_stopwords2(text):

#     """Removes Stopwords from the text uses scapy tokenizer little heavy but good"""

#     text = ' '.join([word.text for word in tokenizer(text.text) if word.text not in stopwords])

#     return text

# pipeline = Pipeline(remove_stopwords2)

# TEXT_test = data.Field(preprocessing=pipeline, tokenize=tokenizer)

# LABEL_test = data.LabelField(dtype = torch.float)

# train_data2, test_data2 = datasets.IMDB.splits(TEXT, LABEL)

# train_data.examples[0].text

# [word.text for word in tokenizer(train_data2.examples[0].text.text) if word.text not in stopwords]

# type(tokenizer(train_data2.examples[0].text))

# TEXT_test.preprocess(train_data2.examples[0].text.text)

# Try something like this

# https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_VOCAB = 25000

TEXT.build_vocab(train_data,

                 max_size=MAX_VOCAB,

                 vectors='glove.6B.100d',

                 unk_init=torch.Tensor.normal_)



LABEL.build_vocab(train_data)
len(TEXT.vocab)
BATCH_SIZE = 128



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(

    (train_data, valid_data, test_data),

    batch_size=BATCH_SIZE,

    sort_within_batch=True,

    device=device

)
class RNN(nn.Module):



    def __init__(self,vocab_size, embedding_dim, padding_idx, 

                 hidden_dim, n_layer, dropout, output_dim):

        super(RNN, self).__init__()



        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layer, bidirectional=True, dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)



    def forward(self, text, text_length):

        embedded = self.embedding(text)

        padded_token = nn.utils.rnn.pack_padded_sequence(embedded, text_length)

        packed_output, (hidden, cell) = self.rnn(padded_token)



        # We can unpack to see just for debug

        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)



        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)



        hidden = self.dropout(hidden)



        return self.fc1(hidden) 
INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

PADDING_IDX = TEXT.vocab.stoi[TEXT.pad_token]

HIDDEN_DIM = 256

NUM_LAYERS = 2

DROPOUT = 0.7

OUTPUT_DIM = 1



model = RNN(INPUT_DIM, EMBEDDING_DIM, PADDING_IDX, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM)
# Number of Parameters

print('Number of Trainable Parameters: {:,}'.format(sum([p.numel() for p in model.parameters() if p.requires_grad])))
# Copy Glove Embeddings

model.embedding.weight.data.copy_(TEXT.vocab.vectors)
# Set Zero

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PADDING_IDX] = torch.zeros(EMBEDDING_DIM)
def calc_accuracy(prediction, true_output):

    """Calculates accuracy"""

    correct = (torch.round(torch.sigmoid(prediction)) == true_output).float()

    return correct.sum() / len(correct)
def train(model, iterator, optimizer, criterion):

    model.train()

    epoch_loss = 0

    epoch_acc = 0



    for batch in iterator:

        optimizer.zero_grad()

        

        text, text_length = batch.text

        prediction = model(text, text_length).squeeze(1)

        

        loss = criterion(prediction, batch.label)

        acc = calc_accuracy(prediction, batch.label)



        loss.backward()

        optimizer.step()



        epoch_loss += loss

        epoch_acc += acc

    

    return epoch_loss/len(iterator) , epoch_acc / len(iterator)
def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    epoch_acc = 0

    with torch.no_grad():

        for batch in iterator:

            text, text_length = batch.text

            prediction  = model(text, text_length).squeeze(1)

            loss = criterion(prediction, batch.label)

            acc = calc_accuracy(prediction, batch.label)



            epoch_loss += loss

            epoch_acc += acc



    return epoch_loss/len(iterator), epoch_acc/len(iterator)



optimizer = optim.Adam(model.parameters(), weight_decay=0.001)

criterion = nn.BCEWithLogitsLoss().to(device)

model = model.to(device)
EPOCHS = 10
for i in tqdm(range(EPOCHS)):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion )

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)



    print("Epoch: [{:02} / {:02}]  \tTraining Loss: {:.4f} \t\tTraining Accuracy: {:.4f}".format(i+1, EPOCHS, train_loss, train_acc))

    print("Epoch: [{:02} / {:02}] \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(i+1, EPOCHS, valid_loss, valid_acc))
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print('Test Loss: {:.4f} \tTest Accuracy: {:.2f}%'.format(valid_loss, valid_acc*100))
model = RNN(INPUT_DIM, EMBEDDING_DIM, PADDING_IDX, HIDDEN_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM)

optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)

criterion = nn.BCEWithLogitsLoss().to(device)

model = model.to(device)



for i in tqdm(range(EPOCHS)):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion )

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)



    print("Epoch: [{:02} / {:02}]  \tTraining Loss: {:.4f} \t\tTraining Accuracy: {:.4f}".format(i+1, EPOCHS, train_loss, train_acc))

    print("Epoch: [{:02} / {:02}] \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(i+1, EPOCHS, valid_loss, valid_acc))
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print('Test Loss: {:.4f} \tTest Accuracy: {:.2f}%'.format(test_loss, test_acc*100))
def generate_n_grams(sentence, n=2):

    grams =  set(zip(*[x[i:] for i in range(n)]))

    for gram in grams:

        sentence.append(' '.join(gram))

    return sentence
# Testing Method

x = ['hello', 'world', 'how', 'are', 'you']

generate_n_grams(x, 2)
MAX_VOCAB = 25000



TEXT_FT = data.Field(tokenize=tokenizer, preprocessing=generate_n_grams)

LABEL_FT = data.LabelField(dtype=torch.float)



train_data_FT, test_data_FT = datasets.IMDB.splits(TEXT_FT, LABEL_FT)

train_data_FT, valid_data_FT = train_data_FT.split(random_state=random.seed(SEED))





TEXT_FT.build_vocab(train_data_FT, 

                    max_size=MAX_VOCAB,

                    vectors='glove.6B.100d',

                    unk_init=torch.Tensor.normal_)

LABEL_FT.build_vocab(train_data_FT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



BATCH_SIZE = 128

train_iter_FT, valid_iter_FT, test_iter_FT = data.BucketIterator.splits(

    (train_data_FT, valid_data_FT, test_data_FT),

    batch_size=BATCH_SIZE,

    device=device

)
class FastText(nn.Module):



    def __init__(self, vocab_size, embeddding_dim, padding_idx, output_dim):

        super(FastText, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embeddding_dim, padding_idx)

        self.fc = nn.Linear(embeddding_dim, output_dim)

    

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(1, 0, 2)

        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze(1)

        return self.fc(x)
INPUT_DIM_FT = len(TEXT_FT.vocab)

EMBEDDING_DIM_FT = 100

PADDING_IDX_FT = TEXT_FT.vocab.stoi[TEXT_FT.pad_token]

OUTPUT_DIM_FT = 1

model_ft = FastText(INPUT_DIM_FT, EMBEDDING_DIM_FT, PADDING_IDX_FT, OUTPUT_DIM_FT)
print('Number of Parameters: {:,}'.format(sum([p.numel() for p in model_ft.parameters() if p.requires_grad])))
model_ft.embedding.weight.data.copy_(TEXT_FT.vocab.vectors)
# Unknown word embeddings zero

model_ft.embedding.weight.data[TEXT_FT.vocab.stoi[TEXT_FT.unk_token]] = torch.zeros(EMBEDDING_DIM_FT)

model_ft.embedding.weight.data[PADDING_IDX_FT] = torch.zeros(EMBEDDING_DIM_FT)
def train(model, iterator, optimizer, criterion):

    model.train()

    epoch_loss = 0

    epoch_acc = 0



    for batch in iterator:

        optimizer.zero_grad()

        

        text = batch.text

        prediction = model(text).squeeze(1)

        

        loss = criterion(prediction, batch.label)

        acc = calc_accuracy(prediction, batch.label)



        loss.backward()

        optimizer.step()



        epoch_loss += loss

        epoch_acc += acc

    

    return epoch_loss/len(iterator) , epoch_acc / len(iterator)
def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    epoch_acc = 0

    with torch.no_grad():

        for batch in iterator:

            text = batch.text

            prediction  = model(text).squeeze(1)

            loss = criterion(prediction, batch.label)

            acc = calc_accuracy(prediction, batch.label)



            epoch_loss += loss

            epoch_acc += acc



    return epoch_loss/len(iterator), epoch_acc/len(iterator)



optimizer = optim.Adam(model_ft.parameters())
criterion = nn.BCEWithLogitsLoss()
model_ft = model_ft.to(device)

criterion = criterion.to(device)
EPOCHS = 10



for i in range(EPOCHS):

    train_loss, train_acc = train(model_ft, train_iter_FT, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model_ft, valid_iter_FT, criterion)





    print("Epoch: [{:02} / {:02}]  \tTraining Loss: {:.4f} \t\tTraining Accuracy: {:.4f}".format(i+1, EPOCHS, train_loss, train_acc))

    print("Epoch: [{:02} / {:02}] \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(i+1, EPOCHS, valid_loss, valid_acc))
test_loss, test_acc = evaluate(model_ft, test_iter_FT, criterion)

print('Test Loss: {:.4f} \tTest Accuracy: {:.2f}%'.format(test_loss, test_acc*100))