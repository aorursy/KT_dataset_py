# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import re
import json
import time
import datetime
import pandas as pd
import numpy as np
from statistics import mean
from tqdm import tqdm_notebook
from uuid import uuid4

## sklearn
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.model_selection import StratifiedKFold

## Keras Modules
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU, Bidirectional, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.regularizers import l1, l2, l1_l2

## Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

import spacy

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')
# !python -m spacy download en
nlp = spacy.load('en')
!pip install transformers
## PyTorch Transformer
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW, get_linear_schedule_with_warmup
## Check if Cuda is Available
print(torch.cuda.is_available())
RANDOM_STATE = 1234

tqdm_notebook().pandas()
data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")[["text", "target"]]
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
data.head()
def clean_text(text):
    
    # Special characters
    text = re.sub(r"\x89Û_", "", text)
    text = re.sub(r"\x89ÛÒ", "", text)
    text = re.sub(r"\x89ÛÓ", "", text)
    text = re.sub(r"\x89ÛÏWhen", "When", text)
    text = re.sub(r"\x89ÛÏ", "", text)
    text = re.sub(r"China\x89Ûªs", "China's", text)
    text = re.sub(r"let\x89Ûªs", "let's", text)
    text = re.sub(r"\x89Û÷", "", text)
    text = re.sub(r"\x89Ûª", "", text)
    text = re.sub(r"\x89Û\x9d", "", text)
    text = re.sub(r"å_", "", text)
    text = re.sub(r"\x89Û¢", "", text)
    text = re.sub(r"\x89Û¢åÊ", "", text)
    text = re.sub(r"fromåÊwounds", "from wounds", text)
    text = re.sub(r"åÊ", "", text)
    text = re.sub(r"åÈ", "", text)
    text = re.sub(r"JapÌ_n", "Japan", text)    
    text = re.sub(r"Ì©", "e", text)
    text = re.sub(r"å¨", "", text)
    text = re.sub(r"SuruÌ¤", "Suruc", text)
    text = re.sub(r"åÇ", "", text)
    text = re.sub(r"å£3million", "3 million", text)
    text = re.sub(r"åÀ", "", text)
    
    # remove url link
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # remove html tag
    text = re.sub(r'<.*?>', '', text)
    
    # remove numbers
    text = re.sub(r'[\d]+', ' ', text)
    
    return text
# remove controversial tweets
unique_targets = data.groupby('text').agg(unique_target=('target', pd.Series.nunique))
controversial_tweets = unique_targets[unique_targets['unique_target'] > 1].index

data = data[~data['text'].isin(controversial_tweets)]

# remove duplicates rows
data = data.drop_duplicates(subset='text', keep='first')

# remove special characters, url, and html tags
data['text'] = data['text'].apply(clean_text) 
test['text'] = test['text'].apply(clean_text)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
model = RobertaForSequenceClassification.from_pretrained('roberta-base') 
def prepare_features(data_set, labels=None, max_seq_length = 100, 
                     zero_pad = True, include_special_tokens = True): 
    
    ## Tokenzine Input
    input_ids = []
    attention_masks = []
    
    for sent in data_set:
        encoded_dict = tokenizer.encode_plus(
                    sent,                      # Sentence to encode.
                    add_special_tokens = include_special_tokens, # Add '[CLS]' and '[SEP]'
                    max_length = max_seq_length,           # Max length according to our text data.
                    pad_to_max_length = zero_pad, # Pad & truncate all sentences.
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
               )
    
        # Add the encoded sentence to the id list. 

        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).

        attention_masks.append(encoded_dict['attention_mask'])
    
    # convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    if labels is not None: 
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels
    else: 
        return input_ids, attention_masks
train_val = 0.8
train = data.sample(frac=train_val, random_state=RANDOM_STATE)
val = data.drop(train.index).reset_index(drop=True)
train = train.reset_index(drop=True)
train_input_ids, train_attention_masks, train_labels = prepare_features(
    train['text'], train['target'])
val_input_ids, val_attention_masks, val_labels = prepare_features(
    val['text'], val['target'])
test_input_ids, test_attention_masks = prepare_features(
    test['text'])
training_set = TensorDataset(train_input_ids, train_attention_masks, train_labels)
validation_set = TensorDataset(val_input_ids, val_attention_masks, val_labels)
test_set = TensorDataset(test_input_ids, test_attention_masks)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
torch.cuda.is_available()
BATCH_SIZE = 32
LEARNING_RATE = 1e-05
EPSILON = 1e-8
MAX_EPOCHS = 10
loading_params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'drop_last': False,
          'num_workers': 1}


training_loader = DataLoader(training_set, **loading_params)
validation_loader = DataLoader(validation_set, **loading_params)

test_loading_params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'drop_last': False,
          'num_workers': 1}

testing_loader = DataLoader(test_set, **test_loading_params)
#https://www.kaggle.com/datafan07/disaster-tweets-nlp-eda-bert-with-transformers

loss_function = nn.CrossEntropyLoss()

optimizer = AdamW(model.parameters(),
                  lr = LEARNING_RATE, # args.learning_rate
                  eps = EPSILON # args.adam_epsilon
                )

# number of training steps
total_steps = len(training_loader) * MAX_EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
def format_time(elapsed):    
    """A function that takes a time in seconds and returns a string hh:mm:ss"""
    
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
def flat_accuracy(preds, labels):
    """A function for calculating accuracy scores"""
    pred_flat = np.argmax(preds, axis=1)
    labels_flat = labels
    return accuracy_score(labels_flat, pred_flat)
# model = model.train()

for epoch in tqdm_notebook(range(MAX_EPOCHS)):
    # start time for each epoch
    t0 = time.time()
    
    total_train_loss = 0
    
    model.train()
    
    print("EPOCH -- {} / {}".format(epoch, MAX_EPOCHS))
    for step, batch in enumerate(training_loader):
        if step % 30 == 0 and not step == 0: 
            elapsed = format_time(time.time() - t0)
            print(' Batch {} of {}. Elapsed: {:}'.format(step, len(training_loader), elapsed))
            
        input_ids = batch[0].to(device).to(torch.int64)
        input_masks = batch[1].to(device).to(torch.int64)
        labels = batch[2].to(device).to(torch.int64)          
                  
        # Always clear any previously calculated gradients before performing a backward pass. PyTorch doesn't do this automatically because accumulating the gradients is 'convenient while training RNNs'. 
        model.zero_grad()
                  
#         optimizer.zero_grad()
#         sent = sent.squeeze(0)
#         if torch.cuda.is_available():
#             sent = sent.cuda()
#             label = label.cuda()
                  
        loss, logits = model(input_ids, 
                           token_type_ids=None,
                           attention_mask=input_masks, 
                           labels=labels)
                  
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        
        total_train_loss += loss.item()
        loss.backward()
                  
        # Clip the norm of the gradients to 1.0. This is to help 
        # prevent the 'exploding gradients' problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # update parameters and move a step forward using the computed gradients          
        optimizer.step()
        scheduler.step()
        
        #         output = model.forward(sent)[0]
        #         _, predicted = torch.max(output, 1)

        #         loss = loss_function(output, label)
        #         loss.backward()
        #         optimizer.step()
    avg_train_loss = total_train_loss / len(training_loader)
    training_time = format_time(time.time() - t0)
            
    print('')
    print(' Average training loss: {0:.4f}'.format(avg_train_loss))
    print(' Training epoch took: {:}'.format(training_time))
    
    print('Running Validation')
                  
    model.eval()
        
    val_predictions = []
    val_labels = []
    for batch in validation_loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            loss, logits = model(b_input_ids, 
                       token_type_ids=None, 
                       attention_mask=b_input_mask,
                       labels=b_labels)
            
        val_predictions.append(logits.detach().cpu().numpy())
        val_labels.append(b_labels.to('cpu').numpy())

    val_predictions = [item for sublist in val_predictions for item in sublist]
    val_labels = [item for sublist in val_labels for item in sublist]
        
    val_accuracy = flat_accuracy(val_predictions, val_labels)
    print('  Accuracy: {0:.4f}'.format(val_accuracy))
model.eval()
predictions = []
for batch in testing_loader:
    batch = tuple(t.to(device) for t in batch)
    
    input_ids, input_masks = batch
    
    with torch.no_grad():
        logits = model(input_ids, 
                        token_type_ids=None,
                        attention_mask=input_masks)[0]
        
        logits = logits.detach().cpu().numpy()
        
        predictions.append(logits)
flat_predictions = [item for sublist in predictions for item in sublist]
targets = np.argmax(flat_predictions, axis=1).flatten() 
test['target'] = targets
tokenizer.convert_tokens_to_ids(tokenizer.tokenize('dick'))
submission = test[['id', 'target']].to_csv("submission_roberta.csv", index=False)
# function to preprocess the tweets.
def preprocess(texts, allowed_postags=['NOUN', "ADJ", "VERB", "ADV", "DET"]): 
    texts_out = []
    for text in texts: 
        # lower case the sentences
        lowered_text = text.lower()
    
        # parse the text with nlp() from spacy. It treats emoticons as words.
        doc = nlp(lowered_text)

        # remove space, numericals, and punctuation.
        tokens = [token for token in doc if not (token.is_punct | 
                                               token.is_space |  
                                               token.is_digit)]

        # filter only words is alpha
        tokens = [token for token in tokens if token.is_alpha]
        
        # lemmatization, filter words by pos tag.
        lemmas = [token.lemma_ for token in tokens if token.pos_ in allowed_postags]
      
        # remove stop words
        words = [lemma for lemma in lemmas if lemma not in stop_words]

        texts_out.append(words)
    return texts_out
data_pruned = data.copy(deep=True)
data_pruned['text'] = data_pruned['text'].progress_apply(clean_text)
data_pruned['text'] = data_pruned['text'].progress_apply(preprocess)
test_pruned = test.copy(deep=True)
test_pruned['text'] = test_pruned['text'].apply(clean_text)
test_pruned['text'] = test_pruned['text'].apply(preprocess)
all_text = pd.concat([data_pruned[['text']], test_pruned[['text']]], ignore_index=True)
all_text
cv = CountVectorizer(ngram_range=(1,2))
tfidf_transformer = TfidfTransformer()
x = cv.fit_transform(all_text['text'])
x_all_tfidf = tfidf_transformer.fit_transform(x)
data_pruned.shape
training_samples = data_pruned.shape[0]
X_train = x_all_tfidf[:training_samples,:]
X_test = x_all_tfidf[training_samples:,:]
y_train = data_pruned['target']
mb_classifier = MB().fit(X_train, y_train)
pred = mb_classifier.predict(X_train)

c = classification_report(y_train,pred)
skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE)
total_accuracy = []
total_precision = []
total_recall = []
for train_index, val_index in skf.split(X_train, y_train):
    current_X_train = X_train[train_index]
    current_y_train = y_train.iloc[train_index]
    current_X_val = X_train[val_index]
    current_y_val = y_train.iloc[val_index]
    
    clf = clf_svm
    clf.fit(current_X_train, current_y_train)
    
    current_predictions = clf.predict(current_X_val)
    total_accuracy.append(accuracy_score(current_y_val, current_predictions))
    total_precision.append(precision_score(current_y_val, current_predictions))
    total_recall.append(recall_score(current_y_val, current_predictions))
    
ave_accuracy = mean(total_accuracy)
ave_precision = mean(total_precision)
ave_recall = mean(total_recall)
print("Average Accuracy: {:.4f}".format(ave_accuracy))
print("Average Precision: {:.4f}".format(ave_precision))
print("Average Recall: {:.4f}".format(ave_recall))

y_train.shape
test_pruned['target'] = mb_classifier.predict(x_test)
test_pruned[['id', 'target']].to_csv("submission_2.csv", index=False)
clf_svm = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
clf_svm.fit(x_train, y_train)
pred = clf_svm.predict(x_train)
print(classification_report(y_train,pred))
test_pruned['target'] = clf_svm.predict(x_test)
test_pruned[['id', 'target']].to_csv("submission_svm.csv", index=False)
TRAIN_VAL_SPLIT = 0.8

train = data.sample(frac=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
val = data[~data.index.isin(train.index)]

X_train = train['text'].values
y_train = train['target'].values

X_val = val['text'].values
y_val = val['target'].values

X_test = test['text'].values
# function to preprocess the tweets.
def preprocess(texts, allowed_postags=['NOUN', "ADJ", "VERB", "ADV", "PROPN", "DET"]): 
    texts_out = []
    for text in texts: 
        # lower case the sentences
        lowered_text = text.lower()
        
        # remove url link
        lowered_text = re.sub(r'https?://\S+|www\.\S+', '', lowered_text)

        # remove html tags
        lowered_text = re.sub(r'<.*?>', '', lowered_text)
    
        # parse the text with nlp() from spacy. It treats emoticons as words.
        doc = nlp(lowered_text)

        # remove space, numericals, and punctuation.
        tokens = [token for token in doc if not (token.is_punct | 
                                               token.is_space |  
                                               token.is_digit)]

        # lemmatization, filter words by pos tag.
        lemmas = [token.lemma_ for token in tokens if token.pos_ in allowed_postags]
      
        # remove stop words
        words = [lemma for lemma in lemmas if lemma not in stop_words]

        texts_out.append(words)
    return texts_out
train_corpus = preprocess(X_train)
val_corpus = preprocess(X_val)
test_corpus = preprocess(X_test)
corpus = train_corpus + val_corpus + test_corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

train_sequences = tokenizer.texts_to_sequences(train_corpus)
val_sequences = tokenizer.texts_to_sequences(val_corpus)
test_sequences = tokenizer.texts_to_sequences(test_corpus)

# the dictionary of word occurrences.
word_index = tokenizer.word_index

train_max_length = max([len(x) for x in train_sequences])
val_max_length = max([len(x) for x in val_sequences])
test_max_length = max([len(x) for x in test_sequences])

max_length = max(train_max_length, val_max_length, test_max_length)

X_train_pad = pad_sequences(train_sequences, maxlen=max_length, padding="post")
X_val_pad = pad_sequences(val_sequences, maxlen=max_length, padding="post")
X_test_pad = pad_sequences(test_sequences, maxlen=max_length, padding="post")

vocab = np.array(list(tokenizer.word_index.keys()))
vocab_size = len(tokenizer.word_index) + 1
EMBEDDING_DIM = 30
model_tuned = Sequential()
model_tuned.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model_tuned.add(GRU(units=30, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model_tuned.add(GRU(units=30, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model_tuned.add(GRU(units=30, dropout=0.2, recurrent_dropout=0.2, 
                               kernel_regularizer=l1_l2(0.01, 0.01), recurrent_regularizer=l1_l2(0.01, 0.01)))
model_tuned.add(Dense(1, activation='sigmoid'))

model_tuned.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model_tuned.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_val_pad, y_val), verbose=2)
y_pred_tuned = model_tuned.predict(X_test_pad)
y_pred_binary = list(map(lambda x: 1 if x >= 0.5 else 0, y_pred_tuned))
y_pred_binary
test['target'] = y_pred_binary
test[['id', 'target']].to_csv("submission_rnn.csv", index=False)
def ConvNet(max_sequence_length, num_words, embedding_dim, labels_index):
 
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            input_length=max_sequence_length)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [3,4,5,6]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=50, 
                        kernel_size=filter_size, 
                        activation='relu')(embedded_sequences)
        l_conv = Dropout(0.2)(l_conv)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)
    l_merge = concatenate(convs, axis=1)
#     x = Dropout(0.2)(l_merge)  
    x = Dense(32, activation='relu')(l_merge)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model
model = ConvNet(max_length, vocab_size, EMBEDDING_DIM, 1)
y_val
y_train
hist = model.fit(X_train_pad, 
                 y_train, 
                 epochs=25, 
                 batch_size=128, 
                 validation_data=(X_val_pad, y_val), 
                 verbose=2) 
y_pred = model.predict(X_test_pad)
y_pred_binary = list(map(lambda x: 1 if x >= 0.5 else 0, y_pred))
test['target'] = y_pred_binary
test[['id', 'target']].to_csv("submission_cnn.csv", index=False)
data[data['target'] == 0].values