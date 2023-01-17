# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
import nltk
import re
import string
from keras.preprocessing.text import one_hot
from keras.layers import LSTM,Embedding,Dense,Bidirectional,GlobalMaxPool2D,BatchNormalization,Dropout,TimeDistributed,GlobalMaxPool1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import SpatialDropout1D,GRU
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
train= pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train.head()
ntrain= train.shape[0]
ntest= test.shape[0]
ntrain,ntest
label= train['target']
train.drop(['target'],axis=1,inplace=True)
data= pd.concat([train,test])
data.shape,train.shape,test.shape
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
# Applying the cleaning function to both test and training datasets
data['text'] = data['text'].apply(lambda x: clean_text(x))
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
data['text']=data['text'].apply(lambda x: remove_emoji(x))
contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def expand_contractions(s, contractions = contractions):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, s)

expand_contractions("can't stop won't stop")
data['text'] =data['text'].apply(expand_contractions)
data['text'].head()
test= data[ntrain:]
train= data[:ntrain]
train.shape,test.shape
tweets= train['text'].copy()
tweets_test= test['text'].copy()

from keras.preprocessing.text import Tokenizer
t = Tokenizer()
t.fit_on_texts(tweets)
vocab_size_train = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(tweets)
# pad documents to a max length of 4 words
max_length = 50
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
tweets= padded_docs
from keras.preprocessing.text import Tokenizer
t = Tokenizer()
t.fit_on_texts(tweets_test)
vocab_size_tesr = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(tweets_test)
# pad documents to a max length of 4 words
max_length = 50
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

tweets_test= padded_docs
# load the whole embedding into memory
embeddings_index = dict()
f = open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size_train, 
                             200))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
e = Embedding(vocab_size_train, 200, weights=[embedding_matrix], input_length=100, trainable=False)
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val= train_test_split(tweets,label)
opt = Adam(lr=0.001, decay=1e-6)
model=Sequential()
model.add(e)
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
model.add(BatchNormalization())
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
history= model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=20,batch_size=64,verbose=1)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);
y_pred_lstm= model.predict_classes(X_val)
from sklearn.metrics import accuracy_score
accuracy_lstm= accuracy_score(y_pred_lstm,y_val)
print("The accuracy for the Lstm model is {} %".format(accuracy_lstm*100))
model=Sequential()
model.add(e)
model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))
model.add(BatchNormalization())
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
hist=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=20,batch_size=64,verbose=1)
y_pred_bilstm= model.predict_classes(X_val)
accuracy_bilstm= accuracy_score(y_pred_bilstm,y_val)
print("The accuracy for the Bidirectional Lstm model is {} %".format(accuracy_bilstm*100))
plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);
model=Sequential()
model.add(e)
model.add(SpatialDropout1D(0.3))
model.add(GRU(100))
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer='adam' ,loss='binary_crossentropy', metrics=['accuracy'])
hist=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=20,batch_size=64,verbose=1)
y_pred_gru= model.predict_classes(X_val)
accuracy_gru= accuracy_score(y_pred_gru,y_val)
print("The accuracy for the GRU model is {} %".format(accuracy_gru*100))
plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);
from keras.layers import Layer
import keras.backend as K
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()
inputs=keras.Input(shape=(max_length,))
x=(e)(inputs)
att_in=LSTM(100,return_sequences=True,dropout=0.3,recurrent_dropout=0.2)(x)
att_out=attention()(att_in)
outputs=Dense(1,activation='sigmoid',trainable=True)(att_out)
modelA=Model(inputs,outputs)
modelA.summary()
modelA.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist=modelA.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=20,batch_size=64,verbose=1)
y_pred_attention= modelA.predict(X_val)

for i in range(len(y_pred_attention)):
    if (y_pred_attention[i]>=0.5):
        y_pred_attention[i]=1
    else:
        y_pred_attention[i]=0
from sklearn.metrics import accuracy_score
accuracy_attention= accuracy_score(y_pred_attention,y_val)
print("The accuracy for the attention model is {} %".format(accuracy_attention*100))
plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);
accuracy_score= {'accuracy_lstm':accuracy_lstm,'accuracy_bilstm':accuracy_bilstm,'accuracy_gru':accuracy_gru,'accuracy_attention':accuracy_attention}
accuracy_score
import torch
import torch.nn as nn
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
PRE_TRAINED_MODEL_NAME = '../input/bert-base-uncased'
sample_txt = 'These are tough times we must stand together'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')
encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
)
encoding.keys()
token_lens = []
for txt in data.text:
    
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count')
MAX_LEN=50
class DisasterTweet(Dataset):
    
    def __init__(self, tweets, label, tokenizer, max_len):
        
        
        self.tweets = tweets
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.tweets)
    def __getitem__(self, item):
        
        tweets = str(self.tweets[item])
        label = self.label[item]
        encoding = self.tokenizer.encode_plus(
        tweets,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')
        return {
        'tweet_text': tweets,
         'input_ids': encoding['input_ids'].flatten(),
         'attention_mask': encoding['attention_mask'].flatten(),
         'labels': torch.tensor(label, dtype=torch.long)
          }    
train= list(zip(train['text'],label))
df = pd.DataFrame(train, columns = ['tweets', 'label'])
df.head()
from sklearn.model_selection import train_test_split
train, val = train_test_split(
  df,
  test_size=0.1,
  random_state=RANDOM_SEED
)
def create_data_loader(data, tokenizer, max_len, batch_size):
    
    ds = DisasterTweet(tweets=data.tweets.to_numpy(),
    label=data.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4)
BATCH_SIZE = 32
train_data_loader = create_data_loader(train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val, tokenizer, MAX_LEN, BATCH_SIZE)
train.shape,val.shape
df = next(iter(train_data_loader))
df.keys()
print(df['input_ids'].shape)
print(df['attention_mask'].shape)
print(df['labels'].shape)

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
class FakeNewsClassifier(nn.Module):
    
    def __init__(self, n_classes):
        
        super(FakeNewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        
        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
        output = self.drop(pooled_output)
        return self.out(output)
n_classes= 2
#setting device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FakeNewsClassifier(n_classes)
model = model.to(device)
input_ids = df['input_ids'].to(device)
attention_mask = df['attention_mask'].to(device)
import torch.nn.functional as F
F.softmax(model(input_ids, attention_mask),dim=1)
model.parameters
EPOCHS = 20
optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)
def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler, n_examples):  
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)
def eval_model(model, data_loader, loss_fn, device, n_examples):
    
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
        return correct_predictions.double() / n_examples, np.mean(losses)
from collections import defaultdict
%%time
history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(model,train_data_loader,loss_fn,optimizer,device,scheduler,len(train))
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(model,val_data_loader,loss_fn,device,len(val))
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc
plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);
best_accuracy
