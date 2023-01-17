!pip install pytorch-transformers
import os
from pytorch_transformers import BertTokenizer, BertConfig
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import AdamW, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))
stemmer_en = SnowballStemmer('english')

import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cpu':
    print('cpu')
else:
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))
dir_data = '../input/imdb-dataset-of-50k-movie-reviews'
# dir_models = '../models'
name_file = 'IMDB Dataset.csv'
# os.makedirs(dir_data, exist_ok=True)
# os.makedirs(dir_models, exist_ok=True)
df = pd.read_csv(os.path.join(dir_data, name_file))
df.sample(5)
config = {
    'TextPreprocessor': {
        'del_orig_col': False,
        'mode_stemming': True,
        'mode_norm': True,
        'mode_remove_stops': True,
        'mode_drop_long_words': True,
        'mode_drop_short_words': True,
        'min_len_word': 3,
        'max_len_word': 15,
        'columns_names': 'review'        
    },
}

class TextPreprocessor(object):
    def __init__(self, config):
        """Preparing text features."""
        self._del_orig_col = config.get('del_orig_col', True)
        self._mode_stemming = config.get('mode_stemming', True)
        self._mode_norm = config.get('mode_norm', True)
        self._mode_remove_stops = config.get('mode_remove_stops', True)
        self._mode_drop_long_words = config.get('mode_drop_long_words', True)
        self._mode_drop_short_words = config.get('mode_drop_short_words', True)
        self._min_len_word = config.get('min_len_word', 3)
        self._max_len_word = config.get('max_len_word', 17)
        self._max_size_vocab = config.get('max_size_vocab', 100000)
        self._max_doc_freq = config.get('max_doc_freq', 0.8) 
        self._min_count = config.get('min_count', 5)
        self._pad_word = config.get('pad_word', None)
        self._columns_names = config.get('columns_names', None)

    def _clean_text(self, input_text):
        """Delete special symbols."""
        input_text = input_text.str.lower()
        input_text = input_text.str.replace(r'[^a-z ]+', ' ')
        input_text = input_text.str.replace(r' +', ' ')
        input_text = input_text.str.replace(r'^ ', '')
        input_text = input_text.str.replace(r' $', '')
        return input_text

    def _text_normalization_en(self, input_text):
        '''Normalization of english text'''
        return ' '.join([lemmatizer.lemmatize(item) for item in input_text.split(' ')])

    def _remove_stops_en(self, input_text):
        '''Delete english stop-words'''
        return ' '.join([w for w in input_text.split() if not w in stop_words_en])

    def _stemming_en(self, input_text):
        '''Stemming of english text'''
        return ' '.join([stemmer_en.stem(item) for item in input_text.split(' ')])

    def _drop_long_words(self, input_text):
        """Delete long words"""
        return ' '.join([item for item in input_text.split(' ') if len(item) < self._max_len_word])

    def _drop_short_words(self, input_text):
        """Delete short words"""
        return ' '.join([item for item in input_text.split(' ') if len(item) > self._min_len_word])
    
    def transform(self, df):        
        
        df[self._columns_names] = df[self._columns_names].astype('str')
        df['union_text'] = df[self._columns_names]
            
        if self._del_orig_col:
            df = df.drop(self._columns_names, 1)
    
        df['union_text'] = self._clean_text(df['union_text'])
        
        if self._mode_norm:
            df['union_text'] = df['union_text'].apply(self._text_normalization_en, 1)
            
        if self._mode_remove_stops:
            df['union_text'] = df['union_text'].apply(self._remove_stops_en, 1)
            
        if self._mode_stemming:
            df['union_text'] = df['union_text'].apply(self._stemming_en)
            
        if self._mode_drop_long_words:
            df['union_text'] = df['union_text'].apply(self._drop_long_words, 1)
            
        if self._mode_drop_short_words:
            df['union_text'] = df['union_text'].apply(self._drop_short_words, 1)
            
        df.loc[(df.union_text == ''), ('union_text')] = 'empt'

        return df
df = TextPreprocessor(config['TextPreprocessor']).transform(df)
df.sample(5)
df.sentiment.value_counts()
sentences = ["[CLS] " + sentence[0:500] + " [SEP]" for sentence in df['union_text'].values]
labels = [[1] if i == 'positive' else [0] for i in df['sentiment'].values]
print(sentences[1000], labels[1000])
train_sentences, test_sentences, train_gt, test_gt = train_test_split(
    sentences, 
    labels, 
    test_size=0.3, 
    random_state=123,
)
print(len(train_gt), len(test_gt))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in train_sentences]
print (tokenized_texts[10])
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(
    input_ids,
    maxlen=250,
    dtype="long",
    truncating="post",
    padding="post"
)
attention_masks = [[float(i>0) for i in seq] for seq in input_ids]
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, train_gt, 
    random_state=123,
    test_size=0.1
)

train_masks, validation_masks, _, _ = train_test_split(
    attention_masks,
    input_ids,
    random_state=123,
    test_size=0.1
)
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(
    train_data,
    sampler=RandomSampler(train_data),
    batch_size=32
)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(
    validation_data,
    sampler=SequentialSampler(validation_data),
    batch_size=32
)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
from IPython.display import clear_output

train_loss_set = []
train_loss = 0

# Switch on training mode
model.train()

for step, batch in enumerate(train_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    
    optimizer.zero_grad()
    
    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

    train_loss_set.append(loss[0].item())  

    loss[0].backward()
    
    optimizer.step()
    train_loss += loss[0].item()
    
    clear_output(True)
    plt.plot(train_loss_set)
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()
    
print("Train Loss: {0:.5f}".format(train_loss / len(train_dataloader)))
# torch.save(model, os.path.join(dir_models, 'Bert_epoch_1'))
# Validate
# Switch on evaluation mode
model.eval()

valid_preds, valid_labels = [], []

for batch in validation_dataloader:   
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = logits[0].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    batch_preds = np.argmax(logits, axis=1)
    batch_labels = np.concatenate(label_ids)     
    valid_preds.extend(batch_preds)
    valid_labels.extend(batch_labels)

print("Valid ACC: {0:.2f}%".format(
    accuracy_score(valid_labels, valid_preds) * 100
))
tokenized_texts = [tokenizer.tokenize(sent) for sent in test_sentences]
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

input_ids = pad_sequences(
    input_ids,
    maxlen=250,
    dtype="long",
    truncating="post",
    padding="post"
)
attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(test_gt)

prediction_data = TensorDataset(
    prediction_inputs,
    prediction_masks,
    prediction_labels
)

prediction_dataloader = DataLoader(
    prediction_data, 
    sampler=SequentialSampler(prediction_data),
    batch_size=32
)
model.eval()
test_preds, test_labels = [], []

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = logits[0].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    batch_preds = np.argmax(logits, axis=1)
    batch_labels = np.concatenate(label_ids)  
    test_preds.extend(batch_preds)
    test_labels.extend(batch_labels)
acc_score = accuracy_score(test_labels, test_preds)
print('Test ACC: {0:.2f}%'.format(
    acc_score*100
))