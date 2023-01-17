# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

import re

import os

import sys

import time

import pickle

import random

import unidecode

from tqdm import tqdm

tqdm.pandas()

from scipy.stats import spearmanr

from gensim.models import Word2Vec

from flashtext import KeywordProcessor

from keras.preprocessing import text, sequence



import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, KFold

train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
sub = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
MODEL = pickle.load(open('/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', 'rb'))
PUNCTS = {

            '》', '〞', '¢', '‹', '╦', '║', '♪', 'Ø', '╩', '\\', '★', '＋', 'ï', '<', '?', '％', '+', '„', 'α', '*', '〰', '｟', '¹', '●', '〗', ']', '▾', '■', '〙', '↓', '´', '【', 'ᴵ',

            '"', '）', '｀', '│', '¤', '²', '‡', '¿', '–', '」', '╔', '〾', '%', '¾', '←', '〔', '＿', '’', '-', ':', '‧', '｛', 'β', '（', '─', 'à', 'â', '､', '•', '；', '☆', '／', 'π',

            'é', '╗', '＾', '▪', ',', '►', '/', '〚', '¶', '♦', '™', '}', '″', '＂', '『', '▬', '±', '«', '“', '÷', '×', '^', '!', '╣', '▲', '・', '░', '′', '〝', '‛', '√', ';', '】', '▼',

            '.', '~', '`', '。', 'ə', '］', '，', '{', '～', '！', '†', '‘', '﹏', '═', '｣', '〕', '〜', '＼', '▒', '＄', '♥', '〛', '≤', '∞', '_', '[', '＆', '→', '»', '－', '＝', '§', '⋅', 

            '▓', '&', 'Â', '＞', '〃', '|', '¦', '—', '╚', '〖', '―', '¸', '³', '®', '｠', '¨', '‟', '＊', '£', '#', 'Ã', "'", '▀', '·', '？', '、', '█', '”', '＃', '⊕', '=', '〟', '½', '』',

            '［', '$', ')', 'θ', '@', '›', '＠', '｝', '¬', '…', '¼', '：', '¥', '❤', '€', '−', '＜', '(', '〘', '▄', '＇', '>', '₤', '₹', '∅', 'è', '〿', '「', '©', '｢', '∙', '°', '｜', '¡', 

            '↑', 'º', '¯', '♫', '#'

          }





mispell_dict = {"aren't" : "are not", "can't" : "cannot", "couldn't" : "could not",

"couldnt" : "could not", "didn't" : "did not", "doesn't" : "does not",

"doesnt" : "does not", "don't" : "do not", "hadn't" : "had not", "hasn't" : "has not",

"haven't" : "have not", "havent" : "have not", "he'd" : "he would", "he'll" : "he will", "he's" : "he is", "i'd" : "I would",

"i'd" : "I had", "i'll" : "I will", "i'm" : "I am", "isn't" : "is not", "it's" : "it is",

"it'll":"it will", "i've" : "I have", "let's" : "let us", "mightn't" : "might not", "mustn't" : "must not", 

"shan't" : "shall not", "she'd" : "she would", "she'll" : "she will", "she's" : "she is", "shouldn't" : "should not", "shouldnt" : "should not",

"that's" : "that is", "thats" : "that is", "there's" : "there is", "theres" : "there is", "they'd" : "they would", "they'll" : "they will",

"they're" : "they are", "theyre":  "they are", "they've" : "they have", "we'd" : "we would", "we're" : "we are", "weren't" : "were not",

"we've" : "we have", "what'll" : "what will", "what're" : "what are", "what's" : "what is", "what've" : "what have", "where's" : "where is",

"who'd" : "who would", "who'll" : "who will", "who're" : "who are", "who's" : "who is", "who've" : "who have", "won't" : "will not", "wouldn't" : "would not", "you'd" : "you would",

"you'll" : "you will", "you're" : "you are", "you've" : "you have", "'re": " are", "wasn't": "was not", "we'll":" will", "didn't": "did not", "tryin'":"trying"}





def clean_punct(text):

  text = str(text)

  for punct in PUNCTS:

    text = text.replace(punct, ' {} '.format(punct))

  

  return text



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re

kp = KeywordProcessor(case_sensitive=True)
for k, v in mispell_dict.items():

    kp.add_keyword(k, v)
def preprocessing(text):

    text = kp.replace_keywords(text)

    text = clean_punct(text)

    text = re.sub(r'\n\r', ' ', text)

    text = re.sub(r'\s{2,}', ' ', text)

    

    return text.split()
train['clean_title'] = train['question_title'].apply(lambda x : preprocessing(x))

train['clean_body'] = train['question_body'].apply(lambda x : preprocessing(x))

train['clean_answer'] = train['answer'].apply(lambda x : preprocessing(x))



test['clean_title'] = test['question_title'].apply(lambda x : preprocessing(x))

test['clean_body'] = test['question_body'].apply(lambda x : preprocessing(x))

test['clean_answer'] = test['answer'].apply(lambda x : preprocessing(x))
y_columns = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
def build_matrix(word_index):



    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    

    unknown_words = []

    unk = {}

    known_count = 0

    unk_count = 0

    for word, i in word_index.items():

        if word in MODEL:

            embedding_matrix[i] = MODEL[word]

            known_count += 1

            continue

        if word.lower() in MODEL:

            embedding_matrix[i] = MODEL[word.lower()]

            known_count += 1

            continue    

        if word.upper() in MODEL:

            embedding_matrix[i] = MODEL[word.upper()]

            known_count += 1

            continue

        if word.capitalize() in MODEL:

            embedding_matrix[i] = MODEL[word.capitalize()]

            known_count += 1

            continue

        if unidecode.unidecode(word) in MODEL:

            embedding_matrix[i] = MODEL[unidecode.unidecode(word)]

            known_count += 1

            continue

        try:

            unk[word] += 1 

        except:

            unk[word] = 1

        

        unk_count += 1

    

    

    print('all token in embedding percentage : {:.2f}%'.format( known_count/(unk_count+known_count)  * 100))

#     print('token in embedding percentage : {}'.format( known_count/(unk_count+known_count)))

    return embedding_matrix, unk
tokenizer = text.Tokenizer(filters='', lower=False)





tokenizer.fit_on_texts(list(train['clean_title']) + list(train['clean_body']) + list(train['clean_answer']) \

                        + list(test['clean_title']) + list(test['clean_body']) + list(test['clean_answer']))



TITLE_MAX_LEN = 50

BODY_MAX_LEN = 500

ANSWER_MAX_LEN = 500

train['clean_title_len'] = train['clean_title'].apply(lambda x : len(x))

train['clean_body_len'] = train['clean_body'].apply(lambda x : len(x))

train['clean_answer_len'] = train['clean_answer'].apply(lambda x : len(x))





test['clean_title_len'] = test['clean_title'].apply(lambda x : len(x))

test['clean_body_len'] = test['clean_body'].apply(lambda x : len(x))

test['clean_answer_len'] = test['clean_answer'].apply(lambda x : len(x))



# train title max 58 test 48



# train body max 4924 test body max 1894



# train answer max 8194 test max 2224
x_train_title = tokenizer.texts_to_sequences(train['clean_title'])

x_test_title = tokenizer.texts_to_sequences(test['clean_title'])



x_train_body = tokenizer.texts_to_sequences(train['clean_body'])

x_test_body = tokenizer.texts_to_sequences(test['clean_body'])



x_train_answer = tokenizer.texts_to_sequences(train['clean_answer'])

x_test_answer = tokenizer.texts_to_sequences(test['clean_answer'])





x_train_title = sequence.pad_sequences(x_train_title, maxlen=TITLE_MAX_LEN,padding='post')

x_test_title = sequence.pad_sequences(x_test_title, maxlen=TITLE_MAX_LEN,padding='post')



x_train_body = sequence.pad_sequences(x_train_body, maxlen=BODY_MAX_LEN,padding='post')

x_test_body = sequence.pad_sequences(x_test_body, maxlen=BODY_MAX_LEN,padding='post')



x_train_answer = sequence.pad_sequences(x_train_answer, maxlen=ANSWER_MAX_LEN,padding='post')

x_test_answer = sequence.pad_sequences(x_test_answer, maxlen=ANSWER_MAX_LEN,padding='post')

from sklearn.preprocessing import OneHotEncoder





c = 'host'

onehotencoder = OneHotEncoder(sparse=False, categories='auto').fit(np.concatenate((train[c].values.reshape(-1, 1).astype('str'), test[c].values.reshape(-1, 1).astype('str'))))

train_trans = onehotencoder.transform(train[c].values.reshape(-1, 1).astype('str'))

test_trans = onehotencoder.transform(test[c].values.reshape(-1, 1).astype('str'))

for i in range(train_trans.shape[1]):

    train['{}_{}'.format(c, i)] = train_trans[:, i]

    test['{}_{}'.format(c, i)] = test_trans[:, i]

print('remove origin column : {}'.format(c))

train = train.drop(columns=c)

test = test.drop(columns=c)

gc.collect()



# ## additional feature

# def get_set_char_len(content):

#     set_char = set()

#     for char in ' '.join(content):

#         set_char.add(char)

#     return len(set_char)



# train['title_set_char_len'] = train['clean_title'].apply(lambda x : get_set_char_len(x)) 

# train['body_set_char_len'] = train['clean_body'].apply(lambda x : get_set_char_len(x)) 

# train['answer_set_char_len'] = train['clean_answer'].apply(lambda x : get_set_char_len(x)) 





# test['title_set_char_len'] = test['clean_title'].apply(lambda x : get_set_char_len(x)) 

# test['body_set_char_len'] = test['clean_body'].apply(lambda x : get_set_char_len(x)) 

# test['answer_set_char_len'] = test['clean_answer'].apply(lambda x : get_set_char_len(x)) 





# train_title_len = train['clean_title_len'] / max(train['clean_title_len'])

# train_body_len = train['clean_body_len'] / max(train['clean_body_len'])

# train_answer_len = train['clean_answer_len'] / max(train['clean_answer_len'])



# test_title_len = test['clean_title_len'] / max(test['clean_title_len'])

# test_body_len = test['clean_body_len'] / max(test['clean_body_len'])

# test_answer_len = test['clean_answer_len'] / max(test['clean_answer_len'])



# train_title_set_char_len = train['title_set_char_len'] / max(train['title_set_char_len'])

# train_body_set_char_len = train['body_set_char_len'] / max(train['body_set_char_len'])

# train_answer_set_char_len = train['answer_set_char_len'] / max(train['answer_set_char_len'])



# test_title_set_char_len = test['title_set_char_len'] / max(test['title_set_char_len'])

# test_body_set_char_len = test['body_set_char_len'] / max(test['body_set_char_len'])

# test_answer_set_char_len = test['answer_set_char_len'] / max(test['answer_set_char_len'])



train_category = pd.get_dummies(train['category'].values).values

test_category = pd.get_dummies(test['category'].values).values



hosts = ['host_{}'.format(i) for i in range(64)]

train_host = train.loc[:, hosts].values

test_host = test.loc[:, hosts].values
word2vec_matrix, unk = build_matrix(tokenizer.word_index)



class Attention(nn.Module):

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):

        super(Attention, self).__init__(**kwargs)



        self.supports_masking = True

        self.bias = bias

        self.feature_dim = feature_dim

        self.step_dim = step_dim

        self.features_dim = 0



        weight = torch.zeros(feature_dim, 1)

        nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)



        if bias:

            self.b = nn.Parameter(torch.zeros(1))



    def forward(self, x, mask=None):



        feature_dim = self.feature_dim

        step_dim = self.step_dim



        eij = torch.mm(

            x.contiguous().view(-1, feature_dim), 

            self.weight

        ).view(-1, step_dim)



        if self.bias:

            eij = eij + self.b



        eij = torch.tanh(eij)

        a = torch.exp(eij)



        if mask is not None:

            a = a * mask



        a = a / torch.sum(a, 1, keepdim=True) + 1e-10



        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1)



class SpatialDropout(nn.Module):

    def __init__(self,p):

        super(SpatialDropout, self).__init__()

        self.dropout = nn.Dropout2d(p)

        

    def forward(self, x):

        x = x.permute(0, 2, 1)   # convert to [batch, feature, timestep]

        x = self.dropout(x)

        x = x.permute(0, 2, 1)   # back to [batch, timestep, feature]

        return x



class LSTM_Model(nn.Module):

    def __init__(self, embedding_matrix, hidden_unit, num_layer=1):

        super(LSTM_Model, self).__init__()

        self.max_feature = embedding_matrix.shape[0]

        self.embedding_size = embedding_matrix.shape[1]

      

        self.embedding_body = nn.Embedding(self.max_feature, self.embedding_size)

        self.embedding_body.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding_body.weight.required_grad = False

        

        self.embedding_answer = nn.Embedding(self.max_feature, self.embedding_size)

        self.embedding_answer.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding_answer.weight.required_grad = False

        

        self.embedding_title = nn.Embedding(self.max_feature, self.embedding_size)

        self.embedding_title.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding_title.weight.required_grad = False

        

        self.embedding_dropout = SpatialDropout(0.4)

        

        self.lstm1_body = nn.LSTM(self.embedding_size, hidden_unit, num_layers=num_layer, bidirectional=True, batch_first=True)

        self.lstm2_body = nn.LSTM(hidden_unit*2, int(hidden_unit/2), num_layers=num_layer, bidirectional=True, batch_first=True)

        

        self.lstm1_answer = nn.LSTM(self.embedding_size, hidden_unit, num_layers=num_layer, bidirectional=True, batch_first=True)

        self.lstm2_answer = nn.LSTM(hidden_unit*2, int(hidden_unit/2), num_layers=num_layer, bidirectional=True, batch_first=True)

        

        self.lstm1_title = nn.LSTM(self.embedding_size, hidden_unit, num_layers=num_layer, bidirectional=True, batch_first=True)

        self.lstm2_title = nn.LSTM(hidden_unit*2, int(hidden_unit/2), num_layers=num_layer, bidirectional=True, batch_first=True)

        

        self.attention_body = Attention(hidden_unit, BODY_MAX_LEN)

        self.attention_answer = Attention(hidden_unit, ANSWER_MAX_LEN)

        self.attention_title = Attention(hidden_unit, TITLE_MAX_LEN)

        

#         self.category = nn.Embedding(5, 10)

#         self.host = nn.Embedding(64, 128)

        

        self.linear_title = nn.Linear(hidden_unit*3, hidden_unit)

        self.linear_body = nn.Linear(hidden_unit*3, hidden_unit)

        self.linear_answer = nn.Linear(hidden_unit*3, hidden_unit)

        

#         self.linear_out = nn.Linear(hidden_unit, 30)

        self.additional_category = nn.Linear(5, 5)

        self.additional_host = nn.Linear(64, 32)

        

        self.linear_q = nn.Linear(hidden_unit*2+37, hidden_unit)

        self.linear_a = nn.Linear(hidden_unit+37, hidden_unit)

        self.linear_q_out = nn.Linear(hidden_unit, 21)

        self.linear_a_out = nn.Linear(hidden_unit, 9)

        

    def forward(self, body, answer, title, category, host):

        

        x_body = self.embedding_dropout(self.embedding_body(body))

        h_lstm1_body, _ = self.lstm1_body(x_body)

        h_lstm2_body, _ = self.lstm2_body(h_lstm1_body)

        

        x_answer = self.embedding_dropout(self.embedding_answer(answer))

        h_lstm1_answer, _ = self.lstm1_answer(x_answer)

        h_lstm2_answer, _ = self.lstm2_answer(h_lstm1_answer)

        

        x_title = self.embedding_dropout(self.embedding_title(title))

        h_lstm1_title, _ = self.lstm1_title(x_title)

        h_lstm2_title, _ = self.lstm2_title(h_lstm1_title)

        

#         print(h_lstm2_body.size())

        att_body = self.attention_body(h_lstm2_body)

        att_answer = self.attention_answer(h_lstm2_answer)

        att_title = self.attention_title(h_lstm2_title)

        

        avg_pool_body = torch.mean(h_lstm2_body, 1)

        max_pool_body, _ = torch.max(h_lstm2_body, 1)

        

        avg_pool_answer = torch.mean(h_lstm2_answer, 1)

        max_pool_answer, _ = torch.max(h_lstm2_answer, 1)

        

        avg_pool_title = torch.mean(h_lstm2_title, 1)

        max_pool_title, _ = torch.max(h_lstm2_title, 1)

        

        body_cat = torch.cat((att_body, avg_pool_body, max_pool_body), 1)

        answer_cat = torch.cat((att_answer, avg_pool_answer, max_pool_answer), 1)

        title_cat = torch.cat((att_title, avg_pool_title, max_pool_title), 1)

        

#         additional_feature = self.addtional_linear()



#         category = self.category(category)

#         host = self.category(host)

        

        

        body_cat = torch.relu(self.linear_body(body_cat))

        answer_cat = torch.relu(self.linear_answer(answer_cat))

        title_cat = torch.relu(self.linear_title(title_cat))



        category = self.additional_category(category)

        host = self.additional_host(host)

        

        hidden_q = self.linear_q(torch.cat((title_cat, body_cat, category, host), 1))

        hidden_a = self.linear_a(torch.cat((answer_cat, category, host), 1))

                                          

        q_result = self.linear_q_out(hidden_q)

        a_result = self.linear_a_out(hidden_a)

        

        out = torch.cat([q_result, a_result], 1)

        return out
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
SEED = 2020

NFOLDS = 4

BATCH_SIZE = 32

EPOCHS = 6

LR = 0.001

hidden_unit = 256

seed_everything(SEED)
kf = list(KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED).split(x_train_title))

class TextDataset(torch.utils.data.TensorDataset):



    def __init__(self, body_data, answer_data, title_data, category_data, host_data, idxs, targets=None):

        self.body_data = body_data[idxs]

        self.answer_data = answer_data[idxs]

        self.title_data = title_data[idxs]

        self.category_data = category_data[idxs]

        self.host_data = host_data[idxs]

        self.targets = targets[idxs] if targets is not None else np.zeros((self.body_data.shape[0], 30))



    def __getitem__(self, idx):

        body = self.body_data[idx]

        answer = self.answer_data[idx]

        title = self.title_data[idx]

        category = self.category_data[idx]

        host = self.host_data[idx]

        target = self.targets[idx]



        return body, answer, title, category, host, target



    def __len__(self):

        return len(self.body_data)
test_loader = torch.utils.data.DataLoader(TextDataset(x_test_body, x_test_answer, x_test_title, test_category, test_host, test.index),

                          batch_size=BATCH_SIZE, shuffle=False)
gc.collect()
y = train.loc[:, y_columns].values



oof = np.zeros((len(train), 30))

test_pred = np.zeros((len(test), 30))



# del train, hosts, onehotencoder

# gc.collect()

for i, (train_idx, valid_idx) in enumerate(kf):

    print(f'fold {i+1}')

    gc.collect()

    train_loader = torch.utils.data.DataLoader(TextDataset(x_train_body, x_train_answer, x_train_title, train_category, train_host, train_idx, y),

                          batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    

    val_loader = torch.utils.data.DataLoader(TextDataset(x_train_body, x_train_answer, x_train_title, train_category, train_host, valid_idx, y),

                          batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    

    net = LSTM_Model(word2vec_matrix, hidden_unit)

    net.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)



    for epoch in range(EPOCHS):  

        start_time = time.time()

        avg_loss = 0.0

        net.train()

        for data in train_loader:



            # get the inputs

            body, answer, title, category, host, labels = data

            pred = net(body.long().cuda(), answer.long().cuda(), title.long().cuda(), category.float().cuda(), host.float().cuda())



            loss = loss_fn(pred, labels.cuda())

            # Before the backward pass, use the optimizer object to zero all of the

            # gradients for the Tensors it will update (which are the learnable weights

            # of the model)

            optimizer.zero_grad()



            # Backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters

            optimizer.step()



            avg_loss += loss.item()

        

        avg_val_loss = 0.0

        net.eval()



        valid_preds = np.zeros((len(valid_idx), 30))

        true_label = np.zeros((len(valid_idx), 30))

        for j, data in enumerate(val_loader):



            # get the inputs

            body, answer, title, category, host, labels = data



            ## forward + backward + optimize

            pred = net(body.long().cuda(), answer.long().cuda(), title.long().cuda(), category.float().cuda(), host.float().cuda())



            loss_val = loss_fn(pred, labels.cuda())

            avg_val_loss += loss_val.item()



            valid_preds[j * BATCH_SIZE:(j+1) * BATCH_SIZE] = torch.sigmoid(pred).cpu().detach().numpy()

            true_label[j * BATCH_SIZE:(j+1) * BATCH_SIZE]  = labels

            

        score = 0

        for i in range(30):

            score += np.nan_to_num(

                    spearmanr(true_label[:, i], valid_preds[:, i]).correlation / 30)

        oof[valid_idx] = valid_preds

        elapsed_time = time.time() - start_time 

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t spearman={:.2f} \t time={:.2f}s'.format(

            epoch + 1, EPOCHS, avg_loss / len(train_loader), avg_val_loss / len(val_loader), score, elapsed_time))

        

    test_pred_fold = np.zeros((len(test), 30))

        

    with torch.no_grad():

        for q, data in enumerate(test_loader):

            body, answer, title, category, host, _ = data

            y_pred = net(body.long().cuda(), answer.long().cuda(), title.long().cuda(), category.float().cuda(), host.float().cuda())

            test_pred_fold[q * BATCH_SIZE:(q+1) * BATCH_SIZE] = torch.sigmoid(y_pred).cpu().detach().numpy()

    test_pred += test_pred_fold/NFOLDS

        

        
sub.loc[:, y_columns] = test_pred

sub.to_csv('submission.csv', index=False)
sub.head()