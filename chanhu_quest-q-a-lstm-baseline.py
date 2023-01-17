import numpy as np

import pandas as pd

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import text

import os

from tqdm import tqdm

import torch

import torch.nn as nn

import pickle

import gc

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split, KFold

import torch.nn.functional as F

import os

import random

import time

import pickle

import joblib

from sklearn.preprocessing import LabelEncoder
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'

GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

train_csv_path       = '../input/google-quest-challenge/train.csv'

test_csv_path        = '../input/google-quest-challenge/test.csv'

seed                 = 0

epochs               = 50

seed_everything(seed)
train = pd.read_csv(train_csv_path)

test  = pd.read_csv(test_csv_path)
train.columns
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path,'rb') as f:

        emb_arr = pickle.load(f)

    return emb_arr
def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        if i <= len(word_index) + 1:

            try:

                embedding_matrix[i] = embedding_index[word]

            except KeyError:

                try:

                    embedding_matrix[i] = embedding_index[word.lower()]

                except KeyError:

                    try:

                        embedding_matrix[i] = embedding_index[word.title()]

                    except KeyError:

                        unknown_words.append(word)

                        

    return embedding_matrix, unknown_words
X_train_question = train['question_body']

X_train_title    = train['question_title']

X_train_answer   = train['answer']



X_test_question  = test['question_body']

X_test_title     = test['question_title']

X_test_answer    = test['answer']
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(X_train_question) + \

                       list(X_train_answer) + \

                       list(X_train_title) + \

                       list(X_test_question) + \

                       list(X_test_answer) + \

                       list(X_test_title)

                      )
crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

print('n unknown words (crawl): ', len(unknown_words_crawl))



glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

print('n unknown words (glove): ', len(unknown_words_glove))
embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

print(embedding_matrix.shape)



del crawl_matrix

del glove_matrix

gc.collect()
X_train_question = tokenizer.texts_to_sequences(X_train_question)

X_train_title    = tokenizer.texts_to_sequences(X_train_title)

X_train_answer   = tokenizer.texts_to_sequences(X_train_answer)





X_test_question  = tokenizer.texts_to_sequences(X_test_question)

X_test_title     = tokenizer.texts_to_sequences(X_test_title)

X_test_answer    = tokenizer.texts_to_sequences(X_test_answer)
question_length_max = np.array([len(x) for x in X_train_question]).max()

answer_length_max   = np.array([len(x) for x in X_train_answer]).max()

print(question_length_max, answer_length_max)

question_length_max = np.array([len(x) for x in X_test_question]).max()

answer_length_max   = np.array([len(x) for x in X_test_answer]).max()

print(question_length_max, answer_length_max)
X_train_question = pad_sequences(X_train_question, maxlen=300)

X_train_answer   = pad_sequences(X_train_answer, maxlen=300)

X_train_title    = pad_sequences(X_train_title,  maxlen=50)



le_cat = LabelEncoder()

le_host = LabelEncoder()

train_cat_enc = le_cat.fit_transform(train.category)

train_host_enc = le_host.fit_transform(train.host)
train.category = train_cat_enc

train.host = train_host_enc
class QuestDataset(Dataset):

    

    def __init__(self, df, questions, answers, titles):

        

        self.df         = df

        self.questions  = questions

        self.answers    = answers

        self.titles     = titles

        self.categories = self.df.category.values

        self.hosts      = self.df.host.values

        self.question_cols = ['question_asker_intent_understanding',

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

                              'question_well_written']

        self.answer_cols   = ['answer_helpful',

                              'answer_level_of_information', 'answer_plausible', 'answer_relevance',

                              'answer_satisfaction', 'answer_type_instructions',

                              'answer_type_procedure', 'answer_type_reason_explanation',

                              'answer_well_written']

        

        

        self.label = self.df[self.question_cols + self.answer_cols].values

        

    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        

        question = self.questions[idx]

        answer   = self.answers[idx]

        title    = self.titles[idx]

        category = self.categories[idx]

        host     = self.hosts[idx]

        

        

        labels = self.label[idx]

        

        return [question, answer, title], labels
class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)    # (N, T, 1, K)

        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)

        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked

        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)

        x = x.squeeze(2)  # (N, T, K)

        return x
class QuestModel(nn.Module):

    

    def __init__(self, embedding_matrix):

        super().__init__()

        

        LSTM_UNITS = 128

        embed_size = embedding_matrix.shape[1]

        DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

        max_features = 30000

        

        self.embedding        = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout(0.3)

        

        

        ###########################################################

        #LSTM 

        ##########################################################

        self.lstm_q_1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm_q_2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        

        self.lstm_a_1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm_a_2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        

        self.lstm_t_1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm_t_2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        

    

        self.linear1 = nn.Sequential(nn.Linear(DENSE_HIDDEN_UNITS * 2, DENSE_HIDDEN_UNITS),

                                     nn.BatchNorm1d(DENSE_HIDDEN_UNITS),

                                     nn.ReLU(inplace=True),

                                     nn.Dropout(0.5))

            

        self.bilinear = nn.Bilinear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2  = nn.Sequential(nn.Linear(DENSE_HIDDEN_UNITS*3, DENSE_HIDDEN_UNITS),

                                      nn.BatchNorm1d(DENSE_HIDDEN_UNITS),

                                      nn.ReLU(inplace=True),

                                      nn.Dropout(0.5))

        

        self.linear_q_out  = nn.Linear(DENSE_HIDDEN_UNITS, 21)

        self.linear_aq_out = nn.Linear(DENSE_HIDDEN_UNITS, 9)

        

    def forward(self, question, answer, title):

        

        #######################################

        #Question

        #######################################

        

        question_embedding = self.embedding(question.long())

        question_embedding = self.embedding_dropout(question_embedding)

        

        q_lstm1, _ = self.lstm_q_1(question_embedding)

        q_lstm2, _ = self.lstm_q_2(q_lstm1)

        

        q_avg_pool    = torch.mean(q_lstm2, 1)

        q_max_pool, _ = torch.max(q_lstm2, 1)

        

        #######################################

        #answer

        #######################################

        answer_embedding   = self.embedding(answer.long())

        answer_embedding   = self.embedding_dropout(answer_embedding)

        

        a_lstm1, _ = self.lstm_a_1(answer_embedding)

        a_lstm2, _ = self.lstm_a_2(a_lstm1)

        

        a_avg_pool    = torch.mean(a_lstm2, 1)

        a_max_pool, _ = torch.max(a_lstm2, 1)

        

        #######################################

        #title

        #######################################

        

        title_embedding   = self.embedding(title.long())

        title_embedding   = self.embedding_dropout(title_embedding)

        

        t_lstm1, _ = self.lstm_t_1(title_embedding)

        t_lstm2, _ = self.lstm_t_2(t_lstm1)

        

        t_avg_pool    = torch.mean(t_lstm2, 1)

        t_max_pool, _ = torch.max(t_lstm2, 1)

        

        q_features = torch.cat((q_max_pool, q_avg_pool), 1) #LSTM_UNIT * 4

        a_features = torch.cat((a_max_pool, a_avg_pool), 1) #LSTM_UNIT * 4

        t_features = torch.cat((t_max_pool, t_avg_pool), 1) #LSTM_UNIT * 4

        

        hidden_q  = self.linear1(torch.cat((q_features, t_features), 1)) #LSTM * 8

        bil_sim   = self.bilinear(q_features, a_features)      

        hidden_aq = self.linear2(torch.cat((q_features, a_features, bil_sim), 1))

        

        q_result  = self.linear_q_out(hidden_q)

        aq_result = self.linear_aq_out(hidden_aq)

        out = torch.cat([q_result, aq_result], 1)

        

        return out
def train_model(train_loader, optimizer, criterion):

    

    model.train()

    avg_loss = 0.

    

    for idx, (inputs, labels) in enumerate(train_loader):

        questions, answers, title = inputs

        questions, answers, title = questions.cuda(), answers.cuda(), title.cuda()

        labels = labels.float().cuda()

        

        optimizer.zero_grad()

        output_train = model(questions, answers, title)

        loss = criterion(output_train,labels)

        loss.backward() 

        optimizer.step()

        avg_loss += loss.item() / len(train_loader)

        

    return avg_loss



def val_model(val_loader):

    avg_val_loss = 0.

    model.eval() #実行モード

    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(val_loader):

            questions, answers, title = inputs

            questions, answers, title = questions.cuda(), answers.cuda(), title.cuda()

            labels = labels.float().cuda()

            output_val = model(questions, answers, title)

            avg_val_loss += criterion(output_val, labels).item() / len(val_loader)

        

    return avg_val_loss
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

for fold, (train_index, val_index) in enumerate(kf.split(range(len(train)))):

    print("fold:", fold)

    train_df = train.iloc[train_index]

    val_df   = train.iloc[val_index]

    

    train_set    = QuestDataset(train_df, X_train_question[train_index], 

                                X_train_answer[train_index],

                                X_train_title[train_index])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    

    val_set      = QuestDataset(val_df, 

                                X_train_question[val_index], 

                                X_train_answer[val_index],

                                X_train_title[val_index],)

    val_loader   = DataLoader(val_set, batch_size=128, shuffle=False)

    

    model = QuestModel(embedding_matrix)

    model.cuda()

    

    best_avg_loss   = 100.0

    best_param_loss = None

    i = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    

    for epoch in range(epochs):

        

        if i == 5: break

        #print(optimizer.param_groups[0]['lr'])

        start_time   = time.time()

        avg_loss     = train_model(train_loader, optimizer, criterion)

        avg_val_loss = val_model(val_loader)

        elapsed_time = time.time() - start_time 

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format( epoch + 1, epochs, avg_loss, avg_val_loss, elapsed_time))

    

        if best_avg_loss > avg_val_loss:

            i = 0

            best_avg_loss = avg_val_loss 

            best_param_loss = model.state_dict()

        else:

            i += 1

            

        torch.save(best_param_loss, 'weight_loss_best_{}.pt'.format(fold))
with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)