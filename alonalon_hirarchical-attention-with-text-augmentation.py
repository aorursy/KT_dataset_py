import numpy as np 

import pandas as pd 

import sentencepiece as spm

import re, json, gc, pickle, shutil, sys

import random, os, nltk, multiprocessing



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader, Dataset

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



from scipy.stats import spearmanr

from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import MinMaxScaler, StandardScaler



from gensim.summarization.textcleaner import get_sentences



from tqdm import tqdm_notebook as tqdm

import seaborn as sns



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)



gc.enable()



seed = 3

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True



ROOT  = '/kaggle/input/google-quest-challenge/' 

data  = pd.read_csv(ROOT+'train.csv')

test  = pd.read_csv(ROOT+'test.csv')

sub   = pd.read_csv(ROOT+'sample_submission.csv')



data.head(3)
text_col        = ['question_title',

                   'question_body',

                   'answer']



numeric_col     = []



categoricals    = ['category',

                   'host']



label_col       = ['question_asker_intent_understanding',

                   'question_body_critical',

                   'question_conversational',

                   'question_expect_short_answer',

                   'question_fact_seeking',

                   'question_has_commonly_accepted_answer',

                   'question_interestingness_others',

                   'question_interestingness_self',

                   'question_multi_intent',

                   'question_not_really_a_question',

                   'question_opinion_seeking',

                   'question_type_choice',

                   'question_type_compare',

                   'question_type_consequence',

                   'question_type_definition',

                   'question_type_entity',

                   'question_type_instructions',

                   'question_type_procedure',

                   'question_type_reason_explanation',

                   'question_type_spelling',

                   'question_well_written',

                   'answer_helpful',

                   'answer_level_of_information',

                   'answer_plausible',

                   'answer_relevance',

                   'answer_satisfaction',

                   'answer_type_instructions',

                   'answer_type_procedure',

                   'answer_type_reason_explanation',

                   'answer_well_written']
def simple_tokenizer(Text, ReturnList=True, with_punc=False, lower=True):           

    Text = re.sub(r'\s\d{2}:\d{2}',' <hour> ', Text)

    Text = re.sub(r'\s\d{1,2}[./]\d{1,2}[./](\d{4}|\d{2})',' <date> ', Text)    

    #Text = re.sub('[0-9,-]{5,}', '', Text)    

    #Text = re.sub('[0-9]{1,}'  , '', Text)

    

    if with_punc:

        Text = re.sub(r'\n+\s*',' \n ', Text)

        Text = re.sub(r'\t+',' \t ', Text)

        Text = re.sub(r'([,]+\s*)+',' , ', Text)

        Text = re.findall(r"[\w'.]+[\"'.][\w']+|[\w']+|[.,!?:;$]", Text)    

    else:

        Text = re.findall(r"[\w'.]+[\"'.][\w']+|[\w']+", Text) 

    

    if lower:

         Text = [txt.lower() for txt in Text]

    

    if ReturnList:

        return Text

    return ' '.join(Text)
data['is_same_user'] = data.apply(lambda row: 1 if row['question_user_name']==row['answer_user_name'] else 0, axis=1)

test['is_same_user'] = test.apply(lambda row: 1 if row['question_user_name']==row['answer_user_name'] else 0, axis=1)

numeric_col.append('is_same_user')



def feature_extractor(data):

    meta_features = []

    for col in tqdm(text_col): 



        pct_Question_Marks       = []

        pct_Exclamation_marks    = [] 

        pct_newlines             = []

        pct_repeated_chars       = []

        total_len                = []

        number_of_words_list     = []

        number_of_sentences_list = []



        for text in data[col].values:



            nuber_of_Question_Mark     = text.count('?')

            number_of_Exclamation_mark = text.count('!')

            number_of_words            = len(simple_tokenizer(text))

            number_of_sentences        = len(list(get_sentences(text)))



            lenn = len(text)



            pct_Question_Mark    = nuber_of_Question_Mark    /number_of_words if number_of_words>0 else 0

            pct_Exclamation_mark = number_of_Exclamation_mark/number_of_words if number_of_words>0 else 0



            prev_char = 'random init char'

            counter = 0

            repeated_counts = []

            for current_char in text:

                if current_char==prev_char:

                    counter+=1

                else:

                    counter = 0

                repeated_counts.append(counter)

                prev_char = current_char



            pct_repeated_char = sum(repeated_counts)/number_of_words if number_of_words>0 else 0



            pct_Question_Marks.append(pct_Question_Mark)

            pct_Exclamation_marks.append(pct_Exclamation_mark)

            pct_repeated_chars.append(pct_repeated_char)

            total_len.append(lenn)   

            number_of_words_list.append(number_of_words)     

            number_of_sentences_list.append(number_of_sentences) 



        data[col+'_Question_Marks']      = pct_Question_Marks

        data[col+'_Exclamation_marks']   = pct_Exclamation_marks

        data[col+'_repeated_chars']      = pct_repeated_chars

        data[col+'_number_of_words']     = number_of_words_list

        data[col+'_number_of_sentences'] = number_of_sentences_list

    

        meta_features += [col+'_Question_Marks', 

                          col+'_Exclamation_marks', 

                          col+'_repeated_chars', 

                          col+'_number_of_words',

                          col+'_number_of_sentences']



    return data, meta_features





data, meta_features = feature_extractor(data)

test, meta_features = feature_extractor(test)



numeric_col += meta_features

data.head(3)
scalers = {}



for col in numeric_col:

    Scaler    = MinMaxScaler()

    data[col] = Scaler.fit_transform(data[col].values.reshape(-1, 1)) 

    assert Scaler.n_samples_seen_ == len(data[col])

    test[col] = Scaler.transform(test[col].values.reshape(-1, 1))

    scalers[col] = Scaler
# corr_matrix = data[meta_features+label_col].corr().abs()

# corr        = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))

# corr.head(30)
data = data.astype({cat:'category' for cat in categoricals})



def Get_Cat2IDs(data, categoricals_cols, IX_start):

    encode = {}

    for col in categoricals_cols:

        encode[col] = dict(enumerate(data[col].cat.categories, start=IX_start)) 



    Cat2Int = {}

    for cat in categoricals_cols:

        ValueKey = {}

        for key, value in encode[cat].items():

            ValueKey[value] = key

        Cat2Int[cat] = ValueKey

    return Cat2Int

    

Cat2Int = Get_Cat2IDs(data = data, 

                      categoricals_cols = categoricals, 

                      IX_start = 1) #index 0 will be for OOV

categorical_sizes = [data[c].nunique() + 1 for c in categoricals]





cat_oov_value = 0

for col in categoricals:

    data[col] = data[col].map(Cat2Int[col])



for col in categoricals:

    test[col] = test[col].map(Cat2Int[col]).fillna(cat_oov_value)

    

del Cat2Int

gc.collect()
all_sentences = [] 



for col in text_col:

    for text in list(data[col].values)+list(test[col].values):

        sentences = get_sentences(text)

        for sentence in sentences:

            all_sentences.append(sentence)

            

with open('one_file_train_SentencePice.txt','a', encoding="utf8") as wf:

    for sentence in all_sentences:

        wf.write(sentence + '\n')
spm.SentencePieceTrainer.Train('--input=one_file_train_SentencePice.txt --model_prefix=sp_model --vocab_size=2000 --pad_id=3')
sp = spm.SentencePieceProcessor()

sp.Load('sp_model.model')
for n in range(10):

    print(sp.SampleEncodeAsPieces('New York', -1, 0.1))
def get_token2ID():

    with open('sp_model.vocab', 'r', encoding="utf8") as f:

        lines = f.readlines()

    tokens      = list(enumerate([line.split("\t")[0] for line in lines]))

    token_to_id = dict([(v,k) for (k,v) in tokens])

    return token_to_id



token_to_id = get_token2ID()
sss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=3)

train_IX, test_IX = iter(next(sss.split(data[text_col+numeric_col+categoricals], data[label_col])))   



train = data.loc[train_IX]

val   = data.loc[test_IX]



train.reset_index(drop=True, inplace=True)

val.  reset_index(drop=True, inplace=True)
def create_path(folder):

    if not os.path.isdir(folder):

        os.makedirs(folder)

        os.makedirs(f'{folder}/categorical')

        os.makedirs(f'{folder}/numerical')

        os.makedirs(f'{folder}/question_title')

        os.makedirs(f'{folder}/question_body')

        os.makedirs(f'{folder}/answer')

        os.makedirs(f'{folder}/label')

    else:

        shutil.rmtree(folder)

        os.makedirs(folder)

        os.makedirs(f'{folder}/categorical')

        os.makedirs(f'{folder}/numerical')

        os.makedirs(f'{folder}/question_title')

        os.makedirs(f'{folder}/question_body')

        os.makedirs(f'{folder}/answer')

        os.makedirs(f'{folder}/label')

            

create_path(folder='data')
def save_data(data, folder, part):

    

    for IX, row in tqdm(data.iterrows(), total=len(data)):



        #question_body

        sentences = list(get_sentences(row['question_body']))

        sentences = json.dumps(sentences)

        with open(f'{folder}/question_body/' +str(row['qa_id'])+'.json', 'w') as f:

            json.dump(sentences, f)



        #answer 

        sentences = list(get_sentences(row['answer']))

        sentences = json.dumps(sentences)

        with open(f'{folder}/answer/'        +str(row['qa_id'])+'.json', 'w') as f:

            json.dump(sentences, f)    



        #question_title 

        sentences = [row['question_title']]

        sentences = json.dumps(sentences)

        with open(f'{folder}/question_title/'+str(row['qa_id'])+'.json', 'w') as f:

            json.dump(sentences, f)    



        #numerical features:

        numericals = torch.tensor(row[numeric_col])

        torch.save(numericals, 

                   f'{folder}/numerical/'    +str(row['qa_id'])+'.pt')



        #categorical features:

        categorical = torch.tensor(row[categoricals])

        torch.save(categorical, 

                   f'{folder}/categorical/'  +str(row['qa_id'])+'.pt')

        

        #labels:

        if part!='test':

            labels = torch.tensor(row[label_col])

            torch.save(labels, 

                       f'{folder}/label/'     +str(row['qa_id'])+'.pt')
save_data(train,'data','train')

save_data(val,  'data','val')

save_data(test, 'data','test')
[n for n in data.qa_id.values if n in test.qa_id.values]
def to_sequence(text):

    seq = []

    for token in text:

        if token in token_to_id:

            id = token_to_id[token]

        else:

            id = token_to_id['<unk>']

        seq.append(id)

    return np.array(seq, dtype='int16')
class QA_Dataset(Dataset):

    def __init__(self, 

                 data, 

                 cat_path        ='/categorical/', 

                 num_path        ='/numerical/', 

                 Q_title_path    ='/question_title/',

                 Q_body_path     ='/question_body/',

                 answer_path     ='/answer/',

                 label_path      = None, 

                 folder          ='data',

                 ID_Column       ='qa_id',

                 padding_val     = token_to_id['<pad>'],

                 max_n_words     = 100,

                 max_n_sentances = 100,

                 augmentation    = True):

        

        self.data            = data        

        self.cat_path        = cat_path

        self.num_path        = num_path

        self.Q_title_path    = Q_title_path

        self.Q_body_path     = Q_body_path

        self.answer_path     = answer_path

        self.label_path      = label_path   

        self.folder          = folder   

        self.ID_Column       = ID_Column

        self.padding_val     = padding_val

        self.max_n_words     = max_n_words

        self.max_n_sentances = max_n_sentances

        self.augmentation    = augmentation

        self.len             = self.data.shape[0]

        

    def __getitem__(self, IXs):



        ID = self.data.loc[IXs, self.ID_Column]



        ### categorical ###

        categorical = torch.load(self.folder+self.cat_path+str(ID)+'.pt')

        

        ### numerical ###

        numerical   = torch.load(self.folder+self.num_path+str(ID)+'.pt')

        

        

        ### question title ###

        Q_title_IDs = []

        with open(self.folder+self.Q_title_path+str(ID)+'.json') as file:

            Q_title_sentance = json.loads( json.load(file) )

        subwords     = (sp.SampleEncodeAsPieces(Q_title_sentance[0], -1, 0.1) 

                        if self.augmentation 

                        else sp.EncodeAsPieces(Q_title_sentance[0]))

        subwords_IDs = to_sequence(subwords)

        Q_title_IDs  = torch.tensor(subwords_IDs) 

        

        

        ### question body: ###

        Q_body_IDs   = []  

        n_words_QBody = []

        with open(self.folder+self.Q_body_path+str(ID)+'.json') as file:

            Q_body_sentances = json.loads( json.load(file) )

        for sentance in Q_body_sentances:

            subwords = (sp.SampleEncodeAsPieces(sentance, -1, 0.1) 

                        if self.augmentation 

                        else sp.EncodeAsPieces(sentance))

            if len(subwords) > self.max_n_words:

                subwords = subwords[:self.max_n_words]

            subwords_IDs = to_sequence(subwords)

            Q_body_IDs.append(torch.tensor(subwords_IDs))

        if len(Q_body_IDs)==0:

            Q_body_IDs = [torch.tensor([self.padding_val], dtype=torch.int16)]

        elif len(Q_body_IDs) > self.max_n_sentances:

            Q_body_IDs = Q_body_IDs[:self.max_n_sentances]

        Q_body_IDs = pad_sequence(Q_body_IDs,  

                                  batch_first=True, 

                                  padding_value=self.padding_val)

        

        

        ### Answer ###

        answer_IDs = []

        with open(self.folder+self.answer_path+str(ID)+'.json') as file:

            answer_sentances = json.loads( json.load(file) )

        for sentance in answer_sentances:

            subwords     = (sp.SampleEncodeAsPieces(sentance, -1, 0.1) 

                            if self.augmentation 

                            else sp.EncodeAsPieces(sentance))

            if len(subwords) > self.max_n_words:

                subwords = subwords[:self.max_n_words]            

            subwords_IDs = to_sequence(subwords)

            answer_IDs.append(torch.tensor(subwords_IDs))

        if len(answer_IDs)==0:

            answer_IDs = [torch.tensor([self.padding_val], dtype=torch.int16)]

        elif len(answer_IDs) > self.max_n_sentances:

            answer_IDs = answer_IDs[:self.max_n_sentances]            

        answer_IDs     = pad_sequence(answer_IDs,  

                                      batch_first=True, 

                                      padding_value=self.padding_val)

        

        

        ### labels ###

        if self.label_path is not None:

            labels  = torch.load(self.folder+self.label_path+str(ID)+'.pt')

            

            

            return [categorical, numerical, Q_title_IDs,Q_body_IDs, answer_IDs, labels]

        return     [categorical, numerical, Q_title_IDs,Q_body_IDs, answer_IDs]

    

    def __len__(self):

        return self.len
class MyCollator(object):

    def __init__(self, 

                 padding_val = token_to_id['<pad>'], 

                 is_test     = False):

        

        self.padding_val = padding_val

        self.is_test     = is_test

    

    def __call__(self, batch):

        

        cat            =     [item[0]          for item in batch]

        num            =     [item[1]          for item in batch]

        Q_title_IDs    =     [item[2]          for item in batch]

        Q_body_IDs     =     [item[3]          for item in batch]

        Q_body_MaxSent = max([item[3].shape[0] for item in batch])

        Q_body_MaxWord = max([item[3].shape[1] for item in batch])

        answer_IDs     =     [item[4]          for item in batch]

        answer_MaxSent = max([item[4].shape[0] for item in batch])

        answer_MaxWord = max([item[4].shape[1] for item in batch])

        

        if not self.is_test:

            labels     =     [item[5]          for item in batch]

        

        

        Q_title_IDs    = pad_sequence(Q_title_IDs, 

                                      batch_first=True, 

                                      padding_value=self.padding_val)

        

        BATCH_SIZE     = len(Q_body_IDs)

        pad_Q_body_IDs = Q_body_IDs[0].new_full(size = (BATCH_SIZE, 

                                                        Q_body_MaxSent, 

                                                        Q_body_MaxWord), 

                                                fill_value = self.padding_val)

        

        for IX, QB_IDs in enumerate(Q_body_IDs): 

            pad_Q_body_IDs[IX, :QB_IDs.shape[0], :QB_IDs.shape[1]] = QB_IDs

            

            

        pad_answer_IDs = answer_IDs[0].new_full(size = (BATCH_SIZE, 

                                                        answer_MaxSent, 

                                                        answer_MaxWord), 

                                                fill_value = self.padding_val)    

        for IX, A_ID in enumerate(answer_IDs): 

            pad_answer_IDs[IX, :A_ID.shape[0], :A_ID.shape[1]] = A_ID

            

            

        if not self.is_test:

            return [torch.stack(cat),        

                    torch.stack(num),        

                    Q_title_IDs.type(torch.long),

                    pad_Q_body_IDs.type(torch.long), 

                    pad_answer_IDs.type(torch.long), 

                    torch.stack(labels)]

        

        return [torch.stack(cat),        

                torch.stack(num),        

                Q_title_IDs.type(torch.long),

                pad_Q_body_IDs.type(torch.long), 

                pad_answer_IDs.type(torch.long)]
class QA_Net(nn.Module):

    def __init__(self,

                 EMBEDDING_DIM,

                 HIDDEN_SIZE):

        super(QA_Net, self).__init__()

        

        self.word_LSTM     = nn.LSTM(input_size   = EMBEDDING_DIM, 

                                    hidden_size  = HIDDEN_SIZE, 

                                    batch_first  = True, 

                                    bidirectional= True)

        self.word_proj    = nn.Linear(in_features=2*HIDDEN_SIZE, 

                                      out_features= HIDDEN_SIZE)

        

        word_atten_weight = torch.randn((HIDDEN_SIZE,1))

        nn.init.xavier_uniform_(word_atten_weight)     

        self.word_context = nn.Parameter(word_atten_weight)

        

        self.softmax      = nn.Softmax(dim=1)

        

        self.sent_LSTM    = nn.LSTM(input_size   =2*HIDDEN_SIZE, 

                                    hidden_size  =HIDDEN_SIZE, 

                                    batch_first  =True, 

                                    bidirectional=True)

        

        self.sent_proj    = nn.Linear(in_features  = 2*HIDDEN_SIZE, 

                                      out_features = HIDDEN_SIZE)



        sent_atten_weight = torch.randn((HIDDEN_SIZE,1))

        nn.init.xavier_uniform_(sent_atten_weight)             

        self.sent_context = nn.Parameter(sent_atten_weight)

        

    def forward(self, embeddings, batch_size):



        result,_ = self.word_LSTM(embeddings)

        

        u_it     = torch.tanh(self.word_proj(result))

        

        w_scores = self.softmax(u_it.matmul(self.word_context)) 

        result   = result.mul(w_scores)

        result   = torch.sum(result, dim=1)

        

        #from token level to sentance level

        result   = result.view(batch_size,

                               int(result.shape[0]/batch_size), 

                               result.shape[-1])

        

        result,_ = self.sent_LSTM(result)

        

        u_i      = torch.tanh(self.sent_proj(result))

        s_scores = self.softmax(u_i.matmul(self.sent_context))

        result   = result.mul(s_scores)

        result   = torch.sum(result, dim=1)

        

        return result        
class QTitle_Net(nn.Module):

    def __init__(self,

                 EMBEDDING_DIM,

                 HIDDEN_SIZE):

        super(QTitle_Net, self).__init__()

        

        self.lstm            =  nn.LSTM(input_size    = EMBEDDING_DIM, 

                                        hidden_size   = HIDDEN_SIZE,

                                        batch_first   = True,

                                        bidirectional = True,

                                        num_layers    = 1) 

        

        self.word_proj    = nn.Linear(in_features =2*HIDDEN_SIZE, 

                                      out_features=HIDDEN_SIZE)

        

        word_atten_weight = torch.randn((HIDDEN_SIZE,1))

        nn.init.xavier_uniform_(word_atten_weight)                     

        self.word_context = nn.Parameter(word_atten_weight)

        

        self.softmax      = nn.Softmax(dim=1)        

        

    def forward(self, embeddings): 

        

        result,_ = self.lstm(embeddings)

        

        u_it     = torch.tanh(self.word_proj(result))

        w_scores = self.softmax(u_it.matmul(self.word_context)) 

        result   = result.mul(w_scores)

        result   = torch.sum(result, dim=1)

        

        return result
class Main_Model(nn.Module):

    def __init__(self,

                 categorical_sizes = categorical_sizes,

                 EMBEDDING_DIM     = 100,

                 HIDDEN_SIZE       = 100,

                 len_numerical     = len(numeric_col),

                 out_size          = len(label_col)):

        super(Main_Model, self).__init__()



        embedding_dims      = [(c, min(50, (c+1)//2)) for c in categorical_sizes]        

        self.cat_embeddings = nn.ModuleList([nn.Embedding(x, y) 

                                             for x, y in embedding_dims])        

        total_embs_size     = sum([y for x, y in embedding_dims])        

        total_nums_and_embs = total_embs_size + len_numerical      



        self.text_embedding = nn.Embedding(len(token_to_id), EMBEDDING_DIM)



        self.QTitle_net     = QTitle_Net(EMBEDDING_DIM, HIDDEN_SIZE)

        self.QBody_Net      = QA_Net(EMBEDDING_DIM, HIDDEN_SIZE)

        self.answer_Net     = QA_Net(EMBEDDING_DIM, HIDDEN_SIZE)



        total_size          = 2*HIDDEN_SIZE+2*HIDDEN_SIZE+2*HIDDEN_SIZE+total_nums_and_embs



        self.fc1            = nn.Linear(total_size, 300)

        self.dropout1       = nn.Dropout(0.3)

        self.fc2            = nn.Linear(300, 128)

        self.dropout2       = nn.Dropout(0.3)

        self.fc3            = nn.Linear(128, 64)

        self.dropout3       = nn.Dropout(0.3)

        self.out            = nn.Linear(64, out_size)



    def forward(self, categorical, numerical, QTitle_IDs, QBody_IDs, answer_IDs):

        

        batch_size =  categorical.shape[0]



        cat_embs   = [emb_layer(categorical[:, i]) 

                      for i, emb_layer in enumerate(self.cat_embeddings)]        

        cat_embs   = torch.cat(cat_embs, 1)        



        QTitle_emb = self.text_embedding(QTitle_IDs)

        QTitle_out = self.QTitle_net(QTitle_emb)

        

        QBody_IDs  = QBody_IDs.view(QBody_IDs.shape[0] * QBody_IDs.shape[1], 

                                    QBody_IDs.shape[-1])

        QBody_emb  = self.text_embedding(QBody_IDs)

        QBody_out  = self.QBody_Net(QBody_emb,

                                    batch_size)    

        

        answer_IDs = answer_IDs.view(answer_IDs.shape[0]*answer_IDs.shape[1], 

                                     answer_IDs.shape[-1])

        answer_emb = self.text_embedding(answer_IDs)

        answer_out = self.answer_Net(answer_emb,

                                     batch_size)

        

        embeddings = torch.cat([cat_embs, 

                                numerical, 

                                QTitle_out, 

                                QBody_out, 

                                answer_out], 1) 

        

        result     = F.relu(self.fc1(embeddings))

        result     = self.dropout1(result)

        

        result     = F.relu(self.fc2(result))

        result     = self.dropout2(result)

        

        result     = F.relu(self.fc3(result))

        result     = self.dropout3(result)

        

        result     = self.out(result)

        

        return result
def sigmoid(x):                                        

    return 1 / (1 + np.exp(-x))



def mean_spearman_correlation_score(y_true, y_pred):

    return np.mean([spearmanr(y_pred[:, idx], y_true[:, idx]).correlation 

                    for idx in range(len(label_col))])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
epochs           = 100

BATCH_SIZE       = 64

max_patience     = 10

label_path       = '/label/'

loader_n_workers = 0 if device.type=='cpu' else multiprocessing.cpu_count()



TrainSet = QA_Dataset(data       = train,

                      label_path = label_path)



ValSet   = QA_Dataset(data       = val,

                      label_path = label_path)



train_loader = DataLoader(TrainSet, 

                          batch_size = BATCH_SIZE, 

                          shuffle    = True,

                          collate_fn = MyCollator(),

                          num_workers= loader_n_workers,

                          pin_memory = True)



val_loader   = DataLoader(ValSet, 

                          batch_size = BATCH_SIZE, 

                          shuffle    = False,

                          collate_fn = MyCollator(),

                          num_workers= loader_n_workers,

                          pin_memory = True)



model = Main_Model()

model.to(device)



criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



scheduler = ReduceLROnPlateau(optimizer, 

                              mode='max', 

                              factor=0.01, 

                              patience=3, 

                              verbose=True)



Best_val_loss = 0

best_counter = 1

for epochs in tqdm(range(epochs)):

    model.train()

    

    total_loss, counter = 0, 0

    train_preds, train_trues = [], []

    for cat, num, Q_title_IDs, Q_body_IDs, answer_IDs, labels in tqdm(train_loader, 

                                                                      leave=False):



        predictions = model.forward(cat.to(device, non_blocking=True), 

                                    num.to(device, non_blocking=True), 

                                    Q_title_IDs.to(device,non_blocking=True),

                                    Q_body_IDs.to(device, non_blocking=True), 

                                    answer_IDs.to(device, non_blocking=True))

        

        optimizer.zero_grad()

        loss = criterion(predictions, labels.to(device, non_blocking=True))

        total_loss+=loss.item()

        counter+=1

        loss.backward()

        optimizer.step()

        

        train_preds.append(predictions.cpu().detach().numpy())

        train_trues.append(labels.cpu().detach().numpy())

        

        print(f'train - batch loss: {loss} avg BCE loss: {total_loss/counter} ' ,end='\r')

        torch.cuda.empty_cache()    

        

    train_preds = np.concatenate(train_preds)

    train_trues = np.concatenate(train_trues)

    train_loss  = mean_spearman_correlation_score(train_trues, 

                                                  sigmoid(train_preds))

    

    print('epoch number: {}'.format(epochs+1))

    print(f'training   MSCS loss: {train_loss}')

       

        

    y_preds = []

    y_trues = []

    model.eval()

    with torch.no_grad():

        for cat, num, Q_title_IDs, Q_body_IDs, answer_IDs, labels in val_loader:



            predictions = model.forward(cat.to(device, non_blocking=True), 

                                        num.to(device, non_blocking=True), 

                                        Q_title_IDs.to(device, non_blocking=True), 

                                        Q_body_IDs.to(device, non_blocking=True), 

                                        answer_IDs.to(device, non_blocking=True))



            y_preds.append(predictions.cpu().detach().numpy())

            y_trues.append(labels.cpu().detach().numpy())

                

        y_preds = np.concatenate(y_preds)

        y_trues = np.concatenate(y_trues)



        val_loss = mean_spearman_correlation_score(y_trues, 

                                               sigmoid(y_preds))

       

        print('Validation MSCS loss:', val_loss)



        is_best = val_loss > Best_val_loss

        Best_val_loss = max(val_loss, Best_val_loss)

        best_counter+=1

        

        if is_best:

            best_counter = 1

            print('Best validation score so far!')

            torch.save(model.state_dict(), 'Best_Model.pt')

        if best_counter>=max_patience:

            print('last epochs is:',epochs+1)

            break

            

    scheduler.step(val_loss)

    
n_augmentation= 50



testSet       = QA_Dataset(data = test)



test_loader   = DataLoader(testSet, 

                           batch_size = 1, 

                           shuffle    = False,

                           collate_fn = MyCollator(is_test = True ),

                           num_workers=0)

model = Main_Model()

model.load_state_dict(torch.load('Best_Model.pt'))

model.to(device)

model.eval()



sum_pred = np.zeros((len(test),len(label_col)))

with torch.no_grad():

    for i in tqdm(range(n_augmentation)):

        y_preds = []

        for cat, num, Q_title_IDs,Q_body_IDs, answer_IDs in test_loader:



            predictions = model.forward(cat.type(torch.long).to(device), 

                                        num.to(device), 

                                        Q_title_IDs.to(device),

                                        Q_body_IDs.to(device), 

                                        answer_IDs.to(device))



            y_preds.append(predictions.cpu().detach().numpy())

            torch.cuda.empty_cache()



        y_preds = sigmoid(np.concatenate(y_preds))



        sum_pred += y_preds

        

mean_pred = sum_pred / n_augmentation
mean_pred.shape
submission = pd.DataFrame(mean_pred, columns=label_col)

submission['qa_id'] = test['qa_id'].values

submission = submission[['qa_id']+label_col]

submission.to_csv('submission.csv', index=False)

shutil.rmtree('data')

submission.head(3)