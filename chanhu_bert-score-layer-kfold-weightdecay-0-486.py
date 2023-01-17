import numpy as np

import pandas as pd

import spacy

from keras.preprocessing.sequence import pad_sequences

import os

from tqdm import tqdm

import torch

print(os.listdir('../input/bert-score-layer-lb-0-475'))

print(os.listdir('../input/gap-coreference'))
!conda remove -y greenlet

!pip install pytorch-pretrained-bert

!pip install allennlp
from allennlp.modules.span_extractors import EndpointSpanExtractor 

from pytorch_pretrained_bert import BertTokenizer, BertModel

from spacy.lang.en import English



nlp = English()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentencizer = nlp.create_pipe('sentencizer')

nlp.add_pipe(sentencizer)
def candidate_length(candidate):

    #count the word length without space

    count = 0

    for i in range(len(candidate)):

        if candidate[i] !=  " ": count += 1

    return count



def count_char(text, offset):

    count = 0

    for pos in range(offset):

        if text[pos] != " ": count +=1

    return count



def count_token_length_special(token):

    count = 0

    special_token = ["#", " "]

    for i in range(len(token)):

        if token[i] not in special_token: 

            count+=1

    return count



def find_word_index(tokenized_text, char_start, target):

    tar_len = candidate_length(target)

    char_count = 0

    word_index = []

    special_token = ["[CLS]", "[SEP]"]

    for i in range(len(tokenized_text)):

        token = tokenized_text[i]

        if char_count in range(char_start, char_start+tar_len):

            if token in special_token: # for the case like "[SEP]. she"

                continue

            word_index.append(i)

        if token not in special_token:

            token_length = count_token_length_special(token)

            char_count += token_length

    

    if len(word_index) == 1:

        return [word_index[0], word_index[0]] #the output will be start index of span, and end index of span

    else:

        return [word_index[0], word_index[-1]]



def create_tokenizer_input(sents):

    tokenizer_input = str()

    for i, sent in enumerate(sents):

        if i == 0:

            tokenizer_input += "[CLS] "+sent.text+" [SEP] "

        elif i == len(sents) - 1:

            tokenizer_input += sent.text+" [SEP]"

        else:

            tokenizer_input += sent.text+" [SEP] "

            

    return  tokenizer_input



def create_inputs(dataframe):

    

    idxs = dataframe.index

    columns = ['indexed_token', 'offset']

    features_df = pd.DataFrame(index=idxs, columns=columns)

    max_len = 0

    for i in tqdm(range(len(dataframe))):

        text           = dataframe.loc[i, 'Text']

        Pronoun_offset = dataframe.loc[i, 'Pronoun-offset']

        A_offset       = dataframe.loc[i, "A-offset"]

        B_offset       = dataframe.loc[i, "B-offset"]

        Pronoun        = dataframe.loc[i, "Pronoun"]

        A              = dataframe.loc[i, "A"]

        B              = dataframe.loc[i, "B"]

        doc            = nlp(text)

        

        sents = []

        for sent in doc.sents: sents.append(sent)

        token_input = create_tokenizer_input(sents)

        token_input = token_input.replace("#", "*") #Remove special symbols “#” from the original sentence

        tokenized_text = tokenizer.tokenize(token_input) #the token text

        if len(tokenized_text) > max_len: 

            max_len = len(tokenized_text) 

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) #token text to index

        

        A_char_start, B_char_start = count_char(text, A_offset), count_char(text, B_offset)

        Pronoun_char_start         = count_char(text, Pronoun_offset)

        

        word_indexes = [] #

        for char_start, target in zip([A_char_start, B_char_start, Pronoun_char_start], [A, B, Pronoun]):

            word_indexes.append(find_word_index(tokenized_text, char_start, target))#

        features_df.iloc[i] = [indexed_tokens, word_indexes]

        

    print('max length of sentence:', max_len)

    

    return features_df
train_df = pd.read_table('../input/gap-coreference/gap-test.tsv')

test_df  = pd.read_table('../input/gap-coreference/gap-development.tsv')

val_df   = pd.read_table('../input/gap-coreference/gap-validation.tsv')

new_train_df = create_inputs(train_df)

new_test_df  = create_inputs(test_df)

new_val_df   = create_inputs(val_df)
def get_label(dataframe):

    labels = []

    for i in range(len(dataframe)):

        if dataframe.loc[i, 'A-coref']:

            labels.append(0)

        elif dataframe.loc[i, 'B-coref']:

            labels.append(1)

        else:

            labels.append(2)

            

    return labels



new_train_df['label'] = get_label(train_df) # Add label columns

new_val_df['label']   = get_label(val_df)

new_df = pd.concat([new_train_df, new_val_df]) # combine train_df with val_df for the Kfold input 

new_df = new_df.reset_index(drop=True)

new_df.to_csv('train.csv', index=False)

new_test_df['label'] = get_label(test_df)

new_test_df.to_csv('test.csv', index=False)
del new_df

del new_val_df

del new_test_df

del new_train_df
import gc

gc.collect()
from torch.utils.data import Dataset

from torchvision import transforms

from ast import literal_eval

import torch.nn.functional as F



class MyDataset(Dataset):

    

    def __init__(self, dataframe, transform=None):

        self.df = dataframe

        self.transform = transform

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        index_token = self.df.loc[idx, 'indexed_token']

        index_token = literal_eval(index_token) # Change string to list

        index_token = pad_sequences([index_token], maxlen=360, padding='post')[0] #pad 

        

        offset = self.df.loc[idx, 'offset']

        offset = literal_eval(offset)

        offset = np.asarray(offset, dtype='int32')

        label  = int(self.df.loc[idx, 'label'])

        

        distP_A = self.df.loc[idx, 'D_PA']

        distP_B = self.df.loc[idx, 'D_PB']

        

        if self.transform:

            index_token = self.transform(index_token)

            offset = self.transform(offset)

            label = self.transform(label)

        

        return (index_token, offset, distP_A, distP_B), label
class score(torch.nn.Module):

    

    def __init__(self, embed_dim, hidden_dim):

        super(score, self).__init__()

        self.score = torch.nn.Sequential(

                     torch.nn.Linear(embed_dim, hidden_dim),

                     torch.nn.ReLU(inplace=True),

                     torch.nn.Dropout(0.6),

                     torch.nn.Linear(hidden_dim, 1))

        

    def forward(self, x):

        return self.score(x)

    

class mentionpair_score(torch.nn.Module):

    

    def __init__(self, input_dim, hidden_dim):

        super(mentionpair_score, self).__init__()

        self.score = score(input_dim, hidden_dim)

    

    def forward(self, g1, g2, dist_embed):

        

        element_wise = g1 * g2

        pair_score   = self.score(torch.cat((g1, g2, element_wise, dist_embed), dim=-1)) 

        

        return pair_score



class score_model(torch.nn.Module):

    

    def __init__(self):

        super(score_model, self).__init__()

        self.buckets        = [1, 2, 3, 4, 5, 8, 16, 32, 64] 

        self.bert           = BertModel.from_pretrained('bert-base-uncased')

        self.embedding      = torch.nn.Embedding(len(self.buckets)+1, 20)

        self.span_extractor = EndpointSpanExtractor(768, "x,y,x*y")

        self.pair_score     = mentionpair_score(2304*3+20, 150)

        

    def forward(self, sent, offsets, distP_A, distP_B):

        

        bert_output, _   = self.bert(sent, output_all_encoded_layers=False) # (batch_size, max_len, 768)

        #Distance Embeddings

        distPA_embed     = self.embedding(distP_A)

        distPB_embed     = self.embedding(distP_B)

        

        #Span Representation

        span_repres     = self.span_extractor(bert_output, offsets) #(batch, 3, 2304)

        span_repres     = torch.unbind(span_repres, dim=1) #[A: (bath, 2304), B: (bath, 2304), Pronoun:  (bath, 2304)]

        span_norm = []

        for i in range(len(span_repres)): 

            span_norm.append(F.normalize(span_repres[i], p=2, dim=1)) #normalizes the words embeddings

    

        ap_score = self.pair_score(span_norm[2], span_norm[0], distPA_embed)

        bp_score = self.pair_score(span_norm[2], span_norm[1], distPB_embed)

        nan_score = torch.zeros_like(ap_score)

        output = torch.cat((ap_score, bp_score, nan_score), dim=1)

        

        return output
# The Code from https://www.kaggle.com/ceshine/pytorch-bert-endpointspanextractor-kfold



def children(m):

    return m if isinstance(m, (list, tuple)) else list(m.children())



def set_trainable_attr(m, b):

    m.trainable = b

    for p in m.parameters():

        p.requires_grad = b



def apply_leaf(m, f):

    c = children(m)

    if isinstance(m, torch.nn.Module):

        f(m)

    if len(c) > 0:

        for l in c:

            apply_leaf(l, f)



            

def set_trainable(l, b):

    apply_leaf(l, lambda m: set_trainable_attr(m, b))
#the distance features(distance between two word) are binned into the following buckets

#[1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+]

#D_PA is the distance of A and Pronoun

#D_PB is the distance of B and Pronoun

#You can check: https://aclweb.org/anthology/D17-1018



train_dist = pd.read_csv('../input/bert-score-layer-lb-0-475/train_dist_df.csv')

val_dist   = pd.read_csv('../input/bert-score-layer-lb-0-475/val_dist_df.csv')

test_dist  = pd.read_csv('../input/bert-score-layer-lb-0-475/test_dist_df.csv')



train_dist = pd.concat([train_dist, val_dist])

train_dist = train_dist.reset_index(drop=True)

train_dist.head()
from sklearn.model_selection import StratifiedKFold

n_split = 5



train = pd.read_csv('../working/train.csv')

test  = pd.read_csv('../working/test.csv')



train = pd.concat([train, train_dist], axis=1)

test  = pd.concat([test, test_dist], axis=1)

train.head()

Kfold = StratifiedKFold(n_splits=n_split, random_state=2019).split(train, train['label'])
import time



def softmax(x):

    exp_x = np.exp(x)

    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    return y



output = np.zeros((len(test_df), 3))

testset = MyDataset(test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=20) #data loader for test dataset



n_epochs = 30

#Use Kfold to get robusted score

for n_fold, (train_index, val_index) in enumerate(Kfold):

    min_val_loss = 100.0 # for save best model

    PATH = "./best_model_{}.hdf5".format(n_fold+1)

    

    train_df = train.loc[train_index]

    train_df = train_df.reset_index(drop=True)

    val_df   = train.loc[val_index]

    val_df   = val_df.reset_index(drop=True)

    

    trainset = MyDataset(train_df)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)

    valset = MyDataset(val_df)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=True)

    

    model = score_model()

    #freeze bert

    set_trainable(model.bert, False)

    set_trainable(model.embedding, True) 

    set_trainable(model.pair_score, True)

    model.cuda() #

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001) 

    criterion = torch.nn.CrossEntropyLoss().cuda()

    

    print('fold:', n_fold+1)

    for i in range(n_epochs):

        #Start training

        start_time = time.time()

        model.train() 

        avg_loss = 0.

        for idx, (inputs, label) in enumerate(train_loader):

            index_token, offset, distP_A, distP_B = inputs

            index_token = index_token.type(torch.LongTensor).cuda() #change IntTensor to LongTensor,

            offset      = offset.type(torch.LongTensor).cuda()

            label       = label.type(torch.LongTensor).cuda()

            distP_A     = distP_A.type(torch.LongTensor).cuda()

            distP_B     = distP_B.type(torch.LongTensor).cuda()

            

            optimizer.zero_grad()

            output_train = model(index_token, offset, distP_A, distP_B)

            loss = criterion(output_train, label)

            loss.backward()

            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

            

        avg_val_loss = 0.

        #Start test

        model.eval()

        with torch.no_grad():

            for idx, (inputs, label) in enumerate(val_loader):

                index_token, offset, distP_A, distP_B = inputs

                index_token = index_token.type(torch.LongTensor).cuda()

                offset      = offset.type(torch.LongTensor).cuda()

                label       = label.type(torch.LongTensor).cuda()

                distP_A     = distP_A.type(torch.LongTensor).cuda()

                distP_B     = distP_B.type(torch.LongTensor).cuda()

                

                output_test =  model(index_token, offset, distP_A, distP_B)

                avg_val_loss += criterion(output_test, label).item() / len(val_loader)

                

        elapsed_time = time.time() - start_time 

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(

                i + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))

        

        # save best model

        if min_val_loss > avg_val_loss:

            min_val_loss = avg_val_loss 

            torch.save(model.state_dict(), PATH)

        

    

    del model

    

    model = score_model()

    model.load_state_dict(torch.load(PATH)) #load best model to predict

    model.cuda()

    model.eval()

    with torch.no_grad():

        for idx, (inputs, label) in enumerate(test_loader):

            index_token, offset, distP_A, distP_B = inputs

            index_token = index_token.type(torch.LongTensor).cuda()

            offset      = offset.type(torch.LongTensor).cuda()

            label       = label.type(torch.LongTensor).cuda()

            distP_A     = distP_A.type(torch.LongTensor).cuda()

            distP_B     = distP_B.type(torch.LongTensor).cuda()

                

            y_pred = model(index_token, offset, distP_A, distP_B)

            y_pred = softmax(y_pred.cpu().numpy())

            start = idx * 20

            end = start + 20

            output[start:end, :] += y_pred                
import os

output /= 5 

sub_df_path = os.path.join('../input/gendered-pronoun-resolution/', 'sample_submission_stage_1.csv')

sub_df = pd.read_csv(sub_df_path)

sub_df.loc[:, 'A'] = pd.Series(output[:, 0])

sub_df.loc[:, 'B'] = pd.Series(output[:, 1])

sub_df.loc[:, 'NEITHER'] = pd.Series(output[:, 2])



sub_df.head(20)
sub_df.to_csv("submission.csv", index=False)
y_test = pd.read_csv('../working/test.csv')['label']



from sklearn.metrics import log_loss

y_one_hot = np.zeros((2000, 3))

for i in range(len(y_test)):

    y_one_hot[i, y_test[i]] = 1

log_loss(y_one_hot, output)

_output = np.argmax(output, axis=1)

print('acc:', np.asarray(np.where(_output == y_test)).shape[1]/ 2000)