import pandas as pd, numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import time

import re

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

from sklearn.model_selection import StratifiedKFold,KFold

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable

import torch.utils.data

import random
maxlen = 100

max_features = 100000

embed_size = 300

embedding_path = "../input/glove840b300dtxt/glove.840B.300d.txt"

#embedding_path = "../input/glove6b100dtxt/glove.6B.100d.txt"



patience = 5

n_epochs = 50

batch_size = 512

seed=1029
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

subm = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')



train_test_boundary = len(train['id'])

df = pd.concat([train.drop(['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],axis=1),test.drop('id',axis=1)])



label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[label_cols].max(axis=1)

train['multi'] = train[label_cols].sum(axis=1)



df['lowered_comment'] = df['comment_text'].apply(lambda x: x.lower())



contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }



def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text

df['treated_question'] = df['lowered_comment'].apply(lambda x: clean_contractions(x, contraction_mapping))



def clean_text(x):

    x = str(x)

    for punct in "/-'—":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    x = re.sub('[0-9]{5,}', ' ##### ', x)

    x = re.sub('[0-9]{4}', ' #### ', x)

    x = re.sub('[0-9]{3}', ' ### ', x)

    x = re.sub('[0-9]{2}', ' ## ', x)

    return x

df['treated_question'] = df['lowered_comment'].apply(lambda x: clean_text(x))



mispell_dict = {"youfuck":"you fuck","niggors":"niggers","néger":'niger',"fucksex":"fuck sex","yourselfgo":"yourself go","bitchbot":"bitch bot","donkeysex":"donkey sex","mothjer":"mother","niggerjew":"nigger jew","gayyour":"gay your","motherfuckerdie":"motherfucker die","radicalnigger":"radical nigger","philippineslong":"philippines long",'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x

df['treated_comment'] = df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

y_train = train[list_classes].values

list_sentences_train = df["treated_comment"].iloc[:train_test_boundary]

list_sentences_test = df["treated_comment"].iloc[train_test_boundary:]

del df



#max_features = 100000

## Whole vocab 200000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)



X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)



#######################################

### How to tackle with the unknown word



def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore'))

all_embs = np.stack(embedding_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

#emb_mean,emb_std = -0.005838499, 0.48782197



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
train.head(1)
import numpy as np

import torch



class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf



    def __call__(self, val_loss, model):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), 'checkpoint.pt')

        self.val_loss_min = val_loss
class CNN_Text(nn.Module):

    def __init__(self, args):

        super(CNN_Text, self).__init__()

        self.args = args

        

        V = args.embed_num

        D = args.embed_dim

        C = args.class_num

        Ci = 1 # input_channel

        Co = args.kernel_num

        Ks = args.kernel_sizes



        if args.max_norm is not None:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True)

        else:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True)

        

        if args.pre_word_Embedding:

            self.embed.weight.data.copy_(torch.tensor(embedding_matrix))

            # fixed the word embedding

            self.embed.weight.requires_grad = True

        print("dddd {} ".format(self.embed.weight.data.size()))



        if args.wide_conv is True:

            print("using wide convolution")

            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),

                                     padding=(K//2, 0), dilation=1, bias=False) for K in Ks]

        else:

            print("using narrow convolution")

            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]

        print(self.convs1)



        if args.init_weight:

            print("Initing W .......")

            for conv in self.convs1:

                torch.nn.init.xavier_normal_(conv.weight.data, gain=np.sqrt(args.init_weight_value))

                fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)

                print(" in {} out {} ".format(fan_in, fan_out))

                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))

        # for cnn cuda

        if args.cuda is True:

            for conv in self.convs1:

                conv = conv.cuda()



        self.dropout = nn.Dropout(args.dropout)

        self.dropout_embed = nn.Dropout(args.dropout_embed)

        

        in_fea = len(Ks) * Co

        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)

        # whether to use batch normalizations

        if args.batch_normalizations is True:

            print("using batch_normalizations in the model......")

            self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea//2, bias=True)

            self.fc2 = nn.Linear(in_features=in_fea//2, out_features=C, bias=True)

            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum,

                                            affine=args.batch_norm_affine)

            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea//2, momentum=args.bath_norm_momentum,

                                         affine=args.batch_norm_affine)

            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum,

                                         affine=args.batch_norm_affine)



    def calculate_fan_in_and_fan_out(tensor):

        dimensions = tensor.ndimension()

        if dimensions < 2:

            raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")



        if dimensions == 2:  # Linear

            fan_in = tensor.size(1)

            fan_out = tensor.size(0)

        else:

            num_input_fmaps = tensor.size(1)

            num_output_fmaps = tensor.size(0)

            receptive_field_size = 1

            if tensor.dim() > 2:

                receptive_field_size = tensor[0][0].numel()

            fan_in = num_input_fmaps * receptive_field_size

            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out



    def forward(self, x):

        x = self.embed(x)  # (N,W,D)

        x = self.dropout_embed(x)

        x = x.unsqueeze(1)  # (N,Ci,W,D)

        if self.args.batch_normalizations is True:

            x = [self.convs1_bn(torch.tanh(conv(x))).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)

            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)

        else:

            x = [F.elu(conv(x)).squeeze(3) for conv in self.convs1]  #[(N,Co,W), ...]*len(Ks)

            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N,len(Ks)*Co)

        if self.args.batch_normalizations is True:

            x = self.fc1_bn(self.fc1(x))

            logit = self.fc2_bn(self.fc2(torch.tanh(x)))

        else:

            logit = self.fc(x)

        return logit



def sigmoid(x):

    return 1 / (1 + np.exp(-x))
class RCNN_Text(nn.Module):

    def __init__(self, args):

        super(RCNN_Text, self).__init__()

        self.args = args

        

        self.output_size = args.output_size

        self.hidden_size = args.hidden_size

        self.vocab_size = args.vocab_size

        self.embedding_size = args.embedding_size

        

        if args.max_norm is not None:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(self.vocab_size, self.embedding_size, max_norm=5, scale_grad_by_freq=True)

        else:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(self.vocab_size, self.embedding_size, scale_grad_by_freq=True)

        

        if args.pre_word_Embedding:

            self.embed.weight.data.copy_(torch.tensor(embedding_matrix))

            # fixed the word embedding

            self.embed.weight.requires_grad = True

        print("dddd {} ".format(self.embed.weight.data.size()))

        

        self.dropout = nn.Dropout(args.dropout)

        self.dropout_embed = nn.Dropout(args.dropout_embed)

        

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=True)

        self.W2 = nn.Linear(2 * self.hidden_size + self.embedding_size, self.hidden_size)

        self.label = nn.Linear(self.hidden_size, self.output_size)

        

    def forward(self,x):

            

        x = self.embed(x)  # (N,W,D)

        x = self.dropout_embed(x)

        batch_size = x.size(0)

           

        x = x.permute(1, 0, 2) # x.size() = (num_sequences, batch_size, embedding_size)

        h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

        c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())



        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))

        final_encoding = torch.cat((output, x), 2).permute(1, 0, 2)

        y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_size)

        y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)

        y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)

        y = y.squeeze(2)

        logits = self.label(y)

        return logits

### Ref:https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/selfAttention.py

class RNN_selfatten(nn.Module):

    def __init__(self,args):

        super(RNN_selfatten, self).__init__()

        

        self.args = args

        self.output_size = args.output_size

        self.hidden_size = args.hidden_size

        self.vocab_size = args.vocab_size

        self.embedding_size = args.embedding_size

        

        # We will use da = 350, r = 30 & penalization_coeff = 1

        # as per given in the self-attention original ICLR paper

        da = 350

        r = 30

        

        self.W_s1 = nn.Linear(2*self.hidden_size, da)

        self.W_s2 = nn.Linear(da, r)

        self.fc_layer = nn.Linear(r*2*self.hidden_size, 2000)

        self.label = nn.Linear(2000, self.output_size)

        

        if args.max_norm is not None:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(self.vocab_size, self.embedding_size, max_norm=5, scale_grad_by_freq=True)

        else:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(self.vocab_size, self.embedding_size, scale_grad_by_freq=True)

        

        if args.pre_word_Embedding:

            self.embed.weight.data.copy_(torch.tensor(embedding_matrix))

            # fixed the word embedding

            self.embed.weight.requires_grad = True

        print("dddd {} ".format(self.embed.weight.data.size()))

        

        self.dropout = nn.Dropout(args.dropout)

        self.dropout_embed = nn.Dropout(args.dropout_embed)

        self.bilstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=True)

        



    def attention_net(self, lstm_output):

        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))

        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    

    def forward(self,x):

        x = self.embed(x)  # (N,W,D)

        x = self.dropout_embed(x)

        batch_size = x.size(0)

        

        x = x.permute(1, 0, 2) # x.size() = (num_sequences, batch_size, embedding_size)

        h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

        c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())



        output, (h_n, c_n) = self.bilstm(x, (h_0, c_0))

        output = output.permute(1, 0, 2)  

        # output.size() = (batch_size, num_seq, 2*hidden_size)

        # h_n.size() = (1, batch_size, hidden_size)

        # c_n.size() = (1, batch_size, hidden_size)

        attn_weight_matrix = self.attention_net(output)

        # attn_weight_matrix.size() = (batch_size, r, num_seq)

        # output.size() = (batch_size, num_seq, 2*hidden_size)

        hidden_matrix = torch.bmm(attn_weight_matrix, output)

        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)

        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.

        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))

        logits = self.label(fc_out)

        # logits.size() = (batch_size, output_size)



        return logits
### soft attention

class RNN_atten(nn.Module):

    def __init__(self,args):

        super(RNN_atten, self).__init__()

        

        self.args = args

        self.output_size = args.output_size

        self.hidden_size = args.hidden_size

        self.vocab_size = args.vocab_size

        self.embedding_size = args.embedding_size

        

        if args.max_norm is not None:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(self.vocab_size, self.embedding_size, max_norm=5, scale_grad_by_freq=True)

        else:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(self.vocab_size, self.embedding_size, scale_grad_by_freq=True)

        

        if args.pre_word_Embedding:

            self.embed.weight.data.copy_(torch.tensor(embedding_matrix))

            # fixed the word embedding

            self.embed.weight.requires_grad = True

        print("dddd {} ".format(self.embed.weight.data.size()))

        

        self.dropout = nn.Dropout(args.dropout)

        self.dropout_embed = nn.Dropout(args.dropout_embed)

        

        self.label = nn.Linear(self.hidden_size, self.output_size)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=False)



    def attention_net(self, lstm_output, final_state):

        """  

        Tensor Size :

                    final_state.size() = (1,batch_size, hidden_size)

                    hidden.size() = (batch_size, hidden_size)

                    attn_weights.size() = (batch_size, num_seq)

                    soft_attn_weights.size() = (batch_size, num_seq)

                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)

        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)

        soft_attn_weights = F.softmax(attn_weights, 1)

        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    

    def forward(self,x):

        x = self.embed(x)  # (N,W,D)

        x = self.dropout_embed(x)

        batch_size = x.size(0)

        

        x = x.permute(1, 0, 2) # x.size() = (num_sequences, batch_size, embedding_size)

        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 

        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)

        

        attn_output = self.attention_net(output, final_hidden_state)

        logits = self.label(attn_output)

        

        return logits
# Ref:https://github.com/dreamgonfly/deep-text-classification-pytorch/blob/master/dictionaries.py

class CharCNNDictionary:

    def __init__(self):

        self.ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}" + '\n'

        self.PAD_TOKEN = '<PAD>'

        

    def build_dictionary(self):

        self.vocab_chars = list(self.ALPHABET) + [self.PAD_TOKEN]

        self.char2idx = {char:idx for idx, char in enumerate(self.vocab_chars)}

        self.vocabulary_size = len(self.vocab_chars)

        self._build_weight()

    

    # One under slash "_" means half private which means using that inside the class 

    def _build_weight(self):

        # one hot embedding plus all-zero vector

        onehot_matrix = np.eye(self.vocabulary_size, self.vocabulary_size - 1)

        self.embedding = onehot_matrix



    def indexer(self, char):

        try:

            return self.char2idx[char]

        except:

            char = self.PAD_TOKEN

            return self.char2idx[char]



def pad_text(text, pad, min_length=None, max_length=None):

    length = len(text)

    if min_length is not None and length < min_length:

        return text + [pad]*(min_length - length)

    if max_length is not None and length > max_length:

        return text[:max_length]

    return text



class TextDataset(Dataset):

    

    def __init__(self, texts, dictionary, sort=False, min_length=None, max_length=None):



        PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)

        

        self.texts = [[dictionary.indexer(token) for token in text]

                          for text in texts]



        if min_length or max_length:

            self.texts = [pad_text(text, PAD_IDX, min_length, max_length) 

                          for text in self.texts]



        if sort:

            self.texts = sorted(self.texts, key=lambda x: len(x[0]))

    

    def get_all(self):

        return self.texts

        

    def __getitem__(self, index):

        tokens = self.texts[index]

        return tokens

        

    def __len__(self):

        return len(self.texts)
### char CNN model

class char_CNN(nn.Module):

    def __init__(self,args):

        super(char_CNN, self).__init__()

        self.args = args

        

        vocabulary_size = args.vocabulary_size

        embed_size = vocabulary_size - 1 # except for padding

        embedding_weight = args.embedding

        

        n_classes = args.output_size

        max_length = args.max_length

        

        if args.mode == 'large':

            conv_features = 1024

            linear_features = 2048

        elif args.mode == 'small':

            conv_features = 256

            linear_features = 1024

        else:

            raise NotImplementedError()

        

        # quantization

        self.embedding = nn.Embedding(vocabulary_size, embed_size)

        if embedding_weight is not None:

            self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_weight), requires_grad=False)

        

        conv1 = nn.Sequential(

            nn.Conv1d(in_channels=embed_size, out_channels=conv_features, kernel_size=7),

            nn.MaxPool1d(kernel_size=3),

            nn.ReLU()

        )

        conv2 = nn.Sequential(

            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=7),

            nn.MaxPool1d(kernel_size=3),

            nn.ReLU()

        )

        conv3 = nn.Sequential(

            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=3),

            nn.ReLU()

        )

        conv4 = nn.Sequential(

            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=3),

            nn.ReLU()

        )

        conv5 = nn.Sequential(

            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=3),

            nn.ReLU()

        )

        conv6 = nn.Sequential(

            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=3),

            nn.MaxPool1d(kernel_size=3),

            nn.ReLU()

        )

         

        # (max_length - 96) // 27 is the output size of conv6 , before feed it to the fc layer, there is a "view" opt.

        initial_linear_size = (max_length - 96) // 27 * conv_features

        

        linear1 = nn.Sequential(

            nn.Linear(initial_linear_size, linear_features),

            nn.Dropout(),

            nn.ReLU()

        )

        linear2 = nn.Sequential(

            nn.Linear(linear_features, linear_features),

            nn.Dropout(),

            nn.ReLU()

        )

        linear3 = nn.Linear(linear_features, n_classes)

        

        self.convolution_layers = nn.Sequential(conv1, conv2, conv3, conv4, conv5, conv6)

        self.linear_layers = nn.Sequential(linear1, linear2, linear3)

        

        

        self.dropout = nn.Dropout(args.dropout)

        self.dropout_embed = nn.Dropout(args.dropout_embed)

        

    def forward(self, sentences):

#         print(sentences.shape)

        x = self.embedding(sentences)

#         print(x.shape)

        x = x.transpose(1,2)

#         print(x.shape)

        x = self.convolution_layers(x)

#         print(x.shape)

        x = x.view(x.size(0), -1)

#         print(x.shape)

        x = self.linear_layers(x)

#         print(x.shape)

        return x
class C_LSTM(nn.Module):

    def __init__(self, args):

        super(C_LSTM, self).__init__()

        self.args = args

        

        V = args.embed_num

        D = args.embed_dim

        C = args.class_num

        Ci = 1 # input_channel

        Co = args.kernel_num

        Ks = args.kernel_sizes

        

        self.hidden_size = args.hidden_size



        if args.max_norm is not None:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True)

        else:

            print("max_norm = {} ".format(args.max_norm))

            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True)

        

        if args.pre_word_Embedding:

            self.embed.weight.data.copy_(torch.tensor(embedding_matrix))

            # fixed the word embedding

            self.embed.weight.requires_grad = True

        print("dddd {} ".format(self.embed.weight.data.size()))



        if args.wide_conv is True:

            print("using wide convolution")

            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),

                                     padding=(K//2, 0), dilation=1, bias=False) for K in Ks]

        else:

            print("using narrow convolution")

            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]

        print(self.convs1)



        if args.init_weight:

            print("Initing W .......")

            for conv in self.convs1:

                torch.nn.init.xavier_normal_(conv.weight.data, gain=np.sqrt(args.init_weight_value))

                fan_in, fan_out = C_LSTM.calculate_fan_in_and_fan_out(conv.weight.data)

                print(" in {} out {} ".format(fan_in, fan_out))

                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))

        # for cnn cuda

        

        if args.cuda is True:

            for conv in self.convs1:

                conv = conv.cuda()

        

        self.dropout = nn.Dropout(args.dropout)

        self.dropout_embed = nn.Dropout(args.dropout_embed)

        

        self.lstm = nn.LSTM(Co, self.hidden_size, bidirectional=True)

        

        in_fea = 2 * self.hidden_size

        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)

        # whether to use batch normalizations

        if args.batch_normalizations:

            print("using batch_normalizations in the model......")

            self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea//2, bias=True)

            self.fc2 = nn.Linear(in_features=in_fea//2, out_features=C, bias=True)

            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum,

                                            affine=args.batch_norm_affine)

            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea//2, momentum=args.bath_norm_momentum,

                                         affine=args.batch_norm_affine)

            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum,

                                         affine=args.batch_norm_affine)



    def calculate_fan_in_and_fan_out(tensor):

        dimensions = tensor.ndimension()

        if dimensions < 2:

            raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")



        if dimensions == 2:  # Linear

            fan_in = tensor.size(1)

            fan_out = tensor.size(0)

        else:

            num_input_fmaps = tensor.size(1)

            num_output_fmaps = tensor.size(0)

            receptive_field_size = 1

            if tensor.dim() > 2:

                receptive_field_size = tensor[0][0].numel()

            fan_in = num_input_fmaps * receptive_field_size

            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out



    def forward(self, x):

        x = self.embed(x)  # (N,W,D)

        x = self.dropout_embed(x)

        x = x.unsqueeze(1)  # (N,Ci,W,D)

        

        batch_size = x.size(0)

        

        if self.args.batch_normalizations:

            x = [self.convs1_bn(torch.tanh(conv(x))).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)

            x = [i.permute(2, 0,1) for i in x]

        else:

            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)

            x = [i.permute(2, 0,1) for i in x] #[(N,Co), ...]*len(Ks)

            

        x = x[0]

        

        h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

        c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))

        

        x = torch.cat([final_hidden_state[i] for i in range(final_hidden_state.size(0))],1)

        if self.args.batch_normalizations:

            x = self.fc1_bn(self.fc1(x))

            logit = self.fc2_bn(self.fc2(torch.tanh(x)))

        else:

            logit = self.fc(x)

        return logit

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.parameter import Parameter



import copy

import math



def gelu(x):

    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))





def swish(x):

    return x * torch.sigmoid(x)





ACT_FNS = {

    'relu': nn.ReLU,

    'swish': swish,

    'gelu': gelu

}



class LayerNorm(nn.Module):

    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."



    def __init__(self, n_state, e=1e-5):

        super(LayerNorm, self).__init__()

        self.g = nn.Parameter(torch.ones(n_state))

        self.b = nn.Parameter(torch.zeros(n_state))

        self.e = e



    # x:(batch*emb)

    def forward(self, x):

        u = x.mean(-1, keepdim=True)

        s = (x - u).pow(2).mean(-1, keepdim=True)

        x = (x - u) / torch.sqrt(s + self.e)

        return self.g * x + self.b





class Conv1D(nn.Module):

    def __init__(self, nf, rf, nx):

        super(Conv1D, self).__init__()

        self.rf = rf

        self.nf = nf

        if rf == 1:  # faster 1x1 conv

            w = torch.empty(nx, nf)

            nn.init.normal_(w, std=0.02)

            self.w = Parameter(w)

            self.b = Parameter(torch.zeros(nf))

        else:  # was used to train LM

            raise NotImplementedError



    def forward(self, x):

        if self.rf == 1:

            size_out = x.size()[:-1] + (self.nf,)

            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)

            x = x.view(*size_out)

        else:

            raise NotImplementedError

        return x



class Attention(nn.Module):

    def __init__(self, nx, n_ctx, cfg, scale=False):

        super(Attention, self).__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)

        # [switch nx => n_state from Block to Attention to keep identical to TF implem]

        assert n_state % cfg.n_head == 0

        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

        self.n_head = cfg.n_head

        self.split_size = n_state

        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, 1, nx)

        self.c_proj = Conv1D(n_state, 1, nx)

        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)

        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)



    def _attn(self, q, k, v):

        w = torch.matmul(q, k)

        if self.scale:

            w = w / math.sqrt(v.size(-1))

        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights

        # XD: self.b may be larger than w, so we need to crop it

        b = self.b[:, :, :w.size(-2), :w.size(-1)]

        w = w * b + -1e9 * (1 - b)



        w = nn.Softmax(dim=-1)(w)

        w = self.attn_dropout(w)

        return torch.matmul(w, v)



    def merge_heads(self, x):

        x = x.permute(0, 2, 1, 3).contiguous()

        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)

        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

        

    def split_heads(self, x, k=False):

        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)

        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states

        if k:

            return x.permute(0, 2, 3, 1)

        else:

            return x.permute(0, 2, 1, 3)



    def forward(self, x):

        # x : batch * seq * embed

        x = self.c_attn(x)

        query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)

        key = self.split_heads(key, k=True)

        value = self.split_heads(value)

        a = self._attn(query, key, value)

        a = self.merge_heads(a)

        a = self.c_proj(a)

        a = self.resid_dropout(a)

        return a



class MLP(nn.Module):

    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)

        super(MLP, self).__init__()

        nx = cfg.n_embd

        self.c_fc = Conv1D(n_state, 1, nx)

        self.c_proj = Conv1D(nx, 1, n_state)

        self.act = ACT_FNS[cfg.ACT_FNS]

        self.dropout = nn.Dropout(cfg.resid_pdrop)



    def forward(self, x):

        h = self.act(self.c_fc(x))

        h2 = self.c_proj(h)

        return self.dropout(h2)



class Block(nn.Module):

    def __init__(self, n_ctx, cfg, scale=False):

        super(Block, self).__init__()

        nx = cfg.n_embd

        self.attn = Attention(nx, n_ctx, cfg, scale)

        self.ln_1 = LayerNorm(nx)

        self.mlp = MLP(4 * nx, cfg)

        self.ln_2 = LayerNorm(nx)



    def forward(self, x):

        a = self.attn(x)

        n = self.ln_1(x + a)

        m = self.mlp(n)

        h = self.ln_2(n + m)

        return h





class PositionalEncoding(nn.Module):

    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):

        super(PositionalEncoding, self).__init__()

        

        # Compute the positional encodings once in log space.

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *

                             -(math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))

        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

        

    def forward(self, x):

        x = x + Variable(self.pe[:, :x.size(1)], 

                         requires_grad=False)

        return x



class TransformerModel(nn.Module):

    """ Transformer model """

    def __init__(self, cfg):

        super(TransformerModel, self).__init__()

        self.vocab = cfg.vocab

        self.n_ctx = cfg.n_ctx

        self.embed = nn.Embedding(self.vocab+1, cfg.n_embd)

        self.PE = PositionalEncoding(cfg.n_embd,self.n_ctx)

        nn.init.normal_(self.embed.weight, std=0.02)

        if cfg.pre_word_Embedding:

            self.embed.weight.data.copy_(torch.tensor(cfg.embedding_matrix))

            # fixed the word embedding

            self.embed.weight.requires_grad = True

        

        self.drop = nn.Dropout(cfg.embd_pdrop)

        block = Block(self.n_ctx, cfg, scale=True)

        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])

        

        self.fc = nn.Linear(in_features=cfg.n_embd, out_features=cfg.C, bias=True)

        



    def forward(self, x):

        #x = x.view(-1, x.size(-2), x.size(-1))

        #print ('after:',x.shape)

        x = self.embed(x)

        x = self.PE(x)

        h = self.drop(x)

        # Add the position information to the input embeddings

        for block in self.h:

            h = block(h)

        logit = self.fc(h[:,-1,:])

        return logit
target = torch.tensor([[0,1,1]], dtype=torch.float32)  # 3 classes, batch size = 2

output = torch.tensor([[0.1,0.7,0.8]], dtype=torch.float32)  # A prediction (logit)

#pos_weight = torch.ones([64])  # All weights are equal to 1

criterion = torch.nn.BCEWithLogitsLoss()#pos_weight=pos_weight)

criterion(output, target)  # -log(sigmoid(0.999))
m=nn.Sigmoid()

m(output)
def sigmoid(x):

    return 1 / (1+np.exp(-x))

-np.log((1-sigmoid(0.1))*(sigmoid(0.7))*(sigmoid(0.8)))/3
class FocalLoss(nn.Module):

    def __init__(self,pos_weight,alpha=1, gamma=2,logits=True, reduce=True):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.logits = logits

        self.reduce = reduce

        

        self.BCELossLogits = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=pos_weight)

        self.BCELoss = torch.nn.BCELoss(reduction='none' )



    def forward(self, inputs, targets):

        if self.logits:

            BCE_loss = self.BCELossLogits(inputs, targets)

        else:

            BCE_loss = self.BCELoss(inputs, targets)

    

        pt = torch.exp(-BCE_loss)

        F_loss = (1-pt)**self.gamma * BCE_loss



        if self.reduce:

            return torch.mean(F_loss)

        else:

            return F_loss





class FocalLoss_new(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):

        super(FocalLoss_new, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.logits = logits

        self.reduce = reduce



    def forward(self, inputs, targets):

        if self.logits:

            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=None)

        else:

            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=None)

        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss



        if self.reduce:

            return torch.mean(F_loss)

        else:

            return F_loss

 
def train_model(model, x_train, y_train, x_val, y_val, validate=True):

    #optimizer = torch.optim.Adam(model.parameters())

    optimizer = torch.optim.RMSprop(model.parameters(),lr=0.001)

    #optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

    #optimizer = torch.optim.Adadelta(model.parameters())

    # scheduler = CosineAnnealingLR(optimizer, T_max=5)

    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    

    train = torch.utils.data.TensorDataset(x_train, y_train)

    valid = torch.utils.data.TensorDataset(x_val, y_val)

    

    train_loss,valid_loss,valid_auc = [],[],[]

    

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    

    pos_label = y_train.cpu().sum(0)

    #whole = y_train.size(0)*torch.ones(pos_label.shape)

    #pos_weight = (whole-pos_label) / pos_label

    #loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos_weight).cuda()

    #loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()

    

    pos_weight = 0.25 * torch.ones(pos_label.shape)

    

    loss_fn = FocalLoss(pos_weight=pos_weight).cuda()

    

    best_score = -np.inf

    

    for epoch in range(n_epochs):

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        start_time = time.time()

        

        avg_loss = 0.

        

        model.train()

        for x_batch, y_batch in tqdm(train_loader, disable=True):

            y_pred = model(x_batch)

            y_batch = y_batch.squeeze(dim=1)

            loss = loss_fn(y_pred, y_batch)

            '''

            lambda_2 = torch.tensor(0.1).cuda()

            l2_reg = torch.tensor(0.0).cuda()

            for name, param in model.named_parameters():

                #print (name)

                if name == "fc1.weight":

                    l2_reg += torch.norm(param).cuda()

            loss += lambda_2 * l2_reg

            '''

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

         

        model.eval()

        valid_preds=None

        valid_labels = None

        

        if validate:

            avg_val_loss = 0.

            for i, (x_batch, y_batch) in enumerate(valid_loader):

                y_pred = model(x_batch).detach()

                y_batch = y_batch.squeeze(dim=1)

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

                #valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

                if valid_preds is None:

                    valid_preds = sigmoid(y_pred.cpu().numpy())

                    valid_labels = y_batch.cpu().numpy()

                else:

                    valid_preds = np.concatenate((valid_preds,sigmoid(y_pred.cpu().numpy())),axis=0)

                    valid_labels = np.concatenate((valid_labels,y_batch.cpu().numpy()),axis=0)

                    

            #search_result = threshold_search(y_val.cpu().numpy(), valid_preds)

            #val_f1, val_threshold = search_result['f1'], search_result['threshold']

            

            avg_auc = roc_auc_score(valid_labels,valid_preds)

            elapsed_time = time.time() - start_time

            

            train_loss.append(avg_loss)

            valid_loss.append(avg_val_loss)

            valid_auc.append(avg_auc)

            

            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_auc={:.4f} time={:.2f}s'.format(

                epoch + 1, n_epochs, avg_loss, avg_val_loss, avg_auc, elapsed_time))    

            

            early_stopping(avg_val_loss, model)

            if early_stopping.early_stop:

                print("Early stopping")

                break

                

        else:

            train_loss.append(avg_loss)

            

            elapsed_time = time.time() - start_time

            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(

                epoch + 1, n_epochs, avg_loss, elapsed_time))

    

    '''

    valid_preds = np.zeros((x_val_fold.size(0)))

    avg_val_loss = 0.

    for i, (x_batch, y_batch) in enumerate(valid_loader):

        y_pred = model(x_batch).detach()

        y_batch = y_batch.squeeze(dim=1)

        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

        valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    print('Validation loss: ', avg_val_loss)

    '''

    

    model.load_state_dict(torch.load('checkpoint.pt'))

    test_preds = np.zeros((len(test_loader.dataset),len(label_cols)))

    for i, (x_batch,) in enumerate(test_loader):

        y_pred = model(x_batch).detach()

        test_preds[i * batch_size:(i+1) * batch_size,:] = sigmoid(y_pred.cpu().numpy())

    

    return valid_preds, test_preds,train_loss,valid_loss,valid_auc#, test_preds_local
x_test_cuda = torch.tensor(X_test, dtype=torch.long).cuda()

test = torch.utils.data.TensorDataset(x_test_cuda)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)



import os



def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result



def seed_everything(seed=1234):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()



class Args:

    #modelType = "charCNN"

    #modelType = "CLSTM"

    #modelType = "TextCNN"

    #modelType = "RCNN_Text"

    #modelType = "RNN_selfAtten"

    modelType = "RNN_atten"

    max_norm = None

    embed_num = max_features+1

    embed_dim = embed_size

    pre_word_Embedding = True

    init_weight = True

    init_weight_value  = 1 ##  gain = sqrt(init_weight_value) 

    cuda = True

    class_num = 6

    dropout = 0.1

    dropout_embed = 0.6

    

    ### CNN_text Param

    kernel_num = 32 #400 

    kernel_sizes = [1,2,3,5]#[3,4,5]

    wide_conv = False

    

    batch_normalizations = False

    bath_norm_momentum = 0.1

    batch_norm_affine = False

    

    ### RCNN_text/ RNN_selfatten Param

    output_size = class_num

    hidden_size = 100

    vocab_size = embed_num

    embedding_size = embed_dim

    

    ### char_CNN

    if modelType == "charCNN":

        charDict = CharCNNDictionary()

        charDict.build_dictionary()

        charDict

        vocabulary_size = charDict.vocabulary_size

        embedding = charDict.embedding

        mode = "small"

        max_length = 300 # length of char

    

    ### CLSTM

    if modelType == "CLSTM":

        kernel_num = 400

        kernel_sizes = [3]

        wide_conv = True

    

args=Args()
class Config():

    vocab = max_features

    n_ctx = maxlen

    n_head = 6

    n_embd = embed_size

    ACT_FNS = 'gelu'

    n_layer = 1

    C = 6 # class_num

    

    pre_word_Embedding = True

    embedding_matrix = embedding_matrix

    

    

    embd_pdrop = 0.5

    attn_pdrop = 0.5

    resid_pdrop = 0.5

    

cfg = Config()
if args.modelType == "charCNN":

    x_train = list_sentences_train.map(lambda x:x[:args.max_length])

    x_test = list_sentences_test.map(lambda x:x[:args.max_length])

    

    X_train = TextDataset(x_train,args.charDict,False,args.max_length,args.max_length).get_all()

    X_test = TextDataset(x_test,args.charDict,False,args.max_length,args.max_length).get_all()

    

    X_train = np.asarray(X_train)

    X_test = np.asarray(X_test)
splits = list(KFold(n_splits=5, shuffle=True, random_state=10).split(X_train, y_train))

train_preds = np.zeros(len(X_train))

test_preds = np.zeros((len(X_test), len(splits)))

from tqdm import tqdm

from sklearn.metrics import f1_score

from functools import reduce

 

for size in [1]:

    print ("The kernel size is {}".format(size))

    tmp = []

    for i, (train_idx, valid_idx) in enumerate(splits):    

        x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()

        y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()

        x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()

        y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()



        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)



        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)



        print(f'Fold {i + 1}')



        seed_everything(seed + i)

        #model = CNN_Text(args)

        #model = RCNN_Text(args)

        #model = RNN_selfatten(args)

        model = RNN_atten(args)

        ##model = char_CNN(args)

        ##model = C_LSTM(args)

        #model = TransformerModel(cfg)

        model.cuda()



        valid_preds_fold, test_preds_fold,train_loss,valid_loss,valid_auc = train_model(model,

                                                        x_train_fold, 

                                                        y_train_fold, 

                                                        x_val_fold, 

                                                        y_val_fold, validate=True)

        tmp.append(test_preds_fold/len(splits))

    

    test_preds_fold = reduce(lambda x,y:x+y,tmp)

        

    submid = pd.DataFrame({'id': subm["id"]})

    submission = pd.concat([submid, pd.DataFrame(test_preds_fold, columns = label_cols)], axis=1)

    submission.to_csv('submission'+str(size)+".csv", index=False)
# import graph objects as "go"

import plotly.offline as ply

import plotly.graph_objs as go

from plotly.tools import make_subplots

from plotly.plotly import iplot

ply.init_notebook_mode(connected=True)



import plotly.plotly as py

py.sign_in('redinton', 'jQfx5zOJGz7GNQa50ISx')



t = np.linspace(0,len(train_loss),len(train_loss))

# Creating trace1

trace1 = go.Scatter(

                    x = t,

                    y = train_loss,

                    mode = "lines",

                    name = "train_loss",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))

                    #text= df.university_name)

# Creating trace2

trace2 = go.Scatter(

                    x = t,

                    y = valid_loss,

                    mode = "lines+markers",

                    name = "valid_loss",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'))#,

                    #text= df.university_name)

data = [trace1, trace2]

layout = dict(title = 'The curve of training loss and validing loss',

              xaxis= dict(title= 'Epoch',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
import matplotlib.pyplot as plt

t = np.linspace(0,len(train_loss),len(train_loss))



plt.plot(t,train_loss,label='train_loss')

plt.plot(t,valid_loss,label='valid_loss')

plt.legend(loc='upper right')

plt.show()

pd.DataFrame(valid_auc).plot()