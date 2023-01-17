import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import random

import re

from nltk import sent_tokenize

import nltk

from tqdm import tqdm

from albumentations.core.transforms_interface import DualTransform, BasicTransform

import gensim.downloader as api

from transformers import BertTokenizer,TFBertForMaskedLM

import re

!pip install nlpaug

!pip install -q colored



import nlpaug.augmenter.word as naw

import nlpaug.model.word_stats as nmw

import albumentations

from torch.utils.data import Dataset



import torch

import torch.nn as nn

from torch.optim import Adam

from torch.optim.lr_scheduler import ReduceLROnPlateau



from torch.multiprocessing import Pipe, Process

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler



from tqdm.notebook import tqdm

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score

from transformers import BertModel, BertTokenizer

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

import gc



from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences as pad

import time



import time

import colored

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from colored import fg, bg, attr

class NLPTransform(BasicTransform):

    """ Transform for nlp task."""



    @property

    def targets(self):

        return {"data": self.apply}

    

    def update_params(self, params, **kwargs):

        if hasattr(self, "interpolation"):

            params["interpolation"] = self.interpolation

        if hasattr(self, "fill_value"):

            params["fill_value"] = self.fill_value

        return params



    def get_sentences(self, text, lang='en'):

        return sent_tokenize(text)
class WordEmbeddingSubstitution(NLPTransform):

    """ susbtitute similar words """

    def __init__(self, always_apply=False, p=0.5):

        self.model=api.load('glove-twitter-25')  

        

        super(WordEmbeddingSubstitution, self).__init__(always_apply, p)



    def apply(self, data, **params):

        text=''

        for word in nltk.word_tokenize(data):

            try:

                simword=self.model.most_similar(word,topn=1)

                if simword[0][1]>0.95:

                    text=text+' '+simword[0][0]

                    continue

            except:

                text=text+' '+word

                continue



            text=text+' '+word



        return text
#Uncomment and see the below trick



# transform = WordEmbeddingSubstitution(p=1.0)



# text = 'This is super cool and amazing'



# transform(data=(text))['data']
class LMmask(NLPTransform):

    """ susbtitute similar words """

    def __init__(self, always_apply=False, p=0.5,verbose=False):

        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

        self.model= TFBertForMaskedLM.from_pretrained('bert-base-uncased') 

        self.probability=p

        self.verbose=verbose

        

        super(LMmask, self).__init__(always_apply, p)



    def apply(self, data, **params):

       

        flag=True

        text=data

        

        for ix,n in enumerate(text.split()):

            if random.random() < self.probability and flag:

                flag=False

                text=text.replace(n,'[MASK]')

                break

        

        # DEFINE SENTENCE

        indices = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='tf')



        # PREDICT MISSING WORDS

        pred = self.model(indices)

        masked_indices = np.where(indices==103)[1]



        # DISPLAY MISSING WORDS

        predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

        

        if self.verbose:

            return text,self.tokenizer.decode(predicted_words)

        

        return self.tokenizer.decode(predicted_words)

#Uncomment and see the below trick



# transform = LMmask(p=1,verbose=True)



# text = 'Data science is important'





# transform(data=text)['data']
def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):

    token_pattern = re.compile(token_pattern)

    return token_pattern.findall(text)



# Load sample data

train=pd.read_csv('../input/nlp-getting-started/train.csv')

train_x=train['text'].values.tolist()



# Tokenize input

train_x_tokens = [_tokenizer(x) for x in train_x]



# Train TF-IDF model

tfidf_model = nmw.TfIdf()

tfidf_model.train(train_x_tokens)

tfidf_model.save('.')



# Load TF-IDF augmenter

aug = naw.TfIdfAug(model_path='.', tokenizer=_tokenizer)



texts = [

    'I was wondering if anyone out there could enlighten me',

    'well folks, my mac plus finally gave up'

]



for text in texts:

    augmented_text = aug.augment(text)

    

    print('-'*20)

    print('Original Input:{}'.format(text))

    print('Agumented Output:{}'.format(augmented_text))

class ShuffleSentencesTransform(NLPTransform):

    """ Do shuffle by sentence """

    def __init__(self, always_apply=False, p=0.5):

        super(ShuffleSentencesTransform, self).__init__(always_apply, p)



    def apply(self, data, **params):

        sentences = self.get_sentences(data)

        random.shuffle(sentences)

        return ' '.join(sentences)
transform = ShuffleSentencesTransform(p=1.0)



text = train.loc[45]['text']



transform(data=(text))['data']
class SwapWordsTransform(NLPTransform):

    """ Do shuffle by words """

    def __init__(self, always_apply=False, p=0.5,verbose=False):

        self.probability=p

        self.verbose=verbose

        

        super(SwapWordsTransform, self).__init__(always_apply, p)



    def apply(self, data, **params):

        

        words=data.split()

        

        if random.random() < self.probability:

            # Storing the two elements 

            get = random.choice(words),random.choice(words)

            

            pos1=words.index(get[0])

            pos2=words.index(get[1])

            

            while pos1==pos2:

                pos2=words.index(random.choice(words))

            

            # unpacking those elements 

            words[pos2], words[pos1] = get 



        if self.verbose:

            return data,' '.join(words)

        

        return ' '.join(words)
transform = SwapWordsTransform(p=1.0,verbose=True)



text = train.loc[45]['text']



transform(data=(text))['data']
class DeleteWordsTransform(NLPTransform):

    """ Do shuffle by words """

    def __init__(self, always_apply=False, p=0.5,verbose=False):

        self.probability=p

        self.verbose=verbose

        

        super(DeleteWordsTransform, self).__init__(always_apply, p)



    def apply(self, data, **params):

        

        words=data.split()

        

        if random.random() < self.probability:

            get = random.choice(words)

            words.remove(get)



        if  self.verbose:

            return data,' '.join(words)

        

        return ' '.join(words)
transform = DeleteWordsTransform(p=1.0,verbose=True)



text = train.loc[45]['text']



transform(data=(text))['data']
configs={

    'EPOCHS' : 2,

    'SPLIT' : 0.8,

    'MAXLEN' : 100,

    'DROP_RATE' : 0.3,

    'OUTPUT_UNITS' : 2,

    'BATCH_SIZE' : 64,

    'LR' : (4e-5, 1e-2),

    'BERT_UNITS' : 768,

    'VAL_BATCH_SIZE' : 16,

    'MODEL_SAVE_PATH' : 'model.pt'

    

}
train=pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')
train.head(3)
def get_train_transforms():

    return albumentations.Compose([

        albumentations.OneOf([

            SwapWordsTransform(p=0.5),

            ShuffleSentencesTransform(p=0.8)

        ]),

        DeleteWordsTransform(p=0.3)

    ])
model = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model)
class DatasetRetriever(Dataset):



    def __init__(self, df,tokenizer, train_transforms=None):

        self.comment_texts = df['text'].values

        self.train_transforms = train_transforms

        self.target = df['target'].values

        self.tokenizer=tokenizer

        

    def __len__(self):

        return len(self.comment_texts)



    def __getitem__(self, idx):

        

        start, finish = 101,102

        pg, tg = 'post', 'post'

        tweet = str(self.comment_texts[idx]).strip()

        target=self.target[idx]

        

        if self.train_transforms:

            text = self.train_transforms(data=(tweet))['data']

            

        tweet_ids = self.tokenizer.encode(text)

        mask = [1] * len(tweet_ids)

            

        padding_length = configs['MAXLEN'] - len(tweet_ids)

        

        if padding_length > 0:

            input_ids = np.array(tweet_ids + ([0] * padding_length))

            mask = np.array(mask + ([0] * padding_length))

        else:

            input_ids=np.array(tweet_ids)

            mask = np.array(mask)

        

        #attention_mask = mask.reshape((1, -1))

        

        sentiment = torch.FloatTensor(to_categorical(target, num_classes=2))

        

        return sentiment, torch.LongTensor(input_ids), torch.LongTensor(mask)
dataset = DatasetRetriever(train[:2],tokenizer=tokenizer, train_transforms=get_train_transforms())



dataset[0]
del dataset;gc.collect()
class BERT(nn.Module):

    def __init__(self):

        super(BERT, self).__init__()

        self.softmax = nn.Softmax(dim=1)

        self.drop = nn.Dropout(configs['DROP_RATE'])

        self.bert = BertModel.from_pretrained(model)

        self.dense = nn.Linear(configs['BERT_UNITS'],configs['OUTPUT_UNITS'])

        

    def forward(self, inp, att):

        inp,att = inp.view(-1, configs['MAXLEN']),att.view(-1, configs['MAXLEN'])

        _, self.feat = self.bert(inp, att)

        return self.softmax(self.dense(self.drop(self.feat)))

def cel(inp, target):

    _, labels = target.max(dim=1)

    return nn.CrossEntropyLoss()(inp, labels)



def accuracy(inp, target):

    inp_ind = inp.max(axis=1).indices

    target_ind = target.max(axis=1).indices

    return (inp_ind == target_ind).float().sum(axis=0)
class AverageMeter:

    """

    Computes and stores the average and current value

    """

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count





class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.001):

        self.patience = patience

        self.counter = 0

        self.mode = mode

        self.best_score = None

        self.early_stop = False

        self.delta = delta

        if self.mode == "min":

            self.val_score = np.Inf

        else:

            self.val_score = -np.Inf



    def __call__(self, epoch_score, model, model_path):



        if self.mode == "min":

            score = -1.0 * epoch_score

        else:

            score = np.copy(epoch_score)



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

            self.counter = 0



    def save_checkpoint(self, epoch_score, model, model_path):

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:

            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))

            torch.save(model.state_dict(), model_path)

        self.val_score = epoch_score





def accuracy(inp, target):

    inp_ind = inp.argmax(axis=1)

    target_ind = target.argmax(axis=1)

    return accuracy_score(target_ind,inp_ind)
def train_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    losses = AverageMeter()

    total_accuracy = AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))

    

    for bi, d in enumerate(tk0):



        target,ids,mask=d



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        target = target.to(device, dtype=torch.long)



        model.zero_grad()

        output = model(

            ids,mask

        )

        

        loss = cel(output, target)

        

        output=output.cpu().detach().numpy()

        target=target.cpu().detach().numpy()

    

        loss.backward()

        optimizer.step()

        scheduler.step()



        accuracies = []

        for px, tweet in enumerate(target):

            y_true = target[px].reshape(-1,2)

            y_pred = output[px].reshape(-1,2)

          

            acc_score = accuracy(

               y_true,y_pred

            )

            accuracies.append(acc_score)



        total_accuracy.update(np.mean(accuracies), ids.size(0))

        losses.update(loss.item(), ids.size(0))

        tk0.set_postfix(loss=losses.avg, acc=total_accuracy.avg)

        

        

def eval_fn(data_loader, model, device):

    model.eval()

    losses = AverageMeter()

    total_accuracy = AverageMeter()

    

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        for bi, d in enumerate(tk0):

            

            target,ids,mask=d

            

            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            target = target.to(device, dtype=torch.long)



            output = model(

                ids,mask

            )

            

            loss = cel(output, target)

            

            output = output.cpu().detach().numpy()

            target=target.cpu().detach().numpy()

        

            

            accuracies = []

            for px, tweet in enumerate(target):

                y_true = target[px].reshape(-1,2)

                y_pred = output[px].reshape(-1,2)



                acc_score = accuracy(

                   y_true,y_pred

                )

                accuracies.append(acc_score)



            total_accuracy.update(np.mean(accuracies), ids.size(0))

            losses.update(loss.item(), ids.size(0))

            tk0.set_postfix(loss=losses.avg, acc=total_accuracy.avg)

    

    print(f"Accuracy = {total_accuracy.avg}")

    return total_accuracy.avg
def engine(train_df):

    train_df = shuffle(train_df)

    train_df = train_df.reset_index(drop=True)

    device = torch.device("cuda")



    split = np.int32(configs['SPLIT']*len(train_df))

    val_df, train_df = train_df[split:], train_df[:split]



    val_df = val_df.reset_index(drop=True)

    

    val_dataset = DatasetRetriever(val_df, tokenizer,get_train_transforms())

    val_loader = DataLoader(val_dataset, batch_size=configs['VAL_BATCH_SIZE'],

                            num_workers=4)



    train_df = train_df.reset_index(drop=True)

    

    train_dataset = DatasetRetriever(train_df, tokenizer,get_train_transforms())

    train_loader = DataLoader(train_dataset, batch_size=configs['BATCH_SIZE'],

                              num_workers=4)



    model = BERT().to(device)



    num_train_steps = int(len(train_df) / configs['BATCH_SIZE'] * configs['EPOCHS'])

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]

    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(

        optimizer, 

        num_warmup_steps=0, 

        num_training_steps=num_train_steps

    )



    es = EarlyStopping(patience=2, mode="max")

    

    start = time.time()

    print("STARTING TRAINING ...\n")



    

    for epoch in range(configs['EPOCHS']):

        train_fn(train_loader, model, optimizer, device, scheduler=scheduler)

        acc = eval_fn(val_loader, model, device)

        print(f"Accuracy Score = {acc}")

        es(acc, model, model_path="model.bin")

        if es.early_stop:

            print("Early stopping")

            break

            

gc.collect()
engine(train)