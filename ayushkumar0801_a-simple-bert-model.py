import numpy as np

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.environ["SEED"] = "420"

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig

import re

from tqdm import tqdm
df_train = pd.read_csv("/kaggle/input/gapvalidation/gap-test.tsv", delimiter="\t")

df_val = pd.read_csv("/kaggle/input/gapvalidation/gap-validation.tsv", delimiter="\t")

df_test = pd.read_csv("/kaggle/input/gapvalidation/gap-development.tsv", delimiter="\t")

test_2 = pd.read_csv("/kaggle/input/gendered-pronoun-resolution/test_stage_2.tsv", delimiter="\t")



PRETRAINED_MODEL_NAME = 'bert-large-uncased'



bert_path = "../input/bert-base-uncased/"

tokenizer = BertTokenizer.from_pretrained(bert_path)

pad_len = 300
def conver_lower(df):

    df['Text'] = df.apply(lambda row: row['Text'].lower(), axis = 1)

    df['A'] = df.apply(lambda row: row['A'].lower(), axis = 1)

    df['B'] = df.apply(lambda row: row['B'].lower(), axis = 1)

    df['Pronoun'] = df.apply(lambda row: row['Pronoun'].lower(), axis = 1)

    return df

df_train = conver_lower(df_train)

df_test = conver_lower(df_test)

df_val = conver_lower(df_val)

test_2 = conver_lower(test_2)
tokenizer.add_tokens(['[A]', '[B]', '[P]'])

def insert_tag(row):

    to_be_inserted = sorted([

        (row["A-offset"], " [A] "),

        (row["B-offset"], " [B] "),

        (row["Pronoun-offset"], " [P] ")

    ], key=lambda x: x[0], reverse=True)

    text = row["Text"]

    for offset, tag in to_be_inserted:

        text = text[:offset] + tag + text[offset:]

    return text



def tokenize(text, tokenizer):

    entries = {}

    final_tokens = []

    for token in tokenizer.tokenize(text):

        if token in ("[A]", "[B]", "[P]"):

            entries[token] = len(final_tokens)

            continue

        final_tokens.append(token)

    return final_tokens, (entries["[A]"], entries["[B]"], entries["[P]"])



def target(row):

    if int(row['A-coref']) == 1:

        return 0

    elif int(row['B-coref']) == 1:

        return 1

    else:

        return 2

"""

The lower part was taken from 

            [PyTorch] BERT + EndpointSpanExtractor + KFold

"""

def children(m):

    return m if isinstance(m, (list, tuple)) else list(m.children())



def set_trainable_attr(m, b):

    m.trainable = b

    for p in m.parameters():

        p.requires_grad = b



def apply_leaf(m, f):

    c = children(m)

    if isinstance(m, nn.Module):

        f(m)

    if len(c) > 0:

        for l in c:

            apply_leaf(l, f)

            

def set_trainable(l, b):

    apply_leaf(l, lambda m: set_trainable_attr(m, b))
class modified_dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):

        p_text = []

        offsets = []

        at_mask = []

        self.y_lst = df[['A-coref', 'B-coref']].apply(lambda row: target(row), axis = 1)

        for row in tqdm(range(len(df))):

            tokens, offset = tokenize(insert_tag(df.iloc[row]), tokenizer)

            bla = tokenizer.encode_plus(tokens, max_length = pad_len, pad_to_max_length = True, return_token_type_ids = False)

            p_text.append(bla['input_ids'])

            at_mask.append(bla['attention_mask'])

            offsets.append(offset)

        self.p_text = torch.tensor(p_text)

        self.offsets = torch.tensor(offsets)

        self.at_mask = torch.tensor(at_mask)

        return 

    def __len__(self):

        return len(self.p_text)

    def __getitem__(self,item):

        return self.p_text[item], self.y_lst[item], self.offsets[item], self.at_mask[item]



class modified_dataset_test(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):

        p_text = []

        offsets = []

        at_mask = []

        for row in tqdm(range(len(df))):

            tokens, offset = tokenize(insert_tag(df.iloc[row]), tokenizer)

            bla = tokenizer.encode_plus(tokens, max_length = pad_len, pad_to_max_length = True, return_token_type_ids = False)

            p_text.append(bla['input_ids'])

            at_mask.append(bla['attention_mask'])

            offsets.append(offset)

        self.p_text = torch.tensor(p_text)

        self.offsets = torch.tensor(offsets)

        self.at_mask = torch.tensor(at_mask)

        return  

    def __len__(self):

        return len(self.p_text)

    def __getitem__(self,item):

        return self.p_text[item], self.offsets[item], self.at_mask[item]

 

def collate_fun(batch):

    tmp_lst = list(zip(*batch))

    return torch.stack(tmp_lst[0], axis = 0), torch.tensor(tmp_lst[1]), torch.stack(tmp_lst[2], axis = 0), torch.stack(tmp_lst[3], axis = 0)



def collate_fun2(batch):

    tmp_lst = list(zip(*batch))

    return torch.stack(tmp_lst[0], axis = 0), torch.stack(tmp_lst[1], axis = 0), torch.stack(tmp_lst[2], axis = 0)



train_loader = DataLoader(

        modified_dataset(df_train, tokenizer),

        batch_size=18,

        collate_fn=collate_fun,

        shuffle=True,

        drop_last=True,

        num_workers=2)

val_loader = DataLoader(

        modified_dataset(df_val, tokenizer),

        batch_size=30,

        collate_fn=collate_fun,

        shuffle=False,

        num_workers=2)

test_loader = DataLoader(

        modified_dataset(df_test, tokenizer),

        batch_size=30,

        collate_fn=collate_fun,

        shuffle=False,

        num_workers=2)

test_2_loader = DataLoader(

        modified_dataset_test(test_2, tokenizer),

        batch_size=30,

        collate_fn=collate_fun2,

        shuffle=False,

        num_workers=2)
GPU = torch.cuda.is_available()

torch.save({'train_loader':train_loader,

            'test_loader':test_loader,

            'val_loader':val_loader}, 'dataloader_new.pth')

torch.save({'test_dataloader':test_2_loader},'test_loader.pth')

# train_loader, test_loader, val_loader = torch.load('/kaggle/input/gap-dataloaders/dataloaders2.pth').values()

# test_2_loader = torch.load('/kaggle/input/gap-dataloaders/test_loader(1).pth')['test_dataloader_174']
class bert(nn.Module):

    def __init__(self, bert_path):

        super().__init__()

        BERT = BertModel.from_pretrained(bert_path, config = BertConfig.from_pretrained(bert_path, output_hidden_states = True))

        self.BERT = BERT

        self.fc = nn.Sequential(nn.BatchNorm1d(self.BERT.config.hidden_size * 3),

                                nn.Dropout(0.4),

                                nn.Linear(self.BERT.config.hidden_size * 3, 600),

                                nn.BatchNorm1d(600),

                                nn.Dropout(0.4),

                                nn.Linear(600, 600),

                                nn.BatchNorm1d(600),

                                nn.Dropout(0.4),

                                nn.Linear(600,3))

        

    def forward(self, token, at_mask, offsets, layer):

        out = self.BERT(token, attention_mask = at_mask)[2][layer]

        out_lst = []

        for j in range(out.shape[0]):

            out_lst.append(torch.stack([torch.tensor(out[j,offsets[j,0]]),torch.tensor(out[j,offsets[j,1]]),torch.tensor(out[j,offsets[j,2]])] , dim = 0) )

        out_lst = torch.stack([word_embedding for word_embedding in out_lst], dim = 0)

        out = out_lst.reshape(out_lst.shape[0], -1)

        out = self.fc(out)

        return out

        

def create_model(df_len,epoch_len):        

    model = bert(bert_path)

    criteria = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), eps = 1e-06, lr = 1e-4)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=df_len*epoch_len)

    return model, criteria, optimizer, scheduler
epoch_len = 20

model, criteria, optimizer, scheduler = create_model(len(df_train), epoch_len)

set_trainable(model.BERT, False)

aaa = 0

for t in range(epoch_len):

    tot_loss = 0

    correct_train = 0

    val_loss = 0

    val_correct = 0

    model = model.train()

    

    if GPU:

        model = model.cuda()

    

    for item in tqdm(train_loader):

        

        token = item[0]

        at_mask = item[3]

        offsets = item[2]

        target = item[1]

        if GPU:

            token = token.cuda()

            at_mask = at_mask.cuda()

            target = target.cuda()

            offsets = offsets.cuda()

            

        output = model(token, at_mask, offsets, -2)

        loss = criteria(output, target)

        tot_loss += loss.item()

        correct_train += torch.sum(torch.max(torch.nn.functional.softmax(output, dim = 1), dim = 1)[1] == target)

        

        optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        scheduler.step()

    

    with torch.no_grad():

        model = model.eval()

        

        if GPU:

            model = model.cuda()            

        for item in tqdm(val_loader):

            token = item[0]

            at_mask = item[3]

            offsets = item[2]

            target = item[1]

            

            if GPU:

                token = token.cuda()

                at_mask = at_mask.cuda()

                offsets = offsets.cuda()

                target = target.cuda()

                

            output = model(token, at_mask, offsets, -2)

            val_correct += torch.sum(torch.max(torch.nn.functional.softmax(output, dim = 1), dim = 1)[1] == target)

        if val_correct > aaa:

            bst_model = model

            aaa = val_correct

    print(tot_loss, correct_train,"   ", val_correct," out of ", len(val_loader)*30)
def predict(df, dataloader, model):

    tmp_array = np.zeros((len(df), 3))

    with torch.no_grad():

        model = model.eval()

        if GPU:

            model = model.cuda()

        

        j = 0

        for item in tqdm(dataloader):

            

            token = item[0]

            at_mask = item[2]

            offsets = item[1]



            if GPU:

                token = token.cuda()

                at_mask = at_mask.cuda()

                offsets = offsets.cuda()

            

            output = model(token, at_mask, offsets, -2)

            for zz in output.cpu():

                tmp_array[j] = zz

                j+=1

            

    return tmp_array
a = predict(test_2, test_2_loader, bst_model)
bla = test_2[['ID']].merge(pd.DataFrame(torch.nn.functional.softmax(torch.tensor(a), dim = 1).numpy()), left_index=True, right_index=True).set_index('ID')

bla.columns = ['A', 'B', 'NEITHER']

bla.to_csv('sbmsn2.csv')
torch.save({'model':bst_model}, 'model1.pth')
tst_model = torch.load('/kaggle/input/gendered-model/model1.pth')['model']