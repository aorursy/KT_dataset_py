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
import numpy as np

import pandas as pd

import os

import warnings

import random

import torch 

from torch import nn

import torch.optim as optim

from sklearn.model_selection import StratifiedKFold

import tokenizers



## load transformers

from transformers import RobertaModel, RobertaConfig

from transformers import BertModel, BertConfig, BertForPreTraining



# to check the progress

from tqdm.notebook import tqdm



warnings.filterwarnings('ignore')



# check the gpu availability

print('GPU is available: ', torch.cuda.is_available())
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True



seed = 42

seed_everything(seed)
def get_selected_text(text, start_idx, end_idx, offsets):

    selected_text = ""

    for ix in range(start_idx, end_idx + 1):

        selected_text += text[offsets[ix][0]: offsets[ix][1]]

        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:

            selected_text += " "

    return selected_text



# the one given in Overview-Evaluation part of problem description

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))





def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):

    start_pred = np.argmax(start_logits)

    end_pred = np.argmax(end_logits)

    if start_pred > end_pred:

        pred = text

    else:

        pred = get_selected_text(text, start_pred, end_pred, offsets)

        

    true = get_selected_text(text, start_idx, end_idx, offsets)

    

    return jaccard(true, pred)
def loss_fn(start_logits, end_logits, start_positions, end_positions):

    ce_loss = nn.CrossEntropyLoss()

    start_loss = ce_loss(start_logits, start_positions)

    end_loss = ce_loss(end_logits, end_positions)    

    total_loss = start_loss + end_loss

    return total_loss
class TweetDataset(torch.utils.data.Dataset):

    def __init__(self, df, max_len=96):

        self.df = df

        self.max_len = max_len

        self.labeled = 'selected_text' in df

        self.tokenizer = tokenizers.ByteLevelBPETokenizer(

            vocab_file='../input/roberta-base/vocab.json', 

            merges_file='../input/roberta-base/merges.txt', 

            lowercase=True,

            add_prefix_space=True)



    def __getitem__(self, index):

        data = {}

        row = self.df.iloc[index]

        

        ids, masks, tweet, offsets = self.get_input_data(row)

        data['ids'] = ids

        data['masks'] = masks

        data['tweet'] = tweet

        data['offsets'] = offsets

        

        if self.labeled:

            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)

            data['start_idx'] = start_idx

            data['end_idx'] = end_idx

        

        return data



    def __len__(self):

        return len(self.df)

    

    def get_input_data(self, row):

        tweet = " " + " ".join(row.text.lower().split())

        encoding = self.tokenizer.encode(tweet)

        sentiment_id = self.tokenizer.encode(row.sentiment).ids

        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]

        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

                

        pad_len = self.max_len - len(ids)

        if pad_len > 0:

            ids += [1] * pad_len

            offsets += [(0, 0)] * pad_len

        

        ids = torch.tensor(ids)

        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))

        offsets = torch.tensor(offsets)

        

        return ids, masks, tweet, offsets

        

    def get_target_idx(self, row, tweet, offsets):

        selected_text = " " +  " ".join(row.selected_text.lower().split())



        len_st = len(selected_text) - 1

        idx0 = None

        idx1 = None



        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):

            if " " + tweet[ind: ind+len_st] == selected_text:

                idx0 = ind

                idx1 = ind + len_st - 1

                break



        char_targets = [0] * len(tweet)

        if idx0 != None and idx1 != None:

            for ct in range(idx0, idx1 + 1):

                char_targets[ct] = 1



        target_idx = []

        for j, (offset1, offset2) in enumerate(offsets):

            if sum(char_targets[offset1: offset2]) > 0:

                target_idx.append(j)



        start_idx = target_idx[0]

        end_idx = target_idx[-1]

        

        return start_idx, end_idx

        

def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):

    train_df = df.iloc[train_idx]

    val_df = df.iloc[val_idx]



    train_loader = torch.utils.data.DataLoader(

        TweetDataset(train_df), 

        batch_size=batch_size, 

        shuffle=True, 

        num_workers=2,

        drop_last=True)



    val_loader = torch.utils.data.DataLoader(

        TweetDataset(val_df), 

        batch_size=batch_size, 

        shuffle=False, 

        num_workers=2)



    dataloaders_dict = {"train": train_loader, "val": val_loader}



    return dataloaders_dict



def get_test_loader(df, batch_size=32):

    loader = torch.utils.data.DataLoader(

        TweetDataset(df), 

        batch_size=batch_size, 

        shuffle=False, 

        num_workers=2)    

    return loader
class TweetModel(nn.Module):

    def __init__(self):

        super(TweetModel, self).__init__()

        

        config = RobertaConfig.from_pretrained(

            '../input/roberta-base/config.json', output_hidden_states=True)    

        self.roberta = RobertaModel.from_pretrained(

            '../input/roberta-base/pytorch_model.bin', config=config)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(config.hidden_size, 2)

        nn.init.normal_(self.fc.weight, std=0.02)

        nn.init.normal_(self.fc.bias, 0)

    



    def forward(self, input_ids, attention_mask):

        _, _, hs = self.roberta(input_ids, attention_mask)

         

        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])

        x = torch.mean(x, 0)



        

        x = self.dropout(x)

        x = self.fc(x)

        start_logits, end_logits = x.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

                

        return start_logits, end_logits
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):

    model.cuda()



    for epoch in tqdm(range(num_epochs)):

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()

            else:

                model.eval()



            epoch_loss = 0.0

            epoch_jaccard = 0.0

            

            for data in tqdm((dataloaders_dict[phase])):

                ids = data['ids'].cuda()

                masks = data['masks'].cuda()

                tweet = data['tweet']

                offsets = data['offsets'].numpy()

                start_idx = data['start_idx'].cuda()

                end_idx = data['end_idx'].cuda()



                optimizer.zero_grad()



                with torch.set_grad_enabled(phase == 'train'):



                    start_logits, end_logits = model(ids, masks)



                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                    epoch_loss += loss.item() * len(ids)

                    

                    start_idx = start_idx.cpu().detach().numpy()

                    end_idx = end_idx.cpu().detach().numpy()

                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()

                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    

                    for i in range(len(ids)):                        

                        jaccard_score = compute_jaccard_score(

                            tweet[i],

                            start_idx[i],

                            end_idx[i],

                            start_logits[i], 

                            end_logits[i], 

                            offsets[i])

                        epoch_jaccard += jaccard_score

                    

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            

            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(

                epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))

    

    torch.save(model.state_dict(), filename)
num_epochs = 5

batch_size = 32

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

train_df['text'] = train_df['text'].astype(str)

train_df['selected_text'] = train_df['selected_text'].astype(str)



for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1): 

    print(f'Fold: {fold}')



    model = TweetModel()

    optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.999))

    criterion = loss_fn    

    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)



    train_model(

        model, 

        dataloaders_dict,

        criterion, 

        optimizer, 

        num_epochs,

        f'roberta_fold{fold}.pth')
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

test_df['text'] = test_df['text'].astype(str)

test_loader = get_test_loader(test_df)

predictions = []

models = []

for fold in range(skf.n_splits):

    model = TweetModel()

    model.cuda()

    model.load_state_dict(torch.load(f'roberta_fold{fold+1}.pth'))

    model.eval()

    models.append(model)



for data in test_loader:

    ids = data['ids'].cuda()

    masks = data['masks'].cuda()

    tweet = data['tweet']

    offsets = data['offsets'].numpy()



    start_logits = []

    end_logits = []

    for model in models:

        with torch.no_grad():

            output = model(ids, masks)

            start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())

            end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())



    start_logits = np.mean(start_logits, axis=0)

    end_logits = np.mean(end_logits, axis=0)

    for i in range(len(ids)):    

        start_pred = np.argmax(start_logits[i])

        end_pred = np.argmax(end_logits[i])

        if start_pred > end_pred:

            pred = tweet[i]

        else:

            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])

        predictions.append(pred)
sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

sub_df['selected_text'] = predictions

sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)

sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)

sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)

sub_df.to_csv('submission.csv', index=False)

sub_df.head()