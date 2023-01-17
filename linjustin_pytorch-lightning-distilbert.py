!pip install pytorch-lightning
import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl



from sklearn.model_selection import train_test_split



import numpy as np

import pandas as pd

import random

import time

import string

import re

import bs4

from bs4 import BeautifulSoup



import transformers

from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer



SEED = 1234

np.random.seed(SEED)

random.seed(SEED)

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



print("PyTorch version: ", torch.__version__)

print('Huggingface version: ', transformers.__version__)

print('PyTorch Lightning version: ', pl.__version__)
PATH = '../input/tweet-sentiment-extraction/'



train_df = pd.read_csv(PATH + 'train.csv')

train_df.head()
test_df = pd.read_csv(PATH + 'test.csv')

test_df.head()
def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)



def remove_html(text):

    return BeautifulSoup(text, "lxml").text
train_df['text'] = train_df['text'].apply(lambda x: str(x))

test_df['text'] = test_df['text'].apply(lambda x: str(x))



train_df['text'] = train_df['text'].apply(lambda text: remove_urls(text))

test_df['text'] = test_df['text'].apply(lambda text: remove_urls(text))



train_df['text'] = train_df['text'].apply(lambda text: remove_html(text))

test_df['text'] = test_df['text'].apply(lambda text: remove_html(text))



train_df['text_lower'] = train_df['text'].apply(lambda x: x.lower())

test_df['text_lower'] = test_df['text'].apply(lambda x: x.lower())



train_df['selected_text'] = train_df['selected_text'].apply(lambda x: str(x).lower())
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')



train_df['text_tokenized'] = train_df['text_lower'].apply(tokenizer.tokenize)

test_df['text_tokenized'] = test_df['text_lower'].apply(tokenizer.tokenize)



train_df['selected_text_tokenized'] = train_df['selected_text'].apply(tokenizer.tokenize)
# Start and end positions



start_positions = []

end_positions = []



train_df['select_length'] = train_df['selected_text_tokenized'].map(len)



for i in range(len(train_df)):

    start_position = [j for j, token in enumerate(train_df['text_tokenized'].iloc[i]) if token == train_df['selected_text_tokenized'].iloc[i][0]]

    end_position = [j for j, token in enumerate(train_df['text_tokenized'].iloc[i]) if token == train_df['selected_text_tokenized'].iloc[i][-1]]

    

    start_position = [idx for idx in start_position if idx + train_df['select_length'].iloc[i] - 1 in end_position]

    end_position = [idx for idx in end_position if idx - train_df['select_length'].iloc[i] + 1 in start_position]

    

    start_positions.append(start_position)

    end_positions.append(end_position)
start_positions = [l[0] if len(l) > 0 else -1 for l in start_positions]

end_positions = [l[0] if len(l) > 0 else -1 for l in end_positions]
train_df['start_position'] = start_positions

train_df['end_position'] = end_positions



test_df['start_position'] = -1

test_df['end_position'] = -1



train_df = train_df.query('start_position!=-1')
train_df, val_df = train_test_split(train_df, test_size = 0.2, random_state = SEED)
pos_train = train_df.query('sentiment=="positive"')

neg_train = train_df.query('sentiment=="negative"')

neu_train = train_df.query('sentiment=="neutral"')



pos_val = val_df.query('sentiment=="positive"')

neg_val = val_df.query('sentiment=="negative"')

neu_val = val_df.query('sentiment=="neutral"')



pos_test = test_df.query('sentiment=="positive"')

neg_test = test_df.query('sentiment=="negative"')

neu_test = test_df.query('sentiment=="neutral"')
pos_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

neg_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')



MAX_LENGTH = 128

BATCH_SIZE = 32
class TrainDataset(Dataset):

    def __init__(self, df):

        super().__init__()

        self.texts = df['text_lower'].values

        self.start_ids = df['start_position'].values

        self.end_ids = df['end_position'].values

        self.hash_index = df['textID'].values

        

    def __len__(self):

        return len(self.texts)

    

    def __getitem__(self, idx):

        returns = {'text': self.texts[idx],

                   'start': self.start_ids[idx],

                   'end': self.end_ids[idx],

                   'idx': idx}

        return returns

    

class TestDataset(Dataset):

    def __init__(self, df):

        super().__init__()

        self.texts = df['text_lower'].values

        self.hash_index = df['textID'].values



    def __len__(self):

        return len(self.texts)



    def __getitem__(self, idx):

        returns = {'text' : self.texts[idx],

                   'idx' : idx}

        return returns
ds_pos_train = TrainDataset(pos_train)

ds_neg_train = TrainDataset(neg_train)



ds_pos_val = TrainDataset(pos_val)

ds_neg_val = TrainDataset(neg_val)



ds_pos_test = TestDataset(pos_test)

ds_neg_test = TestDataset(neg_test)







dl_pos_train = DataLoader(ds_pos_train, 

                          batch_size = BATCH_SIZE, 

                          shuffle = True,

                          num_workers = 8)

dl_neg_train = DataLoader(ds_neg_train, 

                          batch_size = BATCH_SIZE, 

                          shuffle = True,

                          num_workers = 8)



dl_pos_val = DataLoader(ds_pos_val, 

                        batch_size = BATCH_SIZE, 

                        shuffle = False,

                        num_workers = 8)

dl_neg_val = DataLoader(ds_neg_val, 

                        batch_size = BATCH_SIZE, 

                        shuffle = False,

                        num_workers = 8)



dl_pos_test = DataLoader(ds_pos_test, 

                         batch_size = BATCH_SIZE, 

                         shuffle = False,

                         num_workers = 8)

dl_neg_test = DataLoader(ds_neg_test, 

                         batch_size = BATCH_SIZE, 

                         shuffle = False,

                         num_workers = 8)
class DistilBERTModule(pl.LightningModule):

    def __init__(self, distilbertmodel, tokenizer, prediction_save_path):

        super().__init__()

        self.distilbertmodel = distilbertmodel

        self.tokenizer = tokenizer

        self.prediction_save_path = prediction_save_path

        

    def get_device(self):

        return self.distilbertmodel.state_dict()['distilbert.embeddings.word_embeddings.weight'].device

    

    def save_predictions(self, start_posit, end_posit):

        df = pd.DataFrame({'start_position': start_posit,

                           'end_position': end_posit})

        df.to_csv(self.prediction_save_path, index = False)

        

    def forward(self, batch):

        encoded_batch = tokenizer.batch_encode_plus(batch['text'],

                                                    max_length = MAX_LENGTH,

                                                    pad_to_max_length = True)

        input_ids = torch.tensor(encoded_batch['input_ids']).to(self.get_device())

        attention_mask = torch.tensor(encoded_batch['attention_mask']).to(self.get_device())

        start_posit = batch['start'].to(self.get_device()) + 1 if 'start' in batch.keys() else None

        end_posit = batch['end'].to(self.get_device()) + 1 if 'end' in batch.keys() else None

        

        model_inputs = {'input_ids': input_ids,

                        'attention_mask': attention_mask,

                        'start_positions': start_posit,

                        'end_positions': end_posit}

        

        return self.distilbertmodel(**model_inputs)

    

    def training_step(self, batch, batch_nb):

        idx = batch['idx']

        loss = self.forward(batch)[0]

        return {'loss': loss, 'idx': idx}

    

    def validation_step(self, batch, batch_nb):

        idx = batch['idx']

        loss = self.forward(batch)[0]

        return {'loss': loss, 'idx': idx}

    

    def test_step(self, batch, batch_nb):

        idx = batch['idx']

        start_scores = self.forward(batch)[0]

        end_scores = self.forward(batch)[1]

        return {'start_scores':start_scores, 'end_scores':end_scores, 'idx':idx}

    

    def training_end(self, outputs):

        return {'loss':outputs['loss']}

    

    def validation_end(self, outputs):

        return {'loss':torch.mean(torch.tensor([output['loss'] for output in outputs])).detach()}

    

    def test_end(self, outputs):

        start_scores = torch.cat([output['start_scores'] for output in outputs]).detach().cpu().numpy()

        start_positions = np.argmax(start_scores, axis=1) - 1



        end_scores = torch.cat([output['end_scores'] for output in outputs]).detach().cpu().numpy()

        end_positions = np.argmax(end_scores, axis=1) - 1

        self.save_predictions(start_positions, end_positions)

        return {}

    

    def configure_optimizers(self):

        return optim.Adam(self.parameters(), lr = 2e-5)

    

    @pl.data_loader

    def train_dataloader(self):

        pass



    @pl.data_loader

    def val_dataloader(self):

        pass



    @pl.data_loader

    def test_dataloader(self):

        pass
class PositiveModule(DistilBERTModule):

    def __init__(self, distilbertmodel, tokenizer, prediction_save_path):

        super().__init__(distilbertmodel, tokenizer, prediction_save_path)



    @pl.data_loader

    def train_dataloader(self):

        return dl_pos_train



    @pl.data_loader

    def val_dataloader(self):

        return dl_pos_val



    @pl.data_loader

    def test_dataloader(self):

        return dl_pos_test

    

    



class NegativeModule(DistilBERTModule):

    def __init__(self, distilbertmodel, tokenizer, prediction_save_path):

        super().__init__(distilbertmodel, tokenizer, prediction_save_path)



    @pl.data_loader

    def train_dataloader(self):

        return dl_neg_train



    @pl.data_loader

    def val_dataloader(self):

        return dl_neg_val



    @pl.data_loader

    def test_dataloader(self):

        return dl_neg_test
pos_module = PositiveModule(pos_model, tokenizer, 'pos_pred.csv')

neg_module = NegativeModule(neg_model, tokenizer, 'neg_pred.csv')



device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'



pos_module.to(device)

neg_module.to(device)
pos_trainer = pl.Trainer(max_nb_epochs = 5, fast_dev_run = False)

neg_trainer = pl.Trainer(max_nb_epochs = 5, fast_dev_run = False)



pos_trainer.fit(pos_module)

neg_trainer.fit(neg_module)
pos_trainer.test()

neg_trainer.test()
pos_pred = pd.read_csv('pos_pred.csv')

neg_pred = pd.read_csv('neg_pred.csv')

pos_pred.head()
neg_pred.head()
test_df.index = test_df['textID']

test_df['selected_text'] = ''



test_df.loc[ds_pos_test.hash_index[:BATCH_SIZE if False else len(test_df)], 'start_position':'end_position'] = pos_pred.values

test_df.loc[ds_neg_test.hash_index[:BATCH_SIZE if False else len(test_df)], 'start_position':'end_position'] = neg_pred.values

test_df.head()
test_df.tail()
for i in range(len(test_df)):

    if test_df['sentiment'].iloc[i] in ('positive', 'negative'):

        tokenized_text = test_df['text_tokenized'].iloc[i]

        start_position = max(test_df['start_position'].iloc[i], 0)

        end_position = min(test_df['end_position'].iloc[i], len(tokenized_text) - 1)

        

        selected_text = tokenizer.convert_tokens_to_string(tokenized_text[start_position:end_position + 1])

        for original_token in test_df['text'].iloc[i].split():

            tokenized_form = tokenizer.convert_tokens_to_string(tokenizer.tokenize(original_token))

            selected_text = selected_text.replace(tokenized_form, original_token, 1)

            

        test_df['selected_text'].iloc[i] = selected_text
for i in range(len(test_df)):

    if test_df['sentiment'].iloc[i] == 'neutral':

        test_df['selected_text'].iloc[i] = test_df['text'].iloc[i]

    else:

        pass
test_df.loc[:,['textID', 'selected_text']]
test_df.loc[:, ['textID', 'selected_text']].to_csv('submission.csv', index = False)