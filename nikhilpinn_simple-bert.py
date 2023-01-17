import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.optim as optim

import torch.nn.functional as F
!pip install ../input/pytorchlightning/test_tube-0.7.5-py3-none-any.whl

!pip install ../input/pytorchlightning/pytorch_lightning-0.5.3.2-py3-none-any.whl

# !pip install transformers

# !pip install pytorch_lightning
from collections import Counter

import spacy

import os

from tqdm import tqdm, tqdm_notebook, tnrange

import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from transformers import *

import pytorch_lightning as pl

from pytorch_lightning import Trainer
targets = [

        'question_asker_intent_understanding',

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

        'answer_well_written'    

    ]



input_columns = ['question_title', 'question_body', 'answer']
class VectorizeData(Dataset):

    def __init__(self, df, maxlen=100):

        self.maxlen = maxlen

        self.df = df

#         self.df['text_padded'] = self.df.vectorized.apply(lambda x: self.pad_data(x))

        

        self.tokenizer = BertTokenizer.from_pretrained("../input/bertbasepytorch/vocab.txt")

        self.targets = [

        'question_asker_intent_understanding',

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

        'answer_well_written'    

    ]

    

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, idx):

        title = self.df.question_title.values[idx]

        ques = self.df.question_body.values[idx]

        ans = self.df.answer.values[idx]

        

        sent2idx = torch.tensor(self.tokenizer.encode(title+" [SEP] "+ques+" [SEP] "+ans, add_special_tokens=True)).unsqueeze(0)

        sent2idx = sent2idx.reshape(-1)

        sent2idx = self.pad_data(sent2idx)

        

        labelVect = torch.tensor([self.df.iloc[idx][x] for x in self.targets],requires_grad=False)

        

        return sent2idx,labelVect

    

    def pad_data(self, s):

        padded = np.zeros((self.maxlen,), dtype=np.int64)

        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]

        else: padded[:len(s)] = s

        return padded
class TestDataset(Dataset):

    def __init__(self, df, maxlen=100):

        self.maxlen = maxlen

        self.df = df

#         self.df['text_padded'] = self.df.vectorized.apply(lambda x: self.pad_data(x))

        

        self.tokenizer = BertTokenizer.from_pretrained("../input/bertbasepytorch/vocab.txt")

    

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, idx):

        title = self.df.question_title.values[idx]

        ques = self.df.question_body.values[idx]

        ans = self.df.answer.values[idx]

        uniq_id = self.df['qa_id'].values[idx]

        

        sent2idx = torch.tensor(self.tokenizer.encode(title+" [SEP] "+ques+" [SEP] "+ans, add_special_tokens=True)).unsqueeze(0)

        sent2idx = sent2idx.reshape(-1)

        sent2idx = torch.tensor(self.pad_data(sent2idx))

                

        return uniq_id, sent2idx

    

    def pad_data(self, s):

        padded = np.zeros((self.maxlen,), dtype=np.int64)

        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]

        else: padded[:len(s)] = s

        return padded
class BertGQA(pl.LightningModule):



    def __init__(self):

        super(BertGQA, self).__init__()



        self.textEnc = BertModel.from_pretrained("../input/bertbasepytorch")

        self.tokenizer = BertTokenizer.from_pretrained("../input/bertbasepytorch/vocab.txt")

        self.fc = nn.Linear(768,30)

        self.loss = nn.MSELoss()

        self.sigm = nn.Sigmoid()

        

    def forward(self, x):

        xVec = self.textEnc(x)[1]

        

        return self.sigm(self.fc(xVec))



    def training_step(self, batch, batch_idx):

        # REQUIRED

        x, y = batch

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}



    def validation_step(self, batch, batch_idx):

        # OPTIONAL

        x, y = batch

        y_hat = self.forward(x)

        return {'val_loss': self.loss(y_hat, y)}



    def validation_end(self, outputs):

        # OPTIONAL

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

        

    def test_step(self, batch, batch_idx):

        # OPTIONAL

        x, y = batch

        y_hat = self.forward(x)

        return {'test_loss': self.loss(y_hat, y)}



    def test_end(self, outputs):

        # OPTIONAL

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss}

        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}



    def configure_optimizers(self):

        # REQUIRED

        # can return multiple optimizers and learning_rate schedulers

        # (LBFGS it is automatically supported, no need for closure function)

        return torch.optim.Adam(self.fc.parameters(), lr=0.02)

    

    @pl.data_loader

    def train_dataloader(self):

        # REQUIRED

        train = pd.read_csv('../input/google-quest-challenge/train.csv')

        trainDataset = VectorizeData(train)

        return DataLoader(dataset=trainDataset, batch_size=32, shuffle=True)



    @pl.data_loader

    def val_dataloader(self):

#         # OPTIONAL

        train = pd.read_csv('../input/google-quest-challenge/train.csv')

        trainDataset = VectorizeData(train)

        return DataLoader(dataset=trainDataset, batch_size=32, shuffle=True)

        pass



    @pl.data_loader

    def test_dataloader(self):

#         # OPTIONAL

#         test =  pd.read_csv('./data/test.csv')

        pass
model = BertGQA()



trainer = Trainer(min_nb_epochs=1, max_nb_epochs=1,gpus=1)

trainer.fit(model)
test =  pd.read_csv('../input/google-quest-challenge/test.csv')

testset = TestDataset(test)

testLoader = DataLoader(dataset=testset, batch_size=1, shuffle=False)
results = []



for i, (idx,text) in enumerate(testLoader):

    text = text.cuda()

    out = model(text)

    finallist = out.reshape(-1).tolist()

    finallist = [x + 0.01 if x == 0 else x for x in finallist]

    finallist = [x - 0.01 if x == 1 else x for x in finallist]

    finallist = [x + np.random.uniform(0,0.001) for x in finallist]

    results.append(finallist)
# import csv



# with open("./submission.csv", "w", newline="") as f:

#     writer = csv.writer(f)

#     writer.writerows(results)
# scaler = MinMaxScaler((0.01, 0.99))

# results = scaler.fit_transform(results)

# results.shape
import pandas as pd



submission_df = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')

submission_df[targets] =  results

submission_df
sub_file_name = 'submission.csv'

submission_df.to_csv(sub_file_name, index=False)