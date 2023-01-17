!pip install ../input/sacremoses/sacremoses-master/

!pip install ../input/transformers/transformers-master/
!ls ../input/pretrained-bert-models-for-pytorch/bert-base-uncased
from collections import defaultdict

from dataclasses import dataclass

import functools

import gc

import itertools

import json

from multiprocessing import Pool

import os

from pathlib import Path

import random

import re

import shutil

import subprocess

import time

from typing import Callable, Dict, List, Generator, Tuple

from os.path import join as path_join



import numpy as np

import pandas as pd

from pandas.io.json._json import JsonReader

from sklearn.preprocessing import LabelEncoder

from tqdm._tqdm_notebook import tqdm_notebook as tqdm



import torch

from torch import nn, optim

from torch.utils.data import Dataset, Subset, DataLoader



from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig

from transformers.optimization import get_linear_schedule_with_warmup
DATA_DIR = Path('../input/google-quest-challenge/')

train_df = pd.read_csv(path_join(DATA_DIR, 'train.csv'))

test_df = pd.read_csv(path_join(DATA_DIR, 'test.csv'))

print(train_df.shape, test_df.shape)
train_df['text'] = train_df['question_title'] + ' ' + train_df['question_body'] + ' ' + train_df['answer']

test_df['text'] = test_df['question_title'] + ' ' + test_df['question_body'] + ' ' + test_df['answer']
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
start_time = time.time()



seed = 42



max_seq_len = 512



num_labels = len(targets)

n_epochs = 1

lr = 2e-5

warmup = 0.05

batch_size = 64

accumulation_steps = 4



bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'



bert_model = 'bert-base-uncased'

do_lower_case = 'uncased' in bert_model

device = torch.device('cuda')



output_model_file = 'bert_pytorch.bin'

output_optimizer_file = 'bert_pytorch_optimizer.bin'

output_amp_file = 'bert_pytorch_amp.bin'



random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
def convert_lines(example, max_seq_length, tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    return np.array(all_tokens)
class BertForSequenceClassification(BertPreTrainedModel):

    r"""

        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:

            Labels for computing the sequence classification/regression loss.

            Indices should be in ``[0, ..., config.num_labels - 1]``.

            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),

            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:

        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:

            Classification (or regression if config.num_labels==1) loss.

        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``

            Classification (or regression if config.num_labels==1) scores (before SoftMax).

        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)

            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)

            of shape ``(batch_size, sequence_length, hidden_size)``:

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.

        **attentions**: (`optional`, returned when ``config.output_attentions=True``)

            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1

        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

    """

    def __init__(self, config):

        super(BertForSequenceClassification, self).__init__(config)

        self.num_labels = config.num_labels



        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)



        self.init_weights()



    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,

                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):



        outputs = self.bert(input_ids,

                            attention_mask=attention_mask,

                            token_type_ids=token_type_ids,

                            position_ids=position_ids,

                            head_mask=head_mask,

                            inputs_embeds=inputs_embeds)



        pooled_output = outputs[1]



        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)



        return logits
def loss_fn(preds, labels):

    loss = nn.BCEWithLogitsLoss()

    class_loss = loss(class_preds, class_labels)

    return class_loss
bert_config = BertConfig.from_json_file(bert_model_config)

bert_config.num_labels = len(targets)



model_path = os.path.join('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/')



model = BertForSequenceClassification.from_pretrained(model_path, config=bert_config)

model = model.to(device)



param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]



optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)



for param in model.parameters():

    param.requires_grad = False

model.eval()



tokenizer = BertTokenizer.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt',

                                          do_lower_case=do_lower_case)
X_test = convert_lines(test_df["text"].fillna("DUMMY_VALUE"), max_seq_len, tokenizer)
test_preds = np.zeros((len(X_test), len(targets)))

test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

tk0 = tqdm(test_loader)

for i, (x_batch,) in enumerate(tk0):

    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

    test_preds[i * batch_size:(i + 1) * batch_size] = pred[:,:].detach().cpu().squeeze().numpy()



test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy()
test_pred.shape
submission = pd.DataFrame.from_dict({

    'qa_id': test_df['qa_id']

})
for i in range(len(targets)):

    submission[targets[i]] = test_pred[:, i]
submission.to_csv('submission.csv', index=False)
submission.head()