# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_path = '../input/tensorflow2-question-answering/simplified-nq-train.jsonl'
import numpy as np

import pandas as pd

import dask.bag as db 

import json

import re 

from tqdm import tqdm



import torch

from transformers import BertTokenizer, BertModel, BertForQuestionAnswering

import torch.nn as nn

import sklearn

from sklearn.model_selection import train_test_split
!pip install transformers 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_token_ids = tokenizer.encode("I am Alexander", "Who am I?")

print(input_token_ids)

segment_ids = [0 if i <= input_token_ids.index(102) else 1 for i in range(len(input_token_ids))]

print(segment_ids)



# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# start_scores, end_scores = model(torch.tensor([input_token_ids]), token_type_ids=torch.tensor([segment_ids]))

# all_tokens = tokenizer.convert_ids_to_tokens(input_token_ids)

# print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
def tokenize_text(text, question, max_seq_length):

    token_ids = tokenizer.encode(text, question)

    segment_ids = [0 if i <= token_ids.index(102) else 1 for i in range(len(token_ids))]

    mask = [1 for i in range(len(token_ids))]



    

    padding = [0] * (max_seq_length - len(token_ids))



    token_ids += padding

    segment_ids += padding

    mask += padding

    



    return token_ids, segment_ids, mask
class WikiArticles:

    def __init__(self, path, max_article_length):

        self.path = path

        self.data = None

        self.max_article_length = max_article_length

        self.input_token_ids = None

        self.segment_ids = None

        self.mask = None

        self.answers = None

    

    def download_data(self, n):

        data = []

        with open(self.path, 'r') as f:

            for key, value in tqdm(enumerate(f)):

                if (key+1) % n == 0:

                    break 

                el = json.loads(value)

                data.append(el)

        

        data = pd.DataFrame(data)

        data['yes_no'] = data['annotations'].apply(lambda x: x[0]['yes_no_answer'])

        

        data['long_start'] = data['annotations'].apply(lambda x: x[0]['long_answer']['start_token'])

        data['long_end'] = data['annotations'].apply(lambda x: x[0]['long_answer']['end_token'])

        

        data['short'] = data['annotations'].apply(lambda x: x[0]['short_answers'])

        

        start_values = []

        end_values = []

            

        for el in data['short']:

            end = -1

            start = -1

            

            if len(el) > 0:

                end = el[0]['end_token']

                start = el[0]['start_token']

            

            start_values.append(start)

            end_values.append(end)

            

        data['short_start'] = start_values

        data['short_end'] = end_values

        

        self.data = data.loc[:, ['document_text', 'question_text', 'example_id', 'yes_no', 

                                 'long_start', 'long_end', 'short_start', 'short_end']]

        

    def process(self):

        data = self.data

        input_token_ids_list = []

        segment_ids_list = []

        mask_list = []

        answers = []



        n = data.shape[0]

        

        for i in range(n):



            article = data.iloc[i, 0]



            if len(article.split()) < self.max_article_length:

                question = data.iloc[i, 1]



                start_long = data.iloc[i, 4]

                end_long = data.iloc[i, 5]

                start_short = data.iloc[i, 6]

                end_short = data.iloc[i, 7]



                answer = np.zeros((1, 2, self.max_article_length+1))



                if data.iloc[i, 3] == "YES":

                    answer[0, :, 0] = 1

                

                if start_long >= 0:

                    answer[0, 0, start_long+1] = 1

                    answer[0, 0, end_long+1] = 1

                

                if start_short >= 0:

                    answer[0, 1, start_short+1] = 1

                    answer[0, 1, end_short+1] = 1

                

                input_token_ids, segment_ids, mask = tokenize_text(article, question, self.max_article_length)

                input_token_ids_list.append(input_token_ids)

                segment_ids_list.append(segment_ids)

                mask_list.append(mask)

                

                answers.append(answer)

          

            self.input_token_ids = np.array(input_token_ids_list).reshape(-1, self.max_article_length)

            self.segment_ids = np.array(segment_ids_list).reshape(-1, self.max_article_length)

            self.mask = np.array(mask_list).reshape(-1, self.max_article_length)

            self.answers = np.array(answers)

    

        

    def summary(self):

        n = self.data.shape[0]

        

        no_long_no_short = 0

        no_long = 0

        no_short = 0

        short_and_long = 0

                

        for i in range(n):

            if self.data.iloc[i, :]['long_start'] == -1 and self.data.iloc[i, :]['short_start'] == -1:

                no_long_no_short += 1

            elif self.data.iloc[i, :]['long_start'] == -1 and self.data.iloc[i, :]['short_start'] != -1:

                no_long += 1

            elif self.data.iloc[i, :]['long_start'] != -1 and self.data.iloc[i, :]['short_start'] == -1:

                no_short += 1

            else:

                short_and_long += 1

            

        print("Yes/No distribution: ", self.data['yes_no'].value_counts())

        print("No short and no long: ", no_long_no_short / n)

        print("No short but long: ", no_short / n)

        print("No long but short: ", no_long / n)

        print("Short and long: ", short_and_long / n)

        

    

    def __getitem__(self, index):

        return self.input_token_ids[index, :], self.segment_ids[index, :], self.mask[index, :], self.answers[index, :, :]



    def __len__(self):

        return self.input_token_ids.shape[0]

        

                
a = WikiArticles(train_path, 100000)

a.download_data(10)

a.data.head()

a.process()

a.data
a[4]
class BERT_QA:

    def __init__(self, config, num_hidden):

        super(BERTQA, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.linear = nn.Linear(config.hidden_size, num_hidden)

        self.tanh = nn.Tanh()

        

    def forward(self):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        print("After BERT: ", pooled_output.shape)

        pooled_output = self.dropout(pooled_output)

        med1 = self.linear(pooled_output)

        print("Atfer FC: ", med.shape)

        output = self.tahn(med)

        

        return output

    

    def freeze_bert_encoder(self):

        for param in self.bert.parameters():

            param.requires_grad = False

    

    def unfreeze_bert_encoder(self):

        for param in self.bert.parameters():

            param.requires_grad = True

        
import pandas as pd

sample_submission = pd.read_csv("../input/tensorflow2-question-answering/sample_submission.csv")