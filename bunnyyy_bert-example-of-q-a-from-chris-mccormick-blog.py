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
!pip install transformers
import torch

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer
model= BertForQuestionAnswering.from_pretrained('bert-base-uncased')

tokenizer= BertTokenizer.from_pretrained('bert-base-uncased')
question = "How many parameters does BERT-large have?"

answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."
input_ids= tokenizer.encode(question, answer_text)

print('The input has total {:} tokens'.format(len(input_ids)))
sep_index= input_ids.index(tokenizer.sep_token_id)



num_seg_a= sep_index + 1



num_seg_b= len(input_ids) - num_seg_a



segment_ids= [0]*num_seg_a + [1]*num_seg_b



start_scores, end_scores= model(torch.tensor([input_ids]), token_type_ids= torch.tensor([segment_ids]))
tokens= tokenizer.convert_ids_to_tokens(input_ids)
print(input_ids)   ##Hence proved that BERT takes only token ids as input and not the words
print(tokens)
answer_start= torch.argmax(start_scores)

answer_end= torch.argmax(end_scores)



assert answer_start < answer_end  ## Think yourself, its common sense

answer= ' '.join(tokens[answer_start: answer_end +1])

print(answer)