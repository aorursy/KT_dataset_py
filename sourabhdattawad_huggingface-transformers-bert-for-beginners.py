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
import numpy as np

import pandas as pd

import torch

from tqdm import tqdm



from transformers import BertTokenizer, BertModel





MODEL_TYPE = 'bert-base-uncased'

MAX_SIZE = 150

BATCH_SIZE = 200
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

model = BertModel.from_pretrained(MODEL_TYPE)
tokenized_input = train_df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print(tokenized_input[1])

print("Here 101 -> [CLS] and 102 -> [SEP]")
padded_tokenized_input = np.array([i + [0]*(MAX_SIZE-len(i)) for i in tokenized_input.values])
print(padded_tokenized_input[0])
attention_masks  = np.where(padded_tokenized_input != 0, 1, 0)
print(attention_masks[0])
input_ids = torch.tensor(padded_tokenized_input)  

attention_masks = torch.tensor(attention_masks)
all_train_embedding = []



with torch.no_grad():

  for i in tqdm(range(0,len(input_ids),200)):    

    last_hidden_states = model(input_ids[i:min(i+200,len(train_df))], attention_mask = attention_masks[i:min(i+200,len(train_df))])[0][:,0,:].numpy()

    all_train_embedding.append(last_hidden_states)

unbatched_train = []

for batch in all_train_embedding:

    for seq in batch:

        unbatched_train.append(seq)



train_labels = train_df['target']
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test =  train_test_split(unbatched_train, train_labels, test_size=0.33, random_state=42, stratify=train_labels)

len(X_train)