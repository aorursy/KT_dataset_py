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
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

import torch # importing torch before transformers causes errors



from tqdm import tqdm, tqdm_notebook
test_df = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
pretrained_weights = 'bert-base-uncased'



config = BertConfig.from_pretrained(pretrained_weights, num_labels=2)

model = BertForSequenceClassification.from_pretrained(pretrained_weights, config=config)

tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)
MAX_LENGTH = 320

BATCH_SIZE = 24
def encode_text(texts):

    

    # encoding

    X = [tokenizer.encode(text, add_special_tokens=True, max_length=MAX_LENGTH) 

         for text in tqdm_notebook(texts)]

           

    # padding

    X = [x + [0 for _ in range(MAX_LENGTH-len(x))] for x in X]            

    

    return X



test_X = encode_text(test_df['question_text'])
device = torch.device('cuda')

model = model.to(device)
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_X, dtype=torch.long))

test_sampler = torch.utils.data.SequentialSampler(test_dataset)

test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
model_state_dict = torch.load('../input/quora-bert-models/pytorch_model.bin')

model = BertForSequenceClassification.from_pretrained(pretrained_weights, state_dict=model_state_dict)

model.to(device)



preds = None



for batch in tqdm_notebook(test_loader, desc="testing"):

    model.eval()

    batch = tuple(t.to(device) for t in batch)



    with torch.no_grad():

        inputs = {'input_ids': batch[0]}

        outputs = model(**inputs)

        logits = outputs[0]



    if preds is None:

        preds = logits.detach().cpu().numpy()

    else:

        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)



preds = np.argmax(preds, axis=1)



submission = test_df[['qid']]

submission['prediction'] = preds

submission.to_csv('submission.csv', index=False)
np.sum(preds)/len(preds) # a ratio of 1