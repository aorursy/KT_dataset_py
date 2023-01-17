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
from __future__ import absolute_import,division,print_function

import torch.utils.data

from tqdm import tqdm

import warnings

warnings.filterwarnings(action='once')

device = torch.device('cuda')
import sys

package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.append(package_dir)



from pytorch_pretrained_bert import BertTokenizer,BertForSequenceClassification,BertAdam,BertConfig
def convert_lines(example,max_seq_length,tokenizer):

    max_seq_length -= 2

    all_tokens = []

    longer = 0

    for text in tqdm(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer +=1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"] 

                                                   )+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

        

    return np.array(all_tokens)
max_seq_len = 220

seed = 1234

batch_size = 32

bert_model_path = '../input/bert-pytorch-from-yuval/'

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True



bert_config = BertConfig(bert_model_path+'bert_config.json')

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
test_df["content"].fillna("DUMMY_VALUE")
test_df = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")

test_df['content'] = test_df['content'].astype(str) 

X_test = convert_lines(test_df["content"].fillna("DUMMY_VALUE"), max_seq_len, tokenizer)
model = BertForSequenceClassification(bert_config,num_labels=1)

model.load_state_dict(torch.load("../input/bert-pytorch-from-yuval/bert_pytorch.bin"))

model.to(device)

for param in model.parameters():

    param.requires_grad = False

model.eval()
test_preds = np.zeros((len(X_test)))

test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))

test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

tk0 = tqdm(test_loader)

for i, (x_batch,) in enumerate(tk0):

    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

    test_preds[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()



test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()
submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'toxic': test_pred

})

submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()