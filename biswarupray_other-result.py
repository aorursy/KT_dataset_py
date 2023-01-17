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
import transformers

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch



import numpy as np

import pandas as pd

import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib import rc

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict

from textwrap import wrap



from torch import nn, optim

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F



%matplotlib inline

%config InlineBackend.figure_format='retina'
df=pd.read_csv("/kaggle/input/avishek-hotel/sentiments.csv",encoding='ISO-8859–1',header=0)

df.head()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
df.loc[ df['review'] ==1, 'review'] = 2

df.loc[ df['review'] ==0, 'review'] = 1

df.loc[ df['review'] ==-1, 'review'] = 0
df
PRE_TRAINED_MODEL_NAME='bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
tokens = tokenizer.tokenize(sample_txt)

token_ids = tokenizer.convert_tokens_to_ids(tokens)



print(f' Sentence: {sample_txt}')

print(f'   Tokens: {tokens}')

print(f'Token IDs: {token_ids}')
encoding = tokenizer.encode_plus(

  sample_txt,

  max_length=32,

  add_special_tokens=True, # Add '[CLS]' and '[SEP]'

  return_token_type_ids=False,

  pad_to_max_length=True,

  return_attention_mask=True,

  return_tensors='pt',  # Return PyTorch tensors

)



encoding.keys()
class hotelDataset(Dataset):



  def __init__(self, reviews, targets, tokenizer, max_len):

    self.reviews = reviews

    self.targets = targets

    self.tokenizer = tokenizer

    self.max_len = max_len

  

  def __len__(self):

    return len(self.reviews)

  

  def __getitem__(self, item):

    review = str(self.reviews[item])

    target = self.targets[item]



    encoding = self.tokenizer.encode_plus(

      review,

      add_special_tokens=True,

      max_length=self.max_len,

      return_token_type_ids=False,

      pad_to_max_length=True,

      return_attention_mask=True,

      return_tensors='pt',

    )



    return {

      'text': review,

      'input_ids': encoding['input_ids'].flatten(),

      'attention_mask': encoding['attention_mask'].flatten(),

      'targets': torch.tensor(target, dtype=torch.long)

    }
MAX_LEN = 160
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
df_train, df_val, df_test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.9*len(df))])
def create_data_loader(df, tokenizer, max_len, batch_size):

  ds = hotelDataset(

    reviews=df.text.to_numpy(),

    targets=df.review.to_numpy(),

    tokenizer=tokenizer,

    max_len=max_len

  )



  return DataLoader(

    ds,

    batch_size=batch_size,

    num_workers=4

  )
BATCH_SIZE = 16



train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)

val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
class SentimentClassifier(nn.Module):



  def __init__(self, n_classes):

    super(SentimentClassifier, self).__init__()

    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self.drop = nn.Dropout(p=0.3)

    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  

  def forward(self, input_ids, attention_mask):

    _, pooled_output = self.bert(

      input_ids=input_ids,

      attention_mask=attention_mask

    )

    output = self.drop(pooled_output)

    return self.out(output)
last_hidden_state, pooled_output = bert_model(

  input_ids=encoding['input_ids'], 

  attention_mask=encoding['attention_mask']

)
class_names = ['negative', 'neutral', 'positive']
model = SentimentClassifier(len(class_names))

model = model.to(device)
model.load_state_dict(torch.load("../input/bert-other/best_model_state.bin"))

model.eval()
def eval_model(model, data_loader, loss_fn, device, n_examples):

  model = model.eval()



  losses = []

  correct_predictions = 0



  with torch.no_grad():

    for d in data_loader:

      input_ids = d["input_ids"].to(device)

      attention_mask = d["attention_mask"].to(device)

      targets = d["targets"].to(device)



      outputs = model(

        input_ids=input_ids,

        attention_mask=attention_mask

      )

      _, preds = torch.max(outputs, dim=1)



      loss = loss_fn(outputs, targets)



      correct_predictions += torch.sum(preds == targets)

      losses.append(loss.item())



  return correct_predictions.double() / n_examples, np.mean(losses)
loss_fn = nn.CrossEntropyLoss().to(device)
test_acc, _ = eval_model(

  model,

  test_data_loader,

  loss_fn,

  device,

  len(df_test)

)



test_acc.item()
def get_predictions(model, data_loader):

  model = model.eval()

  

  review_texts = []

  predictions = []

  prediction_probs = []

  real_values = []



  with torch.no_grad():

    for d in data_loader:



      texts = d["text"]

      input_ids = d["input_ids"].to(device)

      attention_mask = d["attention_mask"].to(device)

      targets = d["targets"].to(device)



      outputs = model(

        input_ids=input_ids,

        attention_mask=attention_mask

      )

      _, preds = torch.max(outputs, dim=1)



      probs = F.softmax(outputs, dim=1)



      review_texts.extend(texts)

      predictions.extend(preds)

      prediction_probs.extend(probs)

      real_values.extend(targets)



  predictions = torch.stack(predictions).cpu()

  prediction_probs = torch.stack(prediction_probs).cpu()

  real_values = torch.stack(real_values).cpu()

  return review_texts, predictions, prediction_probs, real_values
y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(

  model,

  test_data_loader

)
print(classification_report(y_test, y_pred, target_names=class_names))