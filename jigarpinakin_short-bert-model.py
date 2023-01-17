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
dataset_path = '../input/reviews/reviews.csv' #--> works in save version

# dataset_path ='../input/shorts_review.csv'
df = pd.read_csv(dataset_path)

df.head()

df.shape
def to_sentiment(rating):

  rating = int(rating)

  if rating <= 2:

    return 0

  elif rating == 3:

    return 1

  else:

    return 2
df['sentiment'] = df.score.apply(to_sentiment)

class_names = ['negative', 'neutral', 'positive']

ax = sns.countplot(df.sentiment)

plt.xlabel('review sentiment')

ax.set_xticklabels(class_names);


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
tokens = tokenizer.tokenize(sample_txt)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')

print(f'   Tokens: {tokens}')

print(f'Token IDs: {token_ids}')
print(tokenizer.sep_token, tokenizer.sep_token_id)

print(tokenizer.cls_token, tokenizer.cls_token_id)

print(tokenizer.pad_token, tokenizer.pad_token_id)

print(tokenizer.unk_token, tokenizer.unk_token_id)



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
tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
token_lens = []

for txt in df.content:

  tokens = tokenizer.encode(txt, max_length=512)

  token_lens.append(len(tokens))
MAX_LEN = 160
class GPReviewDataset(Dataset):

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

      'review_text': review,

      'input_ids': encoding['input_ids'].flatten(),

      'attention_mask': encoding['attention_mask'].flatten(),

      'targets': torch.tensor(target, dtype=torch.long)

    }
df_train, df_test = train_test_split(

  df,

  test_size=0.4,

  random_state=21

)

df_val, df_test = train_test_split(

  df_test,

  test_size=0.5,

  random_state=21

)
df_train.shape, df_val.shape, df_test.shape
def create_data_loader(df, tokenizer, max_len, batch_size):

  ds = GPReviewDataset(

    reviews=df.content.to_numpy(),

    targets=df.sentiment.to_numpy(),

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
data = next(iter(train_data_loader))

data.keys()
print(data['input_ids'].shape)

print(data['attention_mask'].shape)

print(data['targets'].shape)
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

last_hidden_state, pooled_output = bert_model(

  input_ids=encoding['input_ids'],

  attention_mask=encoding['attention_mask']

)
last_hidden_state.shape
bert_model.config.hidden_size
pooled_output.shape
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
model = SentimentClassifier(len(class_names))
input_ids = data['input_ids']

attention_mask = data['attention_mask']

print(input_ids.shape) # batch size x seq length

print(attention_mask.shape) # batch size x seq length
EPOCHS = 3

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(

  optimizer,

  num_warmup_steps=0,

  num_training_steps=total_steps

)

loss_fn = nn.CrossEntropyLoss()
def train_epoch(

  model,

  data_loader,

  loss_fn,

  optimizer,

  scheduler,

  n_examples

):

  model = model.train()

  losses = []

  correct_predictions = 0

  for d in data_loader:

    input_ids = d["input_ids"]

    attention_mask = d["attention_mask"]

    targets = d["targets"]

    outputs = model(

      input_ids=input_ids,

      attention_mask=attention_mask

    )

    _, preds = torch.max(outputs, dim=1)

    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)

    losses.append(loss.item())

    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    scheduler.step()

    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)
def eval_model(model, data_loader, loss_fn, n_examples):

  model = model.eval()

  losses = []

  correct_predictions = 0

  with torch.no_grad():

    for d in data_loader:

      input_ids = d["input_ids"]

      attention_mask = d["attention_mask"]

      targets = d["targets"]

      outputs = model(

        input_ids=input_ids,

        attention_mask=attention_mask

      )

      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)

      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)
history = defaultdict(list)

best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')

  print('-' * 10)

  train_acc, train_loss = train_epoch(

    model,

    train_data_loader,

    loss_fn,

    optimizer,

    scheduler,

    len(df_train)

  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(

    model,

    val_data_loader,

    loss_fn,

    len(df_val)

  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')

  print()

  history['train_acc'].append(train_acc)

  history['train_loss'].append(train_loss)

  history['val_acc'].append(val_acc)

  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:

    torch.save(model.state_dict(), '../kaggle/working/proper_best_model_state.bin') #--> works in save version

#     torch.save(model.state_dict(), 'short_best_model_state.bin')

    best_accuracy = val_acc

    print("Saved")
def get_predictions(model, data_loader):

  model = model.eval()

  review_texts = []

  predictions = []

  prediction_probs = []

  real_values = []

  with torch.no_grad():

    for d in data_loader:

      texts = d["review_text"]

      input_ids = d["input_ids"]

      attention_mask = d["attention_mask"]

      targets = d["targets"]

      outputs = model(

        input_ids=input_ids,

        attention_mask=attention_mask

      )

      _, preds = torch.max(outputs, dim=1)

      review_texts.extend(texts)

      predictions.extend(preds)

      prediction_probs.extend(outputs)

      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()

  prediction_probs = torch.stack(prediction_probs).cpu()

  real_values = torch.stack(real_values).cpu()

  return review_texts, predictions, prediction_probs, real_values
print(type(df_train))
review_text = "I study in KJ Somaiya college of enginnering. I wish my college supports year long internship. But still my college is tire 1 and I am okay wot hit."
encoded_review = tokenizer.encode_plus(

  review_text,

  max_length=MAX_LEN,

  add_special_tokens=True,

  return_token_type_ids=False,

  pad_to_max_length=True,

  return_attention_mask=True,

  return_tensors='pt',

)
new_model = SentimentClassifier(3)
# new_model.load_state_dict(torch.load('short_best_model_state.bin')) --> works in normal run

new_model.load_state_dict(torch.load('../kaggle/output/kaggle/working/proper_best_model_state.bin'))
new_model.eval()
input_ids_short = encoded_review['input_ids']

attention_mask_short = encoded_review['attention_mask']

output = new_model(input_ids_short, attention_mask_short)

_, prediction = torch.max(output, dim=1)

print(f'Review text: {review_text}')

print(f'Sentiment  : {class_names[prediction]}')
import os



os.chdir('../output')

ls
os.chdir('../kaggle/output/')

ls
os.chdir('../kaggle/working/')

ls
os.chdir('../output/kaggle/working/')

ls
os.chdir('../input/output/')

ls