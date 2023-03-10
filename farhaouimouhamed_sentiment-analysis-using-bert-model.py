import numpy as np

import pandas as pd

import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

import matplotlib.pyplot as rc

import torch

from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from transformers import BertModel, BertConfig, BertTokenizer

from torch import nn

from torch.utils import data

from torch.utils.data import Dataset, DataLoader

device = 'cuda'

%matplotlib inline

%config InlineBackend.figure_format='retina'



sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE","#FFDD00","#FF7D00","#FF006D","#ADFF02","#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

np.random.seed(42)

torch.manual_seed(42)
df = pd.read_csv("../input/sentiment-analysis-datasetgoogle-play-app-reviews/sentiment-analysis-dataset-google-play-app-reviews.csv")

df.head()
df.shape
df.info()
sns.countplot(df.score)

plt.xlabel('review score');
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
token_lens = []

for txt in df.content:

    tokens = tokenizer.encode(txt, max_length=512, truncation=True)

    token_lens.append(len(tokens))
sns.distplot(token_lens)

plt.xlim([0, 256]);

plt.xlabel('Token count');
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

          truncation=True,

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
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)
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

model = model.to(device)
EPOCHS = 50



optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_data_loader) * EPOCHS



scheduler = get_linear_schedule_with_warmup(

  optimizer,

  num_warmup_steps=0,

  num_training_steps=total_steps

)



loss_fn = nn.CrossEntropyLoss().to(device)
def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):

    model = model.train()



    losses = []

    correct_predictions = 0

  

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

    

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()



    return correct_predictions.double() / n_examples, np.mean(losses)
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
%%time

from collections import defaultdict

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

        device, 

        scheduler, 

        len(df_train)

      )



    print(f'Train loss {train_loss} accuracy {train_acc}')



    val_acc, val_loss = eval_model(

        model,

        val_data_loader,

        loss_fn, 

        device, 

        len(df_val)

      )



    print(f'Val   loss {val_loss} accuracy {val_acc}')

    print()



    history['train_acc'].append(train_acc)

    history['train_loss'].append(train_loss)

    history['val_acc'].append(val_acc)

    history['val_loss'].append(val_loss)



    if val_acc > best_accuracy:

        torch.save(model.state_dict(), 'best_model_state.bin')

        best_accuracy = val_acc
plt.plot(history['train_acc'], label='train accuracy')

plt.plot(history['val_acc'], label='validation accuracy')



plt.title('Training history')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend()

plt.ylim([0, 1]);