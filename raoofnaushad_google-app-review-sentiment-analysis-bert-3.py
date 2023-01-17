!pip install transformers
import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange





import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import BertPreTrainedModel, BertModel, get_linear_schedule_with_warmup, AdamW



from transformers import AutoConfig, AutoTokenizer

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
MODEL_OUT_DIR = '/kaggle/working/my_model'



DATA = pd.read_csv('../input/google-app-review-dataset/reviews.csv')



## Model Configurations

MAX_LEN_TRAIN = 160

MAX_LEN_VALID = 160

BATCH_SIZE = 16

LR = 2e-5

NUM_EPOCHS = 3 #10 ####################################################################################### Change it to 10

NUM_THREADS = 4  ## Number of threads for collecting dataset

MODEL_NAME = 'bert-base-uncased'





if not os.path.isdir(MODEL_OUT_DIR):

    os.makedirs(MODEL_OUT_DIR)





def to_sentiment(rating):

  rating = int(rating)

  if rating <= 2:

    return 0

  elif rating == 3:

    return 1

  else:

    return 2

DATA['sentiment'] = DATA.score.apply(to_sentiment)

sns.countplot(DATA.sentiment)

plt.xlabel('review score');

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(DATA, test_size=0.1, random_state=RANDOM_SEED)

valid_df, test_df = train_test_split(DATA, test_size=0.5, random_state=RANDOM_SEED)
train_df.reset_index(inplace=True)

valid_df.reset_index(inplace=True)

test_df.reset_index(inplace=True)
train_df.head()
class GoogleDataset(Dataset):



    def __init__(self, file, maxlen, tokenizer): 

        #Store the contents of the file in a pandas dataframe

        # self.df = pd.read_csv(filename)

        self.df = file

        #Initialize the tokenizer for the desired transformer model

        self.tokenizer = tokenizer

        #Maximum length of the tokens list to keep all the sequences of fixed size

        self.maxlen = maxlen



    def __len__(self):

        return len(self.df)



    def __getitem__(self, index):    

        #Select the sentence and label at the specified index in the data frame

        sentence = self.df.loc[index, 'content']

        label = self.df.loc[index, 'sentiment']

        #Preprocess the text to be suitable for the transformer

        tokens = self.tokenizer.tokenize((sentence)) 

        encoding = self.tokenizer.encode_plus(

          sentence,

          add_special_tokens=True,

          max_length=self.maxlen,

          return_token_type_ids=False,

          pad_to_max_length=True,

          return_attention_mask=True,

          truncation = True,

          return_tensors='pt',

        )

        input_ids = encoding['input_ids'].flatten()

        attention_mask = encoding['attention_mask'].flatten()

        label = torch.tensor(label, dtype=torch.long)

        

        return input_ids, attention_mask, label
## Tokenizer loaded from AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

## Training Dataset

train_set = GoogleDataset(file=train_df, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer)

valid_set = GoogleDataset(file=valid_df, maxlen=MAX_LEN_VALID, tokenizer=tokenizer)

test_set = GoogleDataset(file=test_df, maxlen=MAX_LEN_VALID, tokenizer=tokenizer)
## Data Loaders

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)



print(len(train_loader))

class BertForSentimentClassification(BertPreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(p=0.3)

        #The classification layer that takes the [CLS] representation and outputs the logit

        self.cls_layer = nn.Linear(config.hidden_size, 3)



    def forward(self, input_ids, attention_mask):

        #Feed the input to Bert model to obtain contextualized representations

        reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        #Obtain the representations of [CLS] heads

        cls_reps = reps[:, 0]

        cls_reps = self.dropout(cls_reps)

        logits = self.cls_layer(cls_reps)

        return logits

## Configuration loaded from AutoConfig 

config = AutoConfig.from_pretrained(MODEL_NAME)

## Creating the model from the desired transformer model

model = BertForSentimentClassification.from_pretrained(MODEL_NAME, config=config)

## GPU or CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Putting model to device

model = model.to(device)
## Experimenting

input_ids, attention_mask, label = next(iter(train_loader))

input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)

print(input_ids.shape) # batch size x seq length

print(attention_mask.shape) # batch size x seq length

output = model(input_ids, attention_mask)

F.softmax(model(input_ids, attention_mask), dim=1)
criterion = nn.CrossEntropyLoss().to(device)

## Optimizer

optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)

total_steps = len(train_loader) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(

  optimizer,

  num_warmup_steps=0,

  num_training_steps=total_steps

)
def train(model, criterion, optimizer, train_loader, val_loader, epochs, scheduler):

    best_acc = 0

    for epoch in trange(epochs, desc="Epoch"):

        model.train()

        train_acc = 0

        for i, (input_ids, attention_mask, labels) in enumerate(tqdm(iterable=train_loader, desc="Training")):

            # optimizer.zero_grad()  

            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(logits, labels)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

            train_acc += get_accuracy_from_logits(logits, labels)

        print(f"Training accuracy is {train_acc/len(train_loader)}")

        val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)

        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))

        if val_acc > best_acc:

            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))

            best_acc = val_acc

            model.save_pretrained(save_directory=MODEL_OUT_DIR + '/')

            config.save_pretrained(save_directory=MODEL_OUT_DIR + '/')

            tokenizer.save_pretrained(save_directory=MODEL_OUT_DIR + '/')



## Accuracy Function

def get_accuracy_from_logits(logits, labels):

    probs = F.softmax(logits, dim=1)

    output = torch.argmax(probs, dim=1)

    acc = (output == labels).float().mean()

    return acc
def evaluate(model, criterion, dataloader, device, test=False):

    model.eval()

    mean_acc, mean_loss, count = 0, 0, 0

    total_acc = 0

    with torch.no_grad():

        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):

        # for input_ids, attention_mask, labels in (dataloader):

            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = model(input_ids, attention_mask)

            mean_loss += criterion(logits.squeeze(-1), labels).item()

            mean_acc += get_accuracy_from_logits(logits, labels)

            count += 1

            if test == True:

              total_acc += mean_acc/count

        if test == True:

          return total_acc/count

    return mean_acc / count, mean_loss / count
train(model=model, 

      criterion=criterion,

      optimizer=optimizer, 

      train_loader=train_loader,

      val_loader=valid_loader,

      epochs = NUM_EPOCHS,

      scheduler = scheduler

      )
evaluate(model=model, criterion=criterion, dataloader=test_loader, device=device, test=True)
def classify_sentiment(sentence):

  with torch.no_grad():  

    tokens = tokenizer.tokenize(sentence)

    tokens = ['[CLS]'] + tokens + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = torch.tensor(input_ids).to(device)

    input_ids = input_ids.unsqueeze(0)

    attention_mask = (input_ids != 0).long()

    attention_mask = attention_mask.to(device)

    if len(tokens) < MAX_LEN_VALID:

        tokens = tokens + ['[PAD]' for _ in range(MAX_LEN_VALID - len(tokens))] 

    else:

        tokens = tokens[:self.maxlen-1] + ['[SEP]'] 



    print(input_ids.shape)

    logit = model(input_ids=input_ids, attention_mask=attention_mask)

    prob = F.softmax(logit, dim=1)

    output = torch.argmax(prob)

    prob = prob[0][output]

    if output == 0:

        print('Negative {}'.format(int(prob*100)))

    elif output == 1:

        print('Neutral {}'.format(int(prob*100)))

    elif output == 2:

        print('Positive {}'.format(int(prob*100)))
sentence = "Hope this notebook was informative and useful"

classify_sentiment(sentence)