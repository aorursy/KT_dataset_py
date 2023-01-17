import os

import pandas as pd

from tqdm import tqdm, trange



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from transformers import BertPreTrainedModel, BertModel



from transformers import AutoConfig, AutoTokenizer
MODEL_OUT_DIR = '/kaggle/working/models/my_model'

TRAIN_FILE_PATH = '../input/data/train.tsv'

VALID_FILE_PATH = '../input/data/dev.tsv'

## Model Configurations

MAX_LEN_TRAIN = 30

MAX_LEN_VALID = 30

BATCH_SIZE = 32

LR = 2e-5

NUM_EPOCHS = 2

NUM_THREADS = 1  ## Number of threads for collecting dataset

MODEL_NAME = 'bert-base-uncased'





if not os.path.isdir(MODEL_OUT_DIR):

    os.makedirs(MODEL_OUT_DIR)

class SSTDataset(Dataset):



    def __init__(self, filename, maxlen, tokenizer): 

        #Store the contents of the file in a pandas dataframe

        self.df = pd.read_csv(filename, delimiter = '\t')

        #Initialize the tokenizer for the desired transformer model

        self.tokenizer = tokenizer

        #Maximum length of the tokens list to keep all the sequences of fixed size

        self.maxlen = maxlen



    def __len__(self):

        return len(self.df)



    def __getitem__(self, index):    

        #Select the sentence and label at the specified index in the data frame

        sentence = self.df.loc[index, 'sentence']

        label = self.df.loc[index, 'label']

        #Preprocess the text to be suitable for the transformer

        tokens = self.tokenizer.tokenize(sentence) 

        tokens = ['[CLS]'] + tokens + ['[SEP]'] 

        if len(tokens) < self.maxlen:

            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 

        else:

            tokens = tokens[:self.maxlen-1] + ['[SEP]'] 

        #Obtain the indices of the tokens in the BERT Vocabulary

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 

        input_ids = torch.tensor(input_ids) 

        #Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones

        attention_mask = (input_ids != 0).long()

        return input_ids, attention_mask, label
class BertForSentimentClassification(BertPreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.bert = BertModel(config)

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #The classification layer that takes the [CLS] representation and outputs the logit

        self.cls_layer = nn.Linear(config.hidden_size, 1)



    def forward(self, input_ids, attention_mask):

        #Feed the input to Bert model to obtain contextualized representations

        reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        #Obtain the representations of [CLS] heads

        cls_reps = reps[:, 0]

        # cls_reps = self.dropout(cls_reps)

        logits = self.cls_layer(cls_reps)

        return logits

def train(model, criterion, optimizer, train_loader, val_loader, epochs):

    best_acc = 0

    for epoch in trange(epochs, desc="Epoch"):

        model.train()

        for i, (input_ids, attention_mask, labels) in enumerate(tqdm(iterable=train_loader, desc="Training")):

            optimizer.zero_grad()  

            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(input=logits.squeeze(-1), target=labels.float())

            loss.backward()

            optimizer.step()

        val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)

        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))

        if val_acc > best_acc:

            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))

            best_acc = val_acc

            model.save_pretrained(save_directory=MODEL_OUT_DIR + '/')

            config.save_pretrained(save_directory=MODEL_OUT_DIR + '/')

            tokenizer.save_pretrained(save_directory=MODEL_OUT_DIR + '/')

def get_accuracy_from_logits(logits, labels):

    #Get a tensor of shape [B, 1, 1] with probabilities that the sentiment is positive

    probs = torch.sigmoid(logits.unsqueeze(-1))

    #Convert probabilities to predictions, 1 being positive and 0 being negative

    soft_probs = (probs > 0.5).long()

    #Check which predictions are the same as the ground truth and calculate the accuracy

    acc = (soft_probs.squeeze() == labels).float().mean()

    return acc



def evaluate(model, criterion, dataloader, device):

    model.eval()

    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():

        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):

            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = model(input_ids, attention_mask)

            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()

            mean_acc += get_accuracy_from_logits(logits, labels)

            count += 1

    return mean_acc / count, mean_loss / count
## Configuration loaded from AutoConfig 

config = AutoConfig.from_pretrained(MODEL_NAME)

## Tokenizer loaded from AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

## Creating the model from the desired transformer model

model = BertForSentimentClassification.from_pretrained(MODEL_NAME, config=config)

## GPU or CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Putting model to device

model = model.to(device)

## Takes as the input the logits of the positive class and computes the binary cross-entropy 

criterion = nn.BCEWithLogitsLoss()

## Optimizer

optimizer = optim.Adam(params=model.parameters(), lr=LR)

## Training Dataset

train_set = SSTDataset(filename=TRAIN_FILE_PATH, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer)

valid_set = SSTDataset(filename=VALID_FILE_PATH, maxlen=MAX_LEN_VALID, tokenizer=tokenizer)
## Data Loaders

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
train(model=model, 

      criterion=criterion,

      optimizer=optimizer, 

      train_loader=train_loader,

      val_loader=valid_loader,

      epochs = NUM_EPOCHS

      )
def classify_sentiment(sentence):

    with torch.no_grad():

        tokens = tokenizer.tokenize(sentence)

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor(input_ids).to(device)

        input_ids = input_ids.unsqueeze(0)

        attention_mask = (input_ids != 0).long()

        attention_mask = attention_mask.to(device)

        logit = model(input_ids=input_ids, attention_mask=attention_mask)

        prob = torch.sigmoid(logit.unsqueeze(-1))

        prob = prob.item()

        soft_prob = prob > 0.5

        if soft_prob == 1:

            print('Positive with probability {}%.'.format(int(prob*100)))

        else:

            print('Negative with probability {}%.'.format(int(100-prob*100)))
sentence = 'it is organised '

classify_sentiment(sentence)