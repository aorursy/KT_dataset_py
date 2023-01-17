import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import re
# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")[["text","target"]]
df_train.head()
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_username(text):
    url = re.compile(r'@[A-Za-z0-9_]+')
    return url.sub(r'',text)
def pre_process_text(text):
    text = remove_URL(text)
    text = remove_numbers(text)
    text = remove_html(text)
    text = remove_username(text)
    return " ".join(text.split())
pre_process_text("Why can't gay men donate blood? http://t.co/v2Etl8P9eQ http://t.co/NLnyzeljbw")
# longest tweet length
max(df_train['text'].apply(len))
# Defining some key variables that will be used later on in the training
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-05

BERT_PATH = '/kaggle/input/distillbert-huggingface-model/'
MODEL_PATH = "distilbert-base-uncased-pytorch_model.bin"
tokenizer = DistilBertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
class tweet_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        tweet = str(self.data.text[index])
        tweet = pre_process_text(tweet)
        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.target[index], dtype=torch.float)
        }
        
    def __len__(self):
        return len(self.data)
# Creating the dataset and dataloader for the neural network

train_size = 0.85
train_dataset=df_train.sample(frac=train_size,random_state=200).reset_index(drop=True)
valid_dataset=df_train.drop(train_dataset.index).reset_index(drop=True)


print("FULL Dataset: {}".format(df_train.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VALID Dataset: {}".format(valid_dataset.shape))

training_set = tweet_Dataset(train_dataset, tokenizer, MAX_LEN)
validation_set = tweet_Dataset(valid_dataset, tokenizer, MAX_LEN)
training_set[0]
train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

train_dl = DataLoader(training_set, **train_params)
valid_dl = DataLoader(validation_set, **valid_params)
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = transformers.DistilBertModel.from_pretrained(BERT_PATH)
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.out(output_1)
        return output
model = DistillBERTClass()
model.to(device)
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
def eval_fn(data_loader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        fin_outputs = np.array(fin_outputs) >= 0.5
        f1 = metrics.f1_score(fin_targets, fin_outputs)
    return f1
def fit(num_epochs, model, loss_fn, opt, train_dl, valid_dl):
    
    for epoch in range(num_epochs):
        model.train()
        for _,data in enumerate(train_dl, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask).squeeze()
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()
            opt.zero_grad()

        valid_acc = eval_fn(valid_dl, model)
        print('Epoch [{}/{}], Train Loss: {:.4f} and Validation f1 {:.4f}'.format(epoch+1, num_epochs, loss.item(),valid_acc))
fit(7, model, loss_fn, optimizer, train_dl,valid_dl)
def sentence_prediction(sentence):
    max_len = MAX_LEN
    tweet = str(sentence)
    tweet = " ".join(tweet.split())
    inputs = tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
        )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]


    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)

    outputs = model(ids=ids, mask=mask)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0] > 0.5
sentence_prediction("We are experiencing slight tremors in London right now")
sentence_prediction("Lol this movie is an absolute disaster!")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")[["id","text"]]
df_test.head()
df_test['target'] = df_test['text'].apply(sentence_prediction)
df_test['target'] = df_test['target'].astype(int)
df_test.head()
# submission
df_test.to_csv("submission.csv", columns = ['id','target'], index=False)