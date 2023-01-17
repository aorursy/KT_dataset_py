import pandas as pd
import transformers
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import inspect
import torch.nn as nn
import torch
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')
BATCH_SIZE = 16
MAX_LEN = 60
EPOCHS = 1
DIR = '../input/tweet-sentiment-extraction/'
train = pd.read_csv(DIR+'train.csv')
#test = pd.read_csv(DIR+'train.csv')
#ss = pd.read_csv(DIR+'sample_submission.csv')
train.head()
sns.countplot(train.sentiment)
train.sentiment[0]
mapping = {'positive':2,'negative':0,'neutral':1}
train.replace({'sentiment':mapping},inplace=True)
##test.replace({'sentiment':mapping},inplace=True)
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
class TweetReviewDataset(Dataset):
    def __init__(self,review,tokenizer,targets,max_len):
        self.review = review
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_len = max_len
        
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self,index):
        review = str(self.review[index])
        target = self.targets[index]
        
        encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
        )
        
        return {'review_text':review,'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }
df_train , df_val = train_test_split(train,test_size = 0.2,random_state = 23)
del train
gc.collect()
def create_data_loader(df, tokenizer, max_len, batch_size):
    
    ds = TweetReviewDataset(
    review=df.text.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0
  )
train_len = len(df_train)
val_len = len(df_val)

train_loader = create_data_loader(df_train,tokenizer,MAX_LEN,BATCH_SIZE)
val_loader = create_data_loader(df_val,tokenizer,MAX_LEN,BATCH_SIZE)
del df_train
del df_val

gc.collect()
class SentimentClassifier(nn.Module):
    def __init__(self,n_classes):
        super(SentimentClassifier,self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size,n_classes)
        
    def forward(self,input_ids,attention_mask):
        _,pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
        
        output = self.drop(pooled_output)
        return self.out(output)
model = SentimentClassifier(3)
optimizer = transformers.AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS

scheduler = transformers.get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()
def train_epoch(model,data_loader,loss_fn,optimizer,scheduler,n_examples):
    model = model.train()
    
    losses = []
    correct_predictions = 0
    
    for i,d in tqdm(enumerate(data_loader)):
        
        input_ids = d['input_ids']
        attention_mask = d['attention_mask']
        targets = d["targets"]
        
        outputs = model(input_ids = input_ids,attention_mask = attention_mask)
        
        _,preds = torch.max(outputs,dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)
        
        
def eval_model(model, data_loader,loss_fn, n_examples):
    
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        
        for i,d in tqdm(enumerate(data_loader)):
            
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

for epoch in range(EPOCHS):
    
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    scheduler,
    train_len
  )
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
    model,
    val_loader,
    loss_fn,
    val_len
  )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
PATH = 'model.pth'
torch.save(model,PATH)
