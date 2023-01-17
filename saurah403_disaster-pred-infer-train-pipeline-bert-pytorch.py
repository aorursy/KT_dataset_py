!pip install -qq transformers
import torch 
import transformers
from transformers import BertTokenizer, AdamW, BertModel, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

%matplotlib inline

%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
# Remove the mislabelled tweets
incorrect_labels_df = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)
incorrect_labels_df = incorrect_labels_df[incorrect_labels_df['target'] > 1]
incorrect_texts = incorrect_labels_df.index.tolist()
train = train[~train.text.isin(incorrect_texts)]

# Add the keyword column to the text column
train['keyword'].fillna('', inplace=True)
train['final_text'] = train['keyword'] + ' ' + train['text'] 
test['keyword'].fillna('', inplace=True)
test['final_text'] = test['keyword'] + ' ' + test['text'] 
train.head()
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
token_len = []

for txt in test.final_text:
    tokens = tokenizer.encode(txt, max_length=512)
    token_len.append(len(tokens))
sns.distplot(token_len)
plt.xlim([0,160])
plt.xlabel('tokencount')
MAX_LEN=160

df_train, df_val = train_test_split(
  train,
  test_size=0.2,
  random_state=RANDOM_SEED
)
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

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
    reviews=df.text.to_numpy(),
    targets=df.target.to_numpy(),
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
data = next(iter(val_data_loader))
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

class DisasterClassifier(nn.Module):
    
    def __init__(self,n_classes):
        super(DisasterClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
            
    def forward(self, input_ids, attention_mask):
   
        _,pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        
        output = self.drop(pooled_output)
        return  self.out(output)
model = DisasterClassifier(2)
model = model.to(device)

EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias = False)
total_steps = len(train_data_loader)* EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch( model, dataloader, loss_fn, optimizer, device, scheduler,n_examples):
    
    model = model.train()
    
    losses =[]
    correct_pred = 0
    
    for d in dataloader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        
        outputs = model(
            input_ids =input_ids,
            attention_mask = attention_mask
        )
        
        _,pred = torch.max(outputs , dim=1)
        
        loss = loss_fn(outputs, targets)
        
        correct_pred += torch.sum(pred == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
       
    return correct_pred.double() / n_examples, np.mean(losses)
def train_eval( model, dataloader, loss_fn, device,n_examples):
    
    model = model.eval()
    
    losses =[]
    correct_pred = 0
    
    with torch.no_grad():
        
        for d in dataloader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
        
        
            outputs = model(
                input_ids =input_ids,
                attention_mask = attention_mask
                )
        
            _,pred = torch.max(outputs , dim=1)
        
            loss = loss_fn(outputs, targets)
        
            correct_pred += torch.sum(pred == targets)
            
            losses.append(loss.item())
            
    return correct_pred.double()/ n_examples, np.mean(losses)
%%time

history = defaultdict(list)
best_acc =0

for epoch in range(EPOCHS):
    
    
    print(f'epochs {epoch+1}/{EPOCHS}')
          
    
    train_acc, train_loss = train_epoch(model,train_data_loader,loss_fn, optimizer, device,scheduler, len(df_train))      
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
          
          
    val_acc, val_loss = train_eval(model,val_data_loader,loss_fn,device,len(df_val)  )
          
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
          
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
          
    if val_acc > best_acc:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_acc = val_acc

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);
class DisasterTestDataset(Dataset):
    def __init__(self, tweets, tokenizer, max_len):
        self.tweets = tweets
        self.tokenizer= tokenizer
        self.max_len =max_len
        
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, item):
        
            
        tweet = str(self.tweets[item])
         
                
    
        encoding = self.tokenizer.encode_plus(
                    tweet,
                    add_special_tokens=True,
                    max_len = self.max_len,
                    return_token_type_ids =False,
                    pad_to_max_length = True,
                    return_attention_mask=True,
                    return_tensors='pt',          
                )
        
        return {
                'tweets':tweet,
                'input_ids':encoding['input_ids'].flatten(),
                'attention_mask':encoding['attention_mask'].flatten(),
                
        }
        
        
    

def create_Testdata_loader(df, tokenizer, max_len, batch_size):
    
    ds = DisasterTestDataset(
         tweets=df.text.to_numpy(),
         tokenizer=tokenizer,
         max_len=max_len
      )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
          )
BATCH_SIZE = 16
test_data_loader = create_Testdata_loader(test, tokenizer, MAX_LEN, BATCH_SIZE)

def get_predictions(model, data_loader):
    
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["tweets"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            
            outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                      )
            
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
   
    return review_texts, predictions, prediction_probs

y_review_texts, y_pred, y_pred_probs = get_predictions(
  model,
  test_data_loader
)
print(y_pred_probs[:10])
print(y_pred[:10])
#print(classification_report(y_test, y_pred, target_names=class_names))
submission = pd.DataFrame()
submission['id'] = test['id']
submission['target'] = y_pred
submission.to_csv('submission12.csv', index=False)