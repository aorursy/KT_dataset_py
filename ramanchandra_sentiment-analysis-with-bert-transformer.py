#Check installed version of packages
%reload_ext watermark
%watermark -v -p numpy,pandas,torch,transformers
#Import required libraries 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rc
from pylab import rcParams
import matplotlib.pyplot as plt
from textwrap import wrap
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve,auc


import transformers
from transformers import BertModel, BertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn  
import torch.nn.functional as F  
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid',palette='muted',font_scale=1.2)
color_palette=['#01BEFE','#FFDD00','#FF7D00','#FF006D','#ADFF02','#8F00FF']
sns.set_palette(sns.color_palette(color_palette))

rcParams['figure.figsize']= 12,6

import warnings
warnings.filterwarnings('ignore')

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
!nvidia-smi
df=pd.read_csv('../input/reviews-for-top-30-apps-in-india-in-play-store/App_reviews.csv')
df.head()
df.shape
print(f'There are {df.shape[0]} reviews in the dataset')
df.info()
sns.countplot(df.score)
plt.xlabel('Review score')
def to_sentiment(score):
  score=int(score)
  if score <=4:
    return 0
  else :
    return 1

df['sentiment']=df.score.apply(to_sentiment)
df.head()
ax=sns.countplot(df.sentiment)
plt.xlabel('Review sentiment')
class_names=['Negative','Positive']
ax.set_xticklabels(class_names)
plt.show()
Pre_trained_model='bert-base-uncased'
tokenizer=BertTokenizerFast.from_pretrained(Pre_trained_model);
sample_text="The animal didn't cross the street because it was too tired"

#Convert text to tokens & token_ids
tokens=tokenizer.tokenize(sample_text)
token_ids=tokenizer.convert_tokens_to_ids(tokens)

print(f'Sentence : {sample_text}')
print(f'Tokens :{tokens}')
print(f'Token IDs : {token_ids}')
encoding=tokenizer.encode_plus(
    sample_text,
    max_length=32,
    add_special_tokens=True,   # 'Add [SEP] & [CLS]'
    pad_to_max_length=True,
    truncation=True,
    return_attention_mask=True,  # Reurns array of 0's & 1's to distinguish padded tokens from real tokens.
    return_token_type_ids=False,
    return_tensors='pt'         # Returns pytorch tensors
)

encoding.keys()
# Check input_ids
print('Maximum length of input_ids for each sentence : {}'.format(len(encoding['input_ids'][0])))
encoding['input_ids'][0]

#check attention mask
print(f"Maximum length of attention mask for each sentence : {len(encoding['attention_mask'][0])}")
encoding['attention_mask'][0]
special_tokens=tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
special_tokens
token_lens=[]
for content in df.content:
  tokens_content=tokenizer.encode(content,max_length=150,truncation=True)
  token_lens.append(len(tokens_content))
#Plot the tokens
sns.distplot(token_lens)
plt.xlim([0,150])
plt.xlabel('Token count')
Max_length=100
class reviews_India_Dataset(Dataset):

  def __init__(self,reviews,targets,tokenizer,max_length):
    self.reviews=reviews
    self.targets=targets
    self.tokenizer=tokenizer
    self.max_length=max_length

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self,item):
    review = str(self.reviews[item])
    targets = self.targets[item]

    encoding = self.tokenizer.encode_plus(
        review,
        max_length=Max_length,
        add_special_tokens=True,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
       )
    return {
        'review_text':review,
        'input_ids':encoding['input_ids'].flatten(),
        'attention_mask':encoding['attention_mask'].flatten(),
        'targets' : torch.tensor(targets,dtype=torch.long)
    }
df_train,df_test=train_test_split(df, test_size=0.2, random_state=42)
df_valid,df_test = train_test_split(df_test,test_size=0.5,random_state=42)

print('Print the shape of datasets...')
print(f'Training dataset : {df_train.shape}')
print(f'Testing dataset : {df_test.shape}')
print(f'Validation dataset : {df_valid.shape}')
batch_size=32
def data_loader(df, tokenizer, max_length, batch):
  ds=reviews_India_Dataset(
      reviews=df.content.to_numpy(),
      targets=df.sentiment.to_numpy(),
      tokenizer=tokenizer,
      max_length=Max_length
  )

  return DataLoader(
      ds,
      batch_size=batch_size,
      num_workers=4
  )

# Load datasets
train_DataLoader=data_loader(df_train,tokenizer,Max_length,batch_size)
test_DataLoader=data_loader(df_test,tokenizer,Max_length,batch_size)
valid_DataLoader=data_loader(df_valid,tokenizer,Max_length,batch_size)
data=next(iter(train_DataLoader))
data.keys()
print('Shape of the data keys...')
print(f"Input_ids : {data['input_ids'].shape}")
print(f"Attention_mask : {data['attention_mask'].shape}")
print(f"targets : {data['targets'].shape}")
bert_model = BertModel.from_pretrained(Pre_trained_model)
class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(Pre_trained_model)
    self.drop = nn.Dropout(p=0.5)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)
model = SentimentClassifier(len(class_names))         #Create an instance / object
model = model.to(device)                              # Move instance to GPU           
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
print(input_ids.shape)      # batch size x seq length
print(attention_mask.shape) # batch size x seq length
F.softmax(model(input_ids,attention_mask), dim=1)
epochs=5
optimizer=AdamW(model.parameters(),lr=2e-5,correct_bias=False)
total_steps=len(train_DataLoader)*epochs

scheduler=get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn=nn.CrossEntropyLoss().to(device)
def train(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_observations
):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    #Feed data to BERT model
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)     # Clip gradients to avoid exploding gradient problem
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_observations, np.mean(losses)
def eval_model(model, data_loader,device,loss_fn, n_observations):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      # Feed data to BERT model
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_observations, np.mean(losses)
%%time
history = defaultdict(list)
best_accuracy = 0
for epoch in range(epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)
  train_acc, train_loss = train(
    model,
    train_DataLoader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
    model,
    valid_DataLoader,
    device,
    loss_fn,
    len(df_valid)
  )
  print(f'Validation  loss {val_loss} accuracy {val_acc}')
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
test_acc, _ = eval_model(
  model,
  test_DataLoader,
  device,
  loss_fn,
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
      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
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
y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_DataLoader
)
class_report=classification_report(y_test, y_pred, target_names=class_names)
print(class_report)
def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(),rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(),rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment');
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='BERT')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()
#Calculate AUC_score for PR curve
auc_score = auc(recall, precision)
print('PR AUC_score: %.3f' % auc_score)
review_text=y_review_texts[1]
true_sentiment=y_test[1]
pred_df=pd.DataFrame({
    'class_names':class_names,
    'values':y_pred_probs[1]
})

print('\n'.join(wrap(review_text)))
print()
print(f'True Sentiment : {class_names[true_sentiment]}')
sns.barplot(x='values',y='class_names',data=pred_df,orient='h')
plt.xlabel('Probability')
plt.ylabel('Sentiment')
plt.xlim([0,1]);
review_text='Fake Fake Fake Fake Fake, Please be aware of them, they are very dangerous people please be aware of them. \
They ask your pan and aadhara number for interview, the ppl who calls are unprofessional they talk rubbish and try to trap you in pit'
encoded_review=tokenizer.encode_plus(
    review_text,
    max_length=Max_length,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)
input_ids=encoded_review['input_ids'].to(device)
attention_mask=encoded_review['attention_mask'].to(device)

output=model(input_ids,attention_mask)
_,pred=torch.max(output,dim=1)

print(f'Review_text : {review_text}')
print(f'Sentiment: {class_names[pred]}')
path="./Sentiment_Analysis_Bert.bin"
torch.save(model.state_dict(),path)