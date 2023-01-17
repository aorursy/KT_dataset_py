# installation block

!pip -q  install transformers > /dev/null



#import block

import pandas as pd

import numpy as np

import datetime

import string

import torch

import time

import re



from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.express as px



import transformers as ppb # pytorch transformers

from transformers import AdamW



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc

import xgboost as xgb



from IPython.display import clear_output

from collections import Counter

from tqdm import tqdm



import nltk

from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stop_words = stopwords.words('english')
def clean_text(text):

    """

    Text preprocessing function.

    """

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')



df_train['clean_text'] = df_train['text'].apply(lambda x:clean_text(x))

df_train['Length'] = df_train.text.str.len()

df_train['Word_count'] = df_train['text'].str.split().map(len)
import plotly.graph_objects as go

from plotly.subplots import make_subplots



import pandas as pd

import re



fig = make_subplots(

    rows=3, cols=1, 

    shared_xaxes=True,

    vertical_spacing=0.03,

    specs=[[{"type": "table"}],

           [{"type": "scatter"}],

           [{"type": "scatter"}]]

)



fig.add_trace(

    go.Scatter(

        x=df_train["id"],

        y=df_train["Length"],

        mode="lines",

        name="Number of letters in a tweet"

    ),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(

        x=df_train["id"],

        y=df_train["Word_count"],

        mode="lines",

        name="Number of words in a tweet"

    ),

    row=2, col=1

)



fig.add_trace(

    go.Table(

        header=dict(

        

            values=[

                'ID' , 'Keyword', 'Location', 'Text', 'Target', 'Clean_text', 'Length', 'Word_count'

            ],

            font=dict(size=10),

            align="left"

        ),

        cells=dict(

            values=[df_train[k].tolist() for k in df_train.columns],

            align = "left")

    ),

    row=1, col=1

)

fig.update_layout(

    height=800,

    showlegend=False,

    title_text="TWEETS DATASET",

)



fig.show()
len_word_dis = len(df_train[df_train.target==1])

len_word_nondis = len(df_train[df_train.target==0])



tweets = ['Disaster' , 'Non-Disaster']

colors = [ '#330C73', '#EF553B']

          

fig = go.Figure([go.Bar(x=tweets, y=[len_word_dis, len_word_nondis], marker_color=colors  )])

fig.update_layout(title_text='Compared count Disaster and Non-Disaster tweets')

fig.show()
fig = make_subplots(

    rows=2, cols=2,

    specs=[

           [{}, {"rowspan": 2}],

           [{}, None]

    ],

    subplot_titles=("Real","Bouth", "Fake"))



tweet_real = df_train.query('target == "1"')['Length']

tweet_fake = df_train.query('target == "0"')['Length']



fig.add_trace(go.Histogram(x = tweet_real, marker_color='#330C73'), row=1, col=1)

fig.add_trace(go.Histogram(x = tweet_real, marker_color='#330C73'), row=1, col=2)

fig.add_trace(go.Histogram(x = tweet_fake, marker_color='#EF553B'), row=1, col=2)

fig.add_trace(go.Histogram(x = tweet_fake, marker_color='#EF553B'), row=2, col=1)



fig.update_layout(showlegend=False, title_text="Histograms of the number of characters in the tweets")

fig.show()
fig = make_subplots(

    rows=2, cols=2,

    specs=[

           [{}, {"rowspan": 2}],

           [{}, None]

    ],

    subplot_titles=("Real","Bouth", "Fake"))



tweet_real = df_train.query('target == "1"')['Word_count']

tweet_fake = df_train.query('target == "0"')['Word_count']



fig.add_trace(go.Histogram(x = tweet_real, marker_color='#330C73'), row=1, col=1)

fig.add_trace(go.Histogram(x = tweet_real, marker_color='#330C73'), row=1, col=2)

fig.add_trace(go.Histogram(x = tweet_fake, marker_color='#EF553B'), row=1, col=2)

fig.add_trace(go.Histogram(x = tweet_fake, marker_color='#EF553B'), row=2, col=1)



fig.update_layout(showlegend=False, title_text="Histograms of the number of words in the tweets")

fig.show()
dis = " ".join(df_train.loc[df_train['target'] == 1]['clean_text']).lower().split()

non_dis = " ".join(df_train.loc[df_train['target'] == 0]['clean_text']).lower().split()



top_dis = Counter([i for i in dis if i not in stop_words])

top_non_dis = Counter([i for i in non_dis if i not in stop_words])



temp_dis = pd.DataFrame(top_dis.most_common(40))

temp_non_dis = pd.DataFrame(top_non_dis.most_common(40))



temp_dis.columns = ['Common_words','count']

temp_non_dis.columns = ['Common_words','count']
fig = px.treemap(temp_non_dis, path=['Common_words'], values='count',title='Popular Non-Disasters Words')

fig.update_layout(treemapcolorway = ["#330C73", "slateblue"])

fig.show()
fig = px.treemap(temp_dis, path=['Common_words'], values='count',title='Popular Disasters Words')

fig.update_layout(treemapcolorway = ["#EF553B", "mistyrose"])

fig.show()
#1

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

""" 

You can try 'bert-large-uncased' model

But the kernel will take longer

""" 

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)



#2

tokenized = df_train['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))



#3

max_len = 0

for i in tokenized.values:

    if len(i) > max_len:

        max_len = len(i)



#4

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)
class Dataset(torch.utils.data.Dataset):

    

  def __init__(self, input_ids, attention_masks):

        self.input_ids = input_ids

        self.attention_mask = attention_masks



  def __len__(self):

        return len(self.input_ids)



  def __getitem__(self, index):

        input_ids = self.input_ids[index]

        attention_mask = self.attention_mask[index]

        return input_ids, attention_mask
input_ids = torch.tensor(padded)  

attention_mask = torch.tensor(attention_mask)



dataset = Dataset(input_ids=input_ids, attention_masks=attention_mask)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=False)
features = []



for batch_input_ids, batch_attention_mask in tqdm(dataloader):

  with torch.no_grad():

    # get embeddings

    tmp = model(batch_input_ids, attention_mask=batch_attention_mask)

    # get [CLS] vectors

    features.extend(tmp[0][:,0,:].numpy())

    clear_output(True)
labels = df_train['target'][:len(features)]

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
# clf = xgb.XGBClassifier()

# parameters = {

#      "eta"    : [0.02] ,

#      "max_depth"        : [ 7], # 5, 7, 9],

#      "min_child_weight" : [ 5], #, 3 , 5],

#      "gamma"            : [ 0.0] ,

#      'n_estimators'     : [1000]

#      }

# grid = GridSearchCV(clf,parameters, n_jobs=4, scoring="accuracy",cv=3)

# grid.fit(pd.DataFrame(train_features),train_labels)

# best_est = grid.best_estimator_

# best_est



clf = GradientBoostingClassifier(random_state=0)

clf.fit(train_features, train_labels)

clf.score(test_features, test_labels)
pred_val = clf.predict_proba(pd.DataFrame(test_features))[:,1]

fpr, tpr, _ = roc_curve(test_labels, pred_val)

roc_auc = auc(fpr, tpr)
lw = 2



trace1 = go.Scatter(x=fpr, y=tpr, 

                    mode='lines', 

                    line=dict(color='darkorange', width=lw),

                    name='ROC curve (area = %0.2f)' % roc_auc

                   )



trace2 = go.Scatter(x=[0, 1], y=[0, 1], 

                    mode='lines', 

                    line=dict(color='navy', width=lw, dash='dash'),

                    showlegend=False)



layout = go.Layout(title='Receiver operating characteristic ',

                   xaxis=dict(title='False Positive Rate'),

                   yaxis=dict(title='True Positive Rate'))



fig = go.Figure(data=[trace1, trace2], layout=layout)



fig.show()
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

test_tokenized = test_df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))



test_max_len = 0

for i in test_tokenized.values:

    if len(i) > test_max_len:

        test_max_len = len(i)



test_padded = np.array([i + [0]*(test_max_len-len(i)) for i in test_tokenized.values])

test_attention_mask = np.where(test_padded != 0, 1, 0)



test_input_ids = torch.tensor(test_padded)  

test_attention_mask = torch.tensor(test_attention_mask)



test_dataset = Dataset(input_ids=test_input_ids, attention_masks=test_attention_mask)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, drop_last=False)
test_features = []



for batch_input_ids, batch_attention_mask in tqdm(test_dataloader):

  with torch.no_grad():

    tmp = model(batch_input_ids, attention_mask=batch_attention_mask)

    test_features.extend(tmp[0][:,0,:].numpy())

    clear_output(True)
predict = clf.predict(pd.DataFrame(test_features))

sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub['target'] = predict

sub.to_csv('sample_submission.csv', index=False)
model_class, tokenizer_class, pretrained_weights = (ppb.BertForSequenceClassification, ppb.BertTokenizer, 'bert-base-cased')# ppb.BertTokenizer, 'distilbert-base-cased') #'bert-large-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)



tokenized = df_train['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))



max_len = 0

for i in tokenized.values:

    if len(i) > max_len:

        max_len = len(i)



padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Dataset(torch.utils.data.Dataset):

  def __init__(self, input_ids, attention_masks, labels):

        self.input_ids = input_ids

        self.attention_mask = attention_masks

        self.labels = labels



  def __len__(self):

        return len(self.input_ids)



  def __getitem__(self, index):

        input_ids = self.input_ids[index]

        attention_mask = self.attention_mask[index]

        labels = self.labels[index]

        return input_ids, attention_mask, labels
train_input_ids = torch.tensor(padded[:7000])  

train_attention_mask = torch.tensor(attention_mask[:7000])

train_labels = list(df_train['target'][:7000])



validation_input_ids = torch.tensor(padded[7000:7613])  

validation_attention_mask = torch.tensor(attention_mask[7000:7613])

validation_labels = list(df_train['target'][7000:7613])



dataset = Dataset(input_ids=train_input_ids, attention_masks=train_attention_mask, labels=train_labels)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=False)



validation_dataset = Dataset(input_ids=validation_input_ids, attention_masks=validation_attention_mask, labels=validation_labels)

validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=8, drop_last=False)
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8 )

from transformers import get_linear_schedule_with_warmup



epochs = 2

total_steps = len(dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 

                                            num_warmup_steps = 0, # Default value in run_glue.py

                                            num_training_steps = total_steps)
def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))
import random

seed_val = 42

random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)

loss_values = []

for epoch_i in range(0, epochs):

    

    # ========================================

    #               Training

    # ========================================

    

    # Perform one full pass over the training set.

    print("")

    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    print('Training...')

    t0 = time.time()

    total_loss = 0

    model.train()

    for step, batch in enumerate(dataloader):

        if step % 40 == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, 

                    token_type_ids=None, 

                    attention_mask=b_input_mask, 

                    labels=b_labels)

        loss = outputs[0]

        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_loss / len(dataloader)            

    loss_values.append(avg_train_loss)

    print("")

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        

    # ========================================

    #               Validation

    # ========================================

    print("")

    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0

    nb_eval_steps, nb_eval_examples = 0, 0

    

    for batch in validation_dataloader: 

        

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        

        with torch.no_grad():        

            outputs = model(b_input_ids, 

                            token_type_ids=None, 

                            attention_mask=b_input_mask)



        logits = outputs[0]

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()

        

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy



        nb_eval_steps += 1



    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")

print("Training complete!")
class Dataset(torch.utils.data.Dataset):

  def __init__(self, input_ids, attention_masks):

        self.input_ids = input_ids

        self.attention_mask = attention_masks



  def __len__(self):

        return len(self.input_ids)



  def __getitem__(self, index):

        input_ids = self.input_ids[index]

        attention_mask = self.attention_mask[index]

        return input_ids, attention_mask
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

test_tokenized = test_df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))



test_max_len = 0

for i in test_tokenized.values:

    if len(i) > test_max_len:

        test_max_len = len(i)



test_padded = np.array([i + [0]*(test_max_len-len(i)) for i in test_tokenized.values])

test_attention_mask = np.where(test_padded != 0, 1, 0)



test_input_ids = torch.tensor(test_padded)  

test_attention_mask = torch.tensor(test_attention_mask)



test_dataset = Dataset(input_ids=test_input_ids, attention_masks=test_attention_mask)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, drop_last=False)
from IPython.display import clear_output



model.eval()

test_features = []

preds = []



for batch_input_ids, batch_attention_mask in tqdm(test_dataloader):

  with torch.no_grad():



    logits = model(batch_input_ids, attention_mask=batch_attention_mask)



    a = logits[0]

    a = (np.argmax(a.numpy(),axis=1))



    logits = outputs[0]

    logits = logits.detach().cpu().numpy()





    pred_flat = np.argmax(logits, axis=1).flatten()

    preds.extend(a)

    clear_output(True)
sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub['target'] = preds

sub.to_csv('sample_submission2.csv', index=False)