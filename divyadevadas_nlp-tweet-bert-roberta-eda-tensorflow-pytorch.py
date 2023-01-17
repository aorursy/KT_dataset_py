#!pip install pyspellchecker
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # Library for string operations

import os

# plotly library
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.figure_factory as ff

import matplotlib.pyplot as plt #Another plotting libraray

# word cloud library
from wordcloud import WordCloud

#Regex library
import re

#Spell Checker
#from spellchecker import SpellChecker 
#spell = SpellChecker()
TrainDataSet= pd.read_csv('../input/nlp-getting-started/train.csv')
TestDataSet=pd.read_csv('../input/nlp-getting-started/test.csv')
TrainDataSet.head(3)


Grouped_Disaster = TrainDataSet.groupby(['target'])['id'].count().reset_index()
labels = ['Disaster','Non-Disaster']


# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=Grouped_Disaster['id'], hole=.4)])
fig.update_layout(height=400,title_x=0.5,width=400, title_text='Disaster Tweet Percentage',
                 annotations=[dict(text='#tweet', x=0.5, y=0.5, font_size=20, showarrow=False)])

plotly.offline.iplot(fig)
TrainDataSet['tweetlength'] = TrainDataSet['text'].apply(lambda x:  len(str(x)))
TrainDataSet.head(5)
Top_20_Lengthy_Tweets = TrainDataSet.sort_values('tweetlength',ascending=False)[:20][::-1]
Bottom_20_Lengthy_Tweets = TrainDataSet.sort_values('tweetlength',ascending=True)[:20][::-1]
Tweetlength_Data =TrainDataSet['tweetlength'].describe()


fig = make_subplots(
    rows=3, cols=4,
    specs=[[None,{"type": "indicator"},{"type": "indicator"},{"type": "indicator"}],
           [{"type": "bar" ,"colspan": 2},None, {"type": "bar" ,"colspan": 2},None],
           [{"type": "bar","colspan": 4}, None,None,None]],
    subplot_titles=("","","","Top 20 Tweets by length","Bottom 20 Tweet by length")
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Tweetlength_Data[1],
        title="Mean Tweet Length",
    ),
    row=1, col=2
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Tweetlength_Data[3],
        title="Min Tweet Length",
    ),
    row=1, col=3
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Tweetlength_Data[7],
        title="Max Tweet Length",
    ),
    row=1, col=4
)


fig.add_trace(go.Bar(name='id',text='id', x=list(range(len(Top_20_Lengthy_Tweets))), y=Top_20_Lengthy_Tweets['tweetlength']),
              row=2, col=1)


fig.add_trace(go.Bar(name='id',text='id', x=list(range(len(Bottom_20_Lengthy_Tweets))), y=Bottom_20_Lengthy_Tweets['tweetlength']),
              row=2, col=3)
fig.add_trace(go.Bar(name='id',text='id',  x=list(range(len(TrainDataSet.head(500)))), y=TrainDataSet['tweetlength']),
              row=3, col=1)

fig.update_layout(height=650,width=800,title_x=0.5,title_text="Tweets Length Analysis", showlegend=False)

plotly.offline.iplot(fig)


# x=list(range(len(TrainDataSet)))
TrainDataSet['wordcount'] = TrainDataSet['text'].apply(lambda x:  len(str(x).split()))
TrainDataSet.head(5)
Top_20_Lengthy_Tweets = TrainDataSet.sort_values('wordcount',ascending=False)[:20][::-1]
Bottom_20_Lengthy_Tweets = TrainDataSet.sort_values('wordcount',ascending=True)[:20][::-1]
Wordlength_Data =TrainDataSet['wordcount'].describe()


fig = make_subplots(
    rows=3, cols=4,
    specs=[[None,{"type": "indicator"},{"type": "indicator"},{"type": "indicator"}],
           [{"type": "bar" ,"colspan": 2},None, {"type": "bar" ,"colspan": 2},None],
           [{"type": "bar","colspan": 4}, None,None,None]],
    subplot_titles=("","","","Top 20 Tweets word count","Bottom 20 Tweet by word count")
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[1],
        title="Mean Tweet word count",
    ),
    row=1, col=2
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[3],
        title="Min Tweet word count",
    ),
    row=1, col=3
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[7],
        title="Max Tweet word count",
    ),
    row=1, col=4
)


fig.add_trace(go.Bar(name='id',text='id', x=Top_20_Lengthy_Tweets['keyword'], y=Top_20_Lengthy_Tweets['wordcount']),
              row=2, col=1)


fig.add_trace(go.Bar(name='id',text='id', x=Bottom_20_Lengthy_Tweets['keyword'], y=Bottom_20_Lengthy_Tweets['wordcount']),
              row=2, col=3)
fig.add_trace(go.Bar(name='id',text='id', x=TrainDataSet['keyword'], y=TrainDataSet['wordcount']),
              row=3, col=1)

fig.update_layout(height=600,width=800,title_x=0.5, title_text="Tweets Word Count Analysis", showlegend=False)

plotly.offline.iplot(fig)

import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

wordcollection=[]
TempTextCol= TrainDataSet['text'].str.split()
TempTextCol=TempTextCol.values.tolist()
wordcollection=[word for i in TempTextCol for word in i]

from collections import defaultdict
stopwprddic=defaultdict(int)
for word in wordcollection:
    if word in stop:
        stopwprddic[word]+=1

stopwprddf =  pd.DataFrame(stopwprddic.items(), columns=['word', 'count'])

stopwprddf.head()
Top_20_Lengthy_Tweets = stopwprddf.sort_values('count',ascending=False)[:20][::-1]
Bottom_20_Lengthy_Tweets = stopwprddf.sort_values('count',ascending=True)[:20][::-1]
Wordlength_Data =stopwprddf['count'].describe()


fig = make_subplots(
    rows=3, cols=4,
    specs=[[None,{"type": "indicator"},{"type": "indicator"},{"type": "indicator"}],
           [{"type": "bar" ,"colspan": 2},None, {"type": "bar" ,"colspan": 2},None],
           [{"type": "bar","colspan": 4}, None,None,None]],
    subplot_titles=("","","","Top 20 Tweets stopword count","Bottom 20 Tweet by stopword count")
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[1],
        title="Mean stopwords",
    ),
    row=1, col=2
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[3],
        title="Min stopwords",
    ),
    row=1, col=3
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[7],
        title="Max Tweet stopwords",
    ),
    row=1, col=4
)


fig.add_trace(go.Bar(name='count',text='count', x=Top_20_Lengthy_Tweets['word'], y=Top_20_Lengthy_Tweets['count']),
              row=2, col=1)


fig.add_trace(go.Bar(name='count',text='count', x=Bottom_20_Lengthy_Tweets['word'], y=Bottom_20_Lengthy_Tweets['count']),
              row=2, col=3)
fig.add_trace(go.Bar(name='count',text='count', x=stopwprddf['word'], y=stopwprddf['count']),
              row=3, col=1)

fig.update_layout(height=500,width=800,title_text="Tweets StopWord Analysis", title_x=0.5, showlegend=False)

plotly.offline.iplot(fig)

import spacy
nlp = spacy.load('en')

from collections import defaultdict
dicspy=defaultdict(int)


docs = TrainDataSet['text'].tolist()

def token_filter(token):
    return (token.is_punct | token.is_space )

filtered_tokens = []
for doc in nlp.pipe(docs):
    tokens = [token.lemma_ for token in doc if token_filter(token)]
    filtered_tokens.append(tokens)
    for tk in tokens:
        dicspy[tk]+=1

Punctuationdf =  pd.DataFrame(dicspy.items(), columns=['word', 'count'])

Punctuationdf.head()
Top_20_Lengthy_Tweets = Punctuationdf.sort_values('count',ascending=False)[:20][::-1]
Bottom_20_Lengthy_Tweets = Punctuationdf.sort_values('count',ascending=True)[:20][::-1]
Wordlength_Data =Punctuationdf['count'].describe()


fig = make_subplots(
    rows=3, cols=4,
    specs=[[None,{"type": "indicator"},{"type": "indicator"},{"type": "indicator"}],
           [{"type": "bar" ,"colspan": 2},None, {"type": "bar" ,"colspan": 2},None],
           [{"type": "bar","colspan": 4}, None,None,None]],
    subplot_titles=("","","","Top 20 Tweets Puntuation or Space count","Bottom 20 Tweet by Puntuation or Space count")
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[1],
        title="Mean ",
    ),
    row=1, col=2
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[3],
        title="Min",
    ),
    row=1, col=3
)

fig.add_trace(
    go.Indicator(
        mode="number",
        value=Wordlength_Data[7],
        title="Max",
    ),
    row=1, col=4
)


fig.add_trace(go.Bar(name='count',text='count', x=Top_20_Lengthy_Tweets['word'], y=Top_20_Lengthy_Tweets['count']),
              row=2, col=1)


fig.add_trace(go.Bar(name='count',text='count', x=Bottom_20_Lengthy_Tweets['word'], y=Bottom_20_Lengthy_Tweets['count']),
              row=2, col=3)
fig.add_trace(go.Bar(name='count',text='count', x=Punctuationdf['word'], y=Punctuationdf['count']),
              row=3, col=1)

fig.update_layout(height=500,width=800,title_text="Tweets Punctuation and Space Analysis", title_x=0.5, showlegend=False)

plotly.offline.iplot(fig)

from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
def getNgram(wordCollection, n=None):
    vectorData = CountVectorizer(ngram_range=(n, n)).fit(wordCollection)
    BagOfWords = vectorData.transform(wordCollection)
    SumWords = BagOfWords.sum(axis=0) 
    WordsFq = [(word, SumWords[0, idx]) 
                  for word, idx in vectorData.vocabulary_.items()]
    WordsFq =sorted(WordsFq, key = lambda x: x[1], reverse=True)
    return WordsFq[:10]

getBigrams=getNgram(TrainDataSet['text'],2)[:10]
x,y=map(list,zip(*getBigrams))

import plotly.express as px
fig = px.bar(x=y,y=x)
fig.update_layout(height=500,width=600,title_text="Ngram Analysis", title_x=0.5, showlegend=False)
plotly.offline.iplot(fig)

triGrams=getNgram(TrainDataSet['text'],n=3)
x,y=map(list,zip(*triGrams))
fig = px.bar(x=y,y=x)
fig.update_layout(height=500,width=600,title_text="TriGram Analysis", title_x=0.5, showlegend=False)
plotly.offline.iplot(fig)
TrainDataSet['text'] = TrainDataSet['text'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(x)))
TestDataSet['text'] = TestDataSet['text'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(x)))
TrainDataSet.head(3)

TrainDataSet['text'] = TrainDataSet['text'].apply(lambda x: re.sub(r'#', '', str(x)))
TestDataSet['text'] = TestDataSet['text'].apply(lambda x: re.sub(r'#', '', str(x)))
TrainDataSet.head(3)



TrainDataSet['text'] = TrainDataSet['text'].apply(lambda x: re.sub(r'#', '', str(x)))
TestDataSet['text'] = TestDataSet['text'].apply(lambda x: re.sub(r'#', '', str(x)))
TrainDataSet.head(3)


def EmojiCleanser(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)
TestDataSet['text'] = TestDataSet['text'].apply(lambda x: EmojiCleanser(str(x)))
TrainDataSet['text'] = TrainDataSet['text'].apply(lambda x: EmojiCleanser(str(x)))
TrainDataSet.tail(3)
TestDataSet['text'] = TestDataSet['text'].apply(lambda x: re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', str(x)))
TrainDataSet['text'] = TrainDataSet['text'].apply(lambda x: re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', str(x)))
TrainDataSet.tail(3)
TestDataSet['text'] = TestDataSet['text'].apply(lambda x: str(x).translate(str.maketrans('','',string.punctuation)))
TrainDataSet['text'] = TrainDataSet['text'].apply(lambda x: str(x).translate(str.maketrans('','',string.punctuation)))
TrainDataSet.tail(3)

#TestDataSet['text'] = TestDataSet['text'].apply(lambda x: " ".join([spell.correction(i) for i in str(x).split()]))
#TrainDataSet['text'] = TrainDataSet['text'].apply(lambda x: " ".join([spell.correction(i) for i in str(x).split()]))
#TrainDataSet.tail(3)

import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

def StopWordCleanser(word):
    if word in stop:
        return ""
    else:
        return word


TestDataSet['text'] = TestDataSet['text'].apply(lambda x: " ".join([StopWordCleanser(i) for i in str(x).split()]))
TrainDataSet['text'] = TrainDataSet['text'].apply(lambda x: " ".join([StopWordCleanser(i) for i in str(x).split()]))


TrainDataSet.head(20)

TrainDataSet = TrainDataSet[TrainDataSet.keyword.notnull()]
TrainDataSet = TrainDataSet[TrainDataSet.location.notnull()]


Grouped_Disaster = TrainDataSet.groupby(['keyword'])['id'].count().reset_index()
Grouped_Location = TrainDataSet.groupby(['location'])['id'].count().reset_index()

Grouped_Disaster = Grouped_Disaster.query('keyword !="Not Identified"' )
Grouped_Location = Grouped_Location.query('location !="Not Location"' )

Group_Disaster_filter = Grouped_Disaster.sort_values('id',ascending=False)[:20][::-1]
Grouped_Location_filter = Grouped_Location.sort_values('id',ascending=False)[:20][::-1]

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "bar"}, {"type": "bar"}]],
    subplot_titles=("Top 20 Disaster by Tweets","Top 20 Tweet Location")
)

fig.add_trace(go.Bar(name='id',text='id', x=Group_Disaster_filter['keyword'], y=Group_Disaster_filter['id']),
              row=1, col=1)


fig.add_trace(go.Bar(name='id',text='id', x=Grouped_Location_filter['location'], y=Grouped_Location_filter['id']),
              row=1, col=2)

fig.update_layout(height=500,width=600,title_text="Tweets Breakdown", title_x=0.5, showlegend=False)
plotly.offline.iplot(fig)


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
text = TrainDataSet.text.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

# If there's a GPU available...

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.  
    
    device = torch.device('cuda')    


    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')
DisastweetTokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
Disastweets = pd.concat([TrainDataSet, TestDataSet])
Disastweets = Disastweets.text.values

print('Tokenized: ', DisastweetTokenizer.tokenize(Disastweets[0]))
print('Token IDs: ', DisastweetTokenizer.convert_tokens_to_ids(DisastweetTokenizer.tokenize(Disastweets[0])))
def MapTokens(tweet,labs='None'):
    
    """A function for tokenize all of the sentences and map the tokens to their word IDs."""
    
    global labels
    
    Tokenids = []
    Textmasks = []

    # For every sentence...
    
    for text in tweet:
        #   "encode_plus" will:
        
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        
        encoded_dict = DisastweetTokenizer.encode_plus(
                            text,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            truncation='longest_first', # Activate and control truncation
                            max_length = 84,           # Max length according to our text data.
                            pad_to_max_length = True, # Pad & truncate all sentences.
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the id list. 
        
        Tokenids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        
        Textmasks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    
    Tokenids = torch.cat(Tokenids, dim=0)
    Textmasks = torch.cat(Textmasks, dim=0)
    if labs != 'None': 
        labels = torch.tensor(labs)
    
    return Tokenids,Textmasks,labels
        


Train_Tokenids, Train_Masks, Train_Labels = MapTokens(TrainDataSet['text'].values, TrainDataSet['target'].values)
Test_Tokenids, Test_Masks,Test_Labels = MapTokens(TestDataSet['text'].values)
TweetTensorDataset = TensorDataset(Train_Tokenids, Train_Masks, Train_Labels)
Predict_TweetTensorDataset = TensorDataset(Test_Tokenids, Test_Masks)
Train_TweetTensorDataset, Validation_TweetTensorDataset = random_split(TweetTensorDataset, [int(0.8 * len(TweetTensorDataset)), (len(TweetTensorDataset) - (int(0.8 * len(TweetTensorDataset))))])
batch_size = 32
# Train
BERT_Train_Loader = DataLoader(
            TweetTensorDataset,  # Train Data.
            sampler = RandomSampler(TweetTensorDataset), # Random Batch
            batch_size = batch_size 
        )

# validation

BERT_Validation_Loader = DataLoader(
            Validation_TweetTensorDataset, # Validation Data.
            sampler = SequentialSampler(Validation_TweetTensorDataset), # Sequential Batch.
            batch_size = batch_size # Evaluate with this batch size.
        )

# Prediction.

BERT_Predict_Loader = DataLoader(
            Predict_TweetTensorDataset, # Prediction Data.
            sampler = SequentialSampler(Predict_TweetTensorDataset), # Sequential Batch.
            batch_size = batch_size 
        )

model = BertForSequenceClassification.from_pretrained(
    'bert-large-uncased', 
    num_labels = 2, # binary classification   
    output_attentions = False,
    output_hidden_states = False, 
)

# Tell pytorch to run this model.

model.to(device)
BERT_Tweet_Optimizer = AdamW(model.parameters(),
                  lr = 6e-6, # args.learning_rate
                  eps = 1e-8 # args.adam_epsilon
                )
# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

Linear_Scheduler = get_linear_schedule_with_warmup(BERT_Tweet_Optimizer, 
                                            num_warmup_steps = 0, #The number of steps for the warmup phase
                                            num_training_steps = len(BERT_Train_Loader) * epochs) # The index of the last epoch when resuming training


import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []
BERT_train_predictions = []
BERT_true_labels = []
# Measure the total training time for the whole run.
total_t0 = time.time()

print("Training Started")

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('...')
    
     
    t0 = time.time() # Measure how long the training epoch takes.
    total_train_loss = 0     # Reset the total loss for this epoch.
    
    model.train() # Set the mode and iterate using the dataloader
    
    for step, batch in enumerate(BERT_Train_Loader):
        
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(BERT_Train_Loader), elapsed))

        # Read three pytorch tensors (input ids, attention masks, labels) in each Batch Step
        b_input_ids = batch[0].to(device).to(torch.int64)
        b_input_mask = batch[1].to(device).to(torch.int64)
        b_labels = batch[2].to(device).to(torch.int64)
        
        model.zero_grad()# Clear any previously calculated gradients        
        # Evaluate the model on this training batch
          
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
               
        total_train_loss += loss.item() # Accumulate the training loss  
        
        loss.backward()# Perform a backward pass to calculate the gradients.
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip the norm of the gradients to 1.0 -'exploding gradients' problem.
        
        BERT_Tweet_Optimizer.step() # Move to Next Step
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        BERT_train_predictions.append(logits)
        BERT_true_labels.append(label_ids)
        Linear_Scheduler.step()

    # Calculate the average loss over all of the batches.
    
        # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(BERT_Train_Loader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
      
    
    print(avg_train_loss)
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
        }
    )
        
print('')
print('Training complete!')   
from sklearn.metrics import f1_score, accuracy_score
def flat_accuracy(preds, labels):
    
    """A function for calculating accuracy scores"""
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return accuracy_score(labels_flat, pred_flat)

def flat_f1(preds, labels):
    
    """A function for calculating f1 scores"""
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return f1_score(labels_flat, pred_flat)

validation_stats = []

predictions = []
true_labels = []
print('')
print('Start Validation...')

# Measure the total training time for the whole run.
total_t0 = time.time()

for epoch_i in range(0, 4):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print("")
    t0 = time.time() # Measure how long the training epoch takes.
    total_train_loss = 0     # Reset the total loss for this epoch.
    
    model.eval() # Set evaluation mode
    
    # Tracking variables:
    total_eval_accuracy = 0
    total_eval_loss = 0
    total_eval_f1 = 0
    nb_eval_steps = 0
    
    for batch in BERT_Validation_Loader:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
                    
        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(BERT_Train_Loader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Disable constructing the compute graph.
        with torch.no_grad():
              (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        total_eval_loss += loss.item() # Accumulate the validation loss.

        # Move logits and labels to CPU:
 
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches:
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        total_eval_f1 += flat_f1(logits, label_ids)
    
       
    validation_time = format_time(time.time() - t0)  # Measure how long the validation run took.
    avg_val_accuracy = total_eval_accuracy / len(BERT_Validation_Loader)
    print("Accuracy : " )
    print(avg_val_accuracy)
    avg_val_f1 = total_eval_f1 / len(BERT_Validation_Loader)  
    print("F1 score : " )
    print(avg_val_f1)
    avg_val_loss = total_eval_loss / len(BERT_Validation_Loader)
    print("Validation Loss:" )
    print(avg_val_loss)
    
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    predictions.append(logits)
    true_labels.append(label_ids)
    validation_stats.append(
        {
            'epoch': epoch_i + 1,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Val_F1' : avg_val_f1,
            'Validation Time': validation_time
        }
    )

print('')
print('Validation complete!')


from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_labels)):
    # The predictions for this batch are a 2-column ndarray (one column for "0"
    # and one column for "1"). Pick the label with the highest value and turn this
    # in to a list of 0s and 1s.
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
    # Calculate and store the coef for this batch.
    matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
    matthews_set.append(matthews)
    
# Create a barplot showing the MCC score for each batch of test samples.
ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

plt.title('MCC Score per Batch')
plt.ylabel('MCC Score (-1 to +1)')
plt.xlabel('Batch #')

plt.show()
# Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import matthews_corrcoef
matthews_set = []
for i in range(len(true_labels)):
  matthews = matthews_corrcoef(true_labels[i],
                 np.argmax(predictions[i], axis=1).flatten())
  matthews_set.append(matthews)
  
# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

print('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.format(matthews_corrcoef(flat_true_labels, flat_predictions)))



print('Starting Prediction.')
# Put model in evaluation mode:

model.eval()

# Tracking variables :

predictions = []

# Predict:

for batch in BERT_Predict_Loader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)

print('DONE.')
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
submission['target'] = flat_predictions
submission.head(10)
Grouped_Disaster = submission.groupby(['target'])['id'].count().reset_index()
labels = ['Disaster','Non-Disaster']


# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=Grouped_Disaster['id'], hole=.4)])
fig.update_layout(width=600, height=400,title_text='BERT- Predicted Disaster Tweet Percentage',
                 annotations=[dict(text='#tweet', x=0.5, y=0.5, font_size=20, showarrow=False)])
plotly.offline.iplot(fig)
submission.to_csv("submission.csv", index=False, header=True)
print('Starting Prediction.')
# Put model in evaluation mode:

model.eval()

# Tracking variables :

predictions = []

# Predict:

for batch in BERT_Train_Loader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)

print('DONE.')
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
def plot_cm(y_true, y_pred, title, figsize=(5,5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    
plot_cm(flat_predictions, TrainDataSet['target'].values, 'Confusion matrix for Roberta model', figsize=(7,7))
import tensorflow as tf
import numpy as np
from transformers import TFRobertaModel, RobertaTokenizer, TFRobertaForSequenceClassification
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
Roberta_Tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
Disastweets = pd.concat([TrainDataSet, TestDataSet])
Disastweets = Disastweets.text.values

print('Tokenized: ', Roberta_Tokenizer.tokenize(Disastweets[0]))
print('Token IDs: ', Roberta_Tokenizer.convert_tokens_to_ids(Roberta_Tokenizer.tokenize(Disastweets[0])))
max_len = 0

# For every sentence...
for text in Disastweets:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = Roberta_Tokenizer.tokenize(text,is_pretokenized=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
def MapTokens(texts):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = Roberta_Tokenizer.tokenize(text,is_pretokenized=True)
        CLS = Roberta_Tokenizer.cls_token
        SEP = Roberta_Tokenizer.sep_token    
        text = text[:max_len-2]
        input_sequence = [CLS] + text + [SEP]
        pad_len = max_len - len(input_sequence)
        
        tokens = Roberta_Tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
train_input = MapTokens(TrainDataSet.text.values)
test_input = MapTokens(TestDataSet.text.values)
train_labels = TrainDataSet.target.values
# Ref : https://github.com/huggingface/transformers/issues/1350
    
class _TFRobertaForSequenceClassification(TFRobertaForSequenceClassification):
    
    def __init__(self, config, *inputs, **kwargs):
        super(_TFRobertaForSequenceClassification, self).__init__(config, *inputs, **kwargs)
        self.roberta.call = tf.function(self.roberta.call)


# Load model and collect encodings
roberta = _TFRobertaForSequenceClassification.from_pretrained('roberta-large',num_labels=2)

#Optimizer
Roberta_Tweet_Optimizer = Adam(
                  lr = 6e-6, # args.learning_rate
                  epsilon = 1e-8 # args.adam_epsilon
                )
#Compile the Model
roberta.compile(optimizer=Roberta_Tweet_Optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#roberta.get_layer('predictions').activation=tf.compat.v1.keras.activations.linear
roberta.summary()
Roberta_Tweet_Optimizer = Adam(
                  lr = 6e-6, # args.learning_rate
                  epsilon = 1e-8 # args.adam_epsilon
                )
roberta.compile(optimizer=Roberta_Tweet_Optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#roberta.get_layer('predictions').activation=tf.compat.v1.keras.activations.linear
roberta.summary()

checkpoint = ModelCheckpoint('model_roberta.h5', monitor='val_loss', save_best_only=True)

train_history = roberta.fit(
    train_input, TrainDataSet['target'].values,
    validation_split = 0.2,
    epochs = 4, # recomended 3-5 epochs
    callbacks=[checkpoint],
    batch_size = 32
)
roberta.load_weights('model_roberta.h5')
test_pred_roberta = roberta.predict(test_input)
flat_predictions = [item for sublist in test_pred_roberta for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
pred = pd.DataFrame(flat_predictions, columns=['target'])
#pred.plot.hist()

values = [len(pred.query("target == 1")), len(pred.query("target == 0"))]
labels = ['Disaster','Non-Disaster']


# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(width=600, height=400,title_text='RoBERTa - Predicted Disaster Tweet Percentage',
                 annotations=[dict(text='#tweet', x=0.5, y=0.5, font_size=20, showarrow=False)])
plotly.offline.iplot(fig)

TestPrediction = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
TestPrediction['target'] = flat_predictions
TestPrediction.head(10)

train_pred_roberta = roberta.predict(train_input)

train_predictions = [item for sublist in train_pred_roberta for item in sublist]
train_predictions = np.argmax(train_predictions, axis=1).flatten()
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
def plot_cm(y_true, y_pred, title, figsize=(5,5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

plot_cm(train_predictions, TrainDataSet['target'].values, 'Confusion matrix for Roberta model', figsize=(7,7))
TestPrediction.to_csv("submission.csv", index=False, header=True)