# Basic python functionality
import re
import string
import collections
import numpy as np
import pandas as pd

# This is a great wrapper that displays a progress bar in the notebook
from tqdm.notebook import tqdm

# Plotting and visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette='deep', style='white')
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 100
plt.rc("axes.spines", top=False, right=False)
from wordcloud import WordCloud, STOPWORDS

# NLTK
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#Sci-Kit Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

#Huggingface Transformers
from transformers import BertTokenizer, BertConfig, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup

# this allows multiple outputs to be displayed from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# load data in Kaggle
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

# training dataset, sample of 10 entries
print('Training shape:', train.shape)
train.sample(5)

# test dataset, sample of 10 entries
print('Test shape:', test.shape)
test.sample(5)
print(f'Training null Values:\n{train.isnull().sum()}\n')
print(f'Test null Values:\n{test.isnull().sum()}')
train.dropna(axis=0, inplace=True)
def text_preprocessing(text):
    # makes text lowercase
    text = text.lower()
    # removes text within square brackets
    text = re.sub('\[.*?\]', '', text)
    # remove hyperlinks
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # removes text within brackets
    text = re.sub('<.*?>+', '', text)
    # removes punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # removes new line characters
    text = re.sub('\n', ' ', text)
    # removes words with numbers in them
    text = re.sub('\w*\d\w*', '', text)
    
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokenized_text = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    no_stop_words = [w for w in tokenized_text if w not in stop_words]
           
    text = ' '.join(no_stop_words)
    return text    
train['clean text'] = train['text'].apply(lambda x: text_preprocessing(x))
# tweet length
train['text length'] = train['text'].apply(len)
train['selected text length'] = train['selected_text'].apply(len)

# tweet word count
train['word count'] = train['text'].apply(lambda x: len(x.split()))
train['selected word count'] = train['selected_text'].apply(lambda x: len(x.split()))
# creating a function for the Jaccard Similarity between two strings
def jaccard(a,b):
    a = set(str(a).lower().split())
    b = set(str(b).lower().split())
    return len(a & b) / len(a | b)
jaccard_scores = []
text = train.text.iloc[0]
for i in range(len(train)):
    j = jaccard(train.text.iloc[i],train.selected_text.iloc[i])
    jaccard_scores.append(j)
train['jaccard score'] = jaccard_scores
positive = train[train['sentiment']=='positive']
neutral = train[train['sentiment']=='neutral']
negative = train[train['sentiment']=='negative']
sns.countplot(train.sentiment)
sns.countplot(test.sentiment)
p,neg, neu = train.sentiment.value_counts(normalize=True) - \
test.sentiment.value_counts(normalize=True)
print(f'Difference in positive examples: {p*100}%')
print(f'Difference in negative examples: {neg*100}%')
print(f'Difference in neutral examples: {neu*100}%')
fig, axs = plt.subplots(3,1) 
sns.distplot(positive['text length'], bins=32, color='green', ax=axs[0])
sns.distplot(neutral['text length'], bins=32, ax=axs[1])
sns.distplot(negative['text length'], bins=32, color='red', ax=axs[2])

fig, axs = plt.subplots(3,1) 
sns.distplot(positive['word count'], bins=32, color='green', ax=axs[0])
sns.distplot(neutral['word count'], bins=32, ax=axs[1])
sns.distplot(negative['word count'], bins=32, color='red', ax=axs[2])
# plotting the similiairty between the positive words
fig, axs = plt.subplots(figsize=(10, 4)) 
sns.kdeplot(positive['text length'], color='purple', shade=True)
sns.kdeplot(positive['selected text length'], color='orange', shade=True)
plt.title('Text length vs Selected Text Length: Positive')
# plotting the similiairty between the negative words
fig, axs = plt.subplots(figsize=(10, 4)) 
sns.kdeplot(negative['text length'], color='purple', shade=True)
sns.kdeplot(negative['selected text length'], color='orange', shade=True)
plt.title('Text length vs Selected Text Length: Negative')
# plotting the similiairty between the neutral words
fig, axs = plt.subplots(figsize=(10, 4)) 
sns.kdeplot(neutral['text length'], color='purple', shade=True)
sns.kdeplot(neutral['selected text length'], color='orange', shade=True)
plt.title('Similarity between Neutral Selected Text and Tweet')
plt.title('Text length vs Selected Text Length: Neutral')
fig, axs = plt.subplots(figsize=(10, 4)) 
sns.kdeplot(positive['selected text length'], color='green', shade=True)
sns.kdeplot(negative['selected text length'], color='red', shade=True)
plt.title('Positive Selected Text vs. Negative Selected Text')
fig, axs = plt.subplots(figsize=(10, 4), dpi=100) 
sns.kdeplot(positive['jaccard score'], color='green', shade=True)
sns.kdeplot(negative['jaccard score'], color='red', shade=True)
plt.title('Positive and Negative Jaccard Score Distributions')
# Word cloud
wordcloud = WordCloud(background_color='white', colormap='viridis_r', height=1080, width=1080).generate("".join(t for t in positive['selected_text']))
plt.figure(figsize=(12,12))
plt.imshow(wordcloud)

plt.axis('off')
plt.show()
wordcloud = WordCloud(background_color='white', colormap='inferno_r', height=1080, width=1080).generate("".join(t for t in negative['selected_text']))
plt.figure(figsize=(12,12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
def get_ngrams(corpus, n, length):
    # Convert a collection of text documents to a matrix of token counts 
    # The fit method learns a vocabulary dictionary of all tokens in the raw documents
    vec = CountVectorizer(stop_words='english', ngram_range=(n,n)).fit(corpus)
    
    #bag_of_words a matrix where each row represents a specific text in corpus and each 
    # column represents a word in vocabulary, that is, all words found in corpus
    bag_of_words = vec.transform(corpus)
    
    sum_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    
    return word_freq[:length]
def plot_ngram(corpus, n, length, color):
    color = color + 's_d'
    if n == 1:
        n_gram = 'Unigrams'
    elif n == 2:
        n_gram = 'Bigrams'
    elif n == 3:
        n_gram = 'Trigrams'
    else:
        n_gram = str(n) + '-grams'
        
    df = pd.DataFrame()
    df = pd.DataFrame(get_ngrams(corpus, n, length), columns=[n_gram, 'Count'])
    
    plot = sns.barplot(x='Count', y = n_gram, data=df, palette= color)

    del df
    
    return fig, plot  
plot_ngram(positive['clean text'], 2, 20, 'Green')
plot_ngram(neutral['clean text'], 2, 20, 'Blue')
plot_ngram(negative['clean text'], 2, 20, 'Red')
plot_ngram(positive['selected_text'], 2, 20, 'Green')
plot_ngram(negative['selected_text'], 2, 20, 'Red')
train_batch_size = 32
validate_batch_size = 16
epochs = 5
!mkdir models
model_dir = 'models/'
model_path = 'models/bertbaseuncased.bin'
config = BertConfig.from_pretrained('bert-base-cased')
config.output_hidden_states = True
config.num_labels = 2
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# Define the maximum length
max_len = 0

for i in train['text']:
    encoded_text = tokenizer.encode(str(i))
    
    max_len = max(max_len, len(encoded_text))
    
for i in test['text']:
    encoded_text = tokenizer.encode(str(i))
    
    max_len = max(max_len, len(encoded_text))

max_len +=4
print(f'The maximum length of a tweet in the train and test datasets is {max_len-4}')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using {device}')
class BertBaseCased(nn.Module):
    
    def __init__(self, config):
        super(BertBaseCased, self).__init__()
        # Loading the model config file
        self.config = config 
        # Setting the number of labels we want to train our model for
        self.num_labels = config.num_labels 
        # The Bert Base the model is built on top of
        self.base = 'bert-base-cased'
        # Creates a BERT model instance
        self.bert = BertModel(config = self.config)
        # Creates a Linear Module the 768 hidden layer output from BERT and returns our predictions
        self.classifier = nn.Linear(768, 2)
        
    def __str__(self):
        return f'Bert Model using: {self.base}'
        
    def __repr__(self):
        return f'Bert Model using: {self.base}'
        
    def forward(self, ids, mask, token_type_ids):
        # This is the standard forward sequence common to PyTorch Modules
        sequence_output, _, _ = self.bert(ids,
                                   attention_mask=mask,
                                   token_type_ids=token_type_ids)
        # Takes the sequence output and transforms it to a prediction tensor
        logits = self.classifier(sequence_output)
        # Splitting the tensor output into our start and end positions of the selected text
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # Squeeze reduces the dimension of a tensor along the axis specified.
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits
class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = max_len
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        # This defines how any particular example from the training data is fed into the BERT model
        
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
        
        len_selected_text = len(selected_text)
        
        # This creates a vector encoding of the text, and returns some additional features necessary for the model
        encoded_tweet = self.tokenizer.encode_plus(self.sentiment[item],
                                                   # Original Tweet
                                                   tweet,
                                                   # The Maximum Length
                                                   max_length=max_len,
                                                   # Pads each tensor to the max length with 0
                                                   pad_to_max_length=True,
                                                   # A binary mask identifying the different sequences in the model
                                                   return_token_type_ids=True,
                                                   # Tells the model which tokens to pay attention to
                                                   return_attention_mask=True
                                                  )
        # Returns an encoded version of the selected_text
        encoded_target = self.tokenizer.encode(selected_text, add_special_tokens=False)
                
        targets = [0] * max_len
        target_start_ind = 0
        target_end_ind = 0
        
        # Finds the start and end indices of the target text within the tweet. This will be the 
        # targets the BERT Model Optimizes for
        for ind in (i for i, e in enumerate(encoded_tweet['input_ids']) if e == encoded_target[0]):
            if encoded_tweet['input_ids'][ind:ind+len(encoded_target)] == encoded_target:
                target_start_ind = ind
                target_end_ind = ind + len(encoded_target) - 1
                break
        
        target_start = [0] * max_len
        target_start[target_start_ind] = 1
        target_end = [0] * max_len
        target_end[target_end_ind] = 1
        
        # creating a mask for the targets
        for i in range(target_start_ind, target_end_ind+1):
            targets[i] = 1
                
        return {"ids": torch.tensor(encoded_tweet['input_ids'], dtype=torch.long),
            "mask": torch.tensor(encoded_tweet['attention_mask'], dtype=torch.long),
            "token_type_ids": torch.tensor(encoded_tweet['token_type_ids'], dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "targets_start": torch.tensor(target_start, dtype=torch.long),
            "targets_end": torch.tensor(target_end, dtype=torch.long),
            "orig_tweet": self.tweet[item],
            "orig_sentiment": self.sentiment[item],
            "orig_selected": self.selected_text[item],
        }
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1 + l2
def train_model(model, train_data_loader, optimizer, scheduler, results, epoch):
    
    model.train()
    losses = AverageMeter()

    tq = tqdm(train_data_loader)
    for bi, batch in enumerate(tq):
        ids = batch["ids"]
        token_type_ids = batch["token_type_ids"]
        mask = batch["mask"]
        targets_start = batch["targets_start"]
        targets_end = batch["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)


        # o1 and o2 are the predicited start and end indices of the target phrase
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )


        loss = loss_fn(o1, o2, targets_start, targets_end)

        # clears the gradient for each optimized tensor
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tq.set_postfix(loss=losses.avg)
        
        # Appends the result every 10 batches
        if bi % 10 == 0:
            results.append([(bi)+(epoch*len(train_data_loader)), losses.avg])

# Creating the Model
model = BertBaseCased(config=config)
# Sending the model to the device
model.to(device)
results = []

# The stratified KFold splits the data set 80/20 for the purpose of training and cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(kf.split(train, 
                                                       train['sentiment'])):
    print(f'Training Fold no: {fold+1}')
    
    # To save time and the GPU resources, just running a single fold
    if fold == 0:

        X_train = train.iloc[train_idx].reset_index(drop=True)
        X_validate = train.iloc[valid_idx].reset_index(drop=True)

        train_dataset = TweetDataset(tweet = X_train.text,
                                    sentiment = X_train.sentiment,
                                    selected_text = X_train.selected_text)

        train_data_loader = DataLoader(train_dataset, 
                                       batch_size = train_batch_size)
        
        # I have not had time to set up the cross-validation method yet; as I mentioned PyTorch is new territorry for me, and
        # this is another area I'm still figuring out...
        validate_dataset = TweetDataset(tweet = X_validate.text,
                                    sentiment = X_validate.sentiment,
                                    selected_text = X_validate.selected_text)

        validate_data_loader = DataLoader(validate_dataset, 
                                       batch_size = validate_batch_size)
        
        # Generating a list of the model parameters
        model_params = list(model.named_parameters())

        # These parameters are being excluded from the list given to the optimizer
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        # Generates a list of dicts of parameters to feed into the optimizer; one that has a slight
        # weight decay, and one with no decay
        optimizer_params = [
            {'params': [p for n, p in model_params if not any (nd in n for nd in no_decay)],
            'weight_decay': 0.001}, #TWEAK LATER?
            {'params': [p for n, p in model_params if any (nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]

        # total number of training steps for the learning rate
        num_train_steps = int(len(X_train)/train_batch_size * epochs)
        # Optimizer from HuggingFace to optimize the model weights
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adamw
        optimizer = AdamW(optimizer_params, lr=3e-5)
        # Determines the schedule for the learning rate
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#learning-rate-schedules
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=0,
                                                   num_training_steps=num_train_steps)

        for epoch in range(epochs):
            train_model(model, train_data_loader, optimizer, scheduler, results, epoch)

torch.save(model.state_dict(), 'models/bert_base_uncased.bin')
model.config.to_json_file('models/bert_base_cased_config.json')
tokenizer.save_pretrained('models/')
# Sets the model in eval mode
model.eval()
model.to(device)

#This is a quick way to add the final column needed to create out TweetDataset Object; we won't be using it
test.loc[:, "selected_text"] = test.text.values

validate_dataset = TweetDataset(tweet = test.text,
                            sentiment = test.sentiment,
                                selected_text = test.selected_text
                               )

validate_data_loader = DataLoader(validate_dataset, 
                                  shuffle=False,
                                  batch_size = validate_batch_size)
outputs = []
fin_outputs_start = []
fin_outputs_end = []
fin_token_ids = []
fin_original_tweet = []
fin_original_sentiment = []
fin_original_selected = []

with torch.no_grad():
    for bi, batch in enumerate(tqdm(validate_data_loader)):
        ids = batch["ids"]
        token_type_ids = batch["token_type_ids"]
        mask = batch["mask"]
        original_tweet = batch['orig_tweet']
        original_sentiment = batch['orig_sentiment']
        original_selected = batch['orig_selected']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
        fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())
        fin_original_tweet.extend(original_tweet)
        fin_original_sentiment.extend(original_sentiment)
        fin_token_ids.extend(ids.cpu().detach().numpy().tolist())
        fin_original_selected.extend(original_selected)
        
fin_outputs_start = np.vstack(fin_outputs_start)
fin_outputs_end = np.vstack(fin_outputs_end)
fin_token_ids = np.vstack(fin_token_ids)
predictions = []
start_idx = []
end_idx = []
threshold = 0.3

for j in range(len(fin_token_ids)):
    target_string = fin_original_selected[j]
    tweet_tokens = fin_token_ids[j]
    original_tweet = fin_original_tweet[j]
    sentiment_val = fin_original_sentiment[j]
    mask_start = fin_outputs_start[j]>=threshold
    mask_end = fin_outputs_end[j]>=threshold

    output_mask = [0] * len(mask_end)

    idx_start = np.nonzero(mask_start)[0]
    idx_end = np.nonzero(mask_end)[0]

    if len(idx_start) > 0:
        idx_start = idx_start[0]
        if len(idx_end) > 0:
            idx_end = idx_end[0]
        else:
            idx_end = idx_start   
    else:
        idx_start = 0
        idx_end = 0

    predicted_selected_text = []
    for t in range(idx_start, idx_end+1):
        predicted_selected_text.append(tweet_tokens[t])

    predicted_selected_text = tokenizer.decode(predicted_selected_text)
    predictions.append(predicted_selected_text)
    
predictions = np.vstack(predictions)    
 
submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
submission.loc[:, 'selected_text'] = predictions
submission.to_csv('submission.csv', index=False)