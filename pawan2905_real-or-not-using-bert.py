# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import nltk

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist



# Loading some sklearn packaces for modelling.



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation, NMF

from sklearn.metrics import f1_score, accuracy_score



# Some packages for word clouds and NER.



from wordcloud import WordCloud, STOPWORDS

from collections import Counter, defaultdict

from PIL import Image

import spacy

!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz

import en_core_web_sm



# Core packages for general use throughout the notebook.



import random

import warnings

import time

import datetime



# For customizing our plots.



from matplotlib.ticker import MaxNLocator

import matplotlib.gridspec as gridspec

import matplotlib.patches as mpatches



# Loading pytorch packages.



import torch

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler



# Setting some options for general use.



stop = set(stopwords.words('english'))

# plt.style.use('fivethirtyeight')

sns.set(font_scale=1.5)

pd.options.display.max_columns = 250

pd.options.display.max_rows = 250

warnings.filterwarnings('ignore')





#Setting seeds for consistent results.

seed_val = 42

random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import torch

import transformers as ppb

import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Markdown, display

def printmd(string):

    display(Markdown(string))

#printmd('**bold**')
# Loading the train and test data for visualization & exploration.



trainv = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

testv = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# Taking general look at the both datasets.



display(trainv.sample(5))

display(testv.sample(5))
# Checking observation and feature numbers for train and test data.



print(trainv.shape)

print(testv.shape)
import re

import string
# Some basic helper functions to clean text by removing urls, emojis, html tags and punctuations.



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'', text)





def remove_emoji(text):

    emoji_pattern = re.compile(

        '['

        u'\U0001F600-\U0001F64F'  # emoticons

        u'\U0001F300-\U0001F5FF'  # symbols & pictographs

        u'\U0001F680-\U0001F6FF'  # transport & map symbols

        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)

        u'\U00002702-\U000027B0'

        u'\U000024C2-\U0001F251'

        ']+',

        flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)





def remove_html(text):

    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    return re.sub(html, '', text)





def remove_punct(text):

    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)



# Applying helper functions



trainv['text_clean'] = trainv['text'].apply(lambda x: remove_URL(x))

trainv['text_clean'] = trainv['text_clean'].apply(lambda x: remove_emoji(x))

trainv['text_clean'] = trainv['text_clean'].apply(lambda x: remove_html(x))

trainv['text_clean'] = trainv['text_clean'].apply(lambda x: remove_punct(x))

# Tokenizing the tweet base texts.



trainv['tokenized'] = trainv['text_clean'].apply(word_tokenize)



trainv.head()
# Lower casing clean text.



trainv['lower'] = trainv['tokenized'].apply(

    lambda x: [word.lower() for word in x])



trainv.head()
# Removing stopwords.



trainv['stopwords_removed'] = trainv['lower'].apply(

    lambda x: [word for word in x if word not in stop])



trainv.head()

# Applying part of speech tags.



trainv['pos_tags'] = trainv['stopwords_removed'].apply(nltk.tag.pos_tag)



trainv.head()
# Converting part of speeches to wordnet format.



def get_wordnet_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN





trainv['wordnet_pos'] = trainv['pos_tags'].apply(

    lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])



trainv.head()

# Applying word lemmatizer.



wnl = WordNetLemmatizer()



trainv['lemmatized'] = trainv['wordnet_pos'].apply(

    lambda x: [wnl.lemmatize(word, tag) for word, tag in x])



trainv['lemmatized'] = trainv['lemmatized'].apply(

    lambda x: [word for word in x if word not in stop])



trainv['lemma_str'] = [' '.join(map(str, l)) for l in trainv['lemmatized']]



trainv.head()
# Displaying target distribution.



fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), dpi=100)

sns.countplot(trainv['target'], ax=axes[0])

axes[1].pie(trainv['target'].value_counts(),

            labels=['Not Disaster', 'Disaster'],

            autopct='%1.2f%%',

            shadow=True,

            explode=(0.05, 0),

            startangle=60)

fig.suptitle('Distribution of the Tweets', fontsize=24)

plt.show()
# Creating a new feature for the visualization.



trainv['Character Count'] = trainv['text_clean'].apply(lambda x: len(str(x)))





def plot_dist3(df, feature, title):

    # Creating a customized chart. and giving in figsize and everything.

    fig = plt.figure(constrained_layout=True, figsize=(18, 8))

    # Creating a grid of 3 cols and 3 rows.

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)



    # Customizing the histogram grid.

    ax1 = fig.add_subplot(grid[0, :2])

    # Set the title.

    ax1.set_title('Histogram')

    # plot the histogram.

    sns.distplot(df.loc[:, feature],

                 hist=True,

                 kde=True,

                 ax=ax1,

                 color='#e74c3c')

    ax1.set(ylabel='Frequency')

    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))



    # Customizing the ecdf_plot.

    ax2 = fig.add_subplot(grid[1, :2])

    # Set the title.

    ax2.set_title('Empirical CDF')

    # Plotting the ecdf_Plot.

    sns.distplot(df.loc[:, feature],

                 ax=ax2,

                 kde_kws={'cumulative': True},

                 hist_kws={'cumulative': True},

                 color='#e74c3c')

    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))

    ax2.set(ylabel='Cumulative Probability')



    # Customizing the Box Plot.

    ax3 = fig.add_subplot(grid[:, 2])

    # Set title.

    ax3.set_title('Box Plot')

    # Plotting the box plot.

    sns.boxplot(x=feature, data=df, orient='v', ax=ax3, color='#e74c3c')

    ax3.yaxis.set_major_locator(MaxNLocator(nbins=25))



    plt.suptitle(f'{title}', fontsize=24)

plot_dist3(trainv[trainv['target'] == 0], 'Character Count',

           'Characters Per "Non Disaster" Tweet')
plot_dist3(trainv[trainv['target'] == 1], 'Character Count',

           'Characters Per "Disaster" Tweet')
def plot_word_number_histogram(textno, textye):

    

    """A function for comparing word counts"""



    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)

    sns.distplot(textno.str.split().map(lambda x: len(x)), ax=axes[0], color='#e74c3c')

    sns.distplot(textye.str.split().map(lambda x: len(x)), ax=axes[1], color='#e74c3c')

    

    axes[0].set_xlabel('Word Count')

    axes[0].set_ylabel('Frequency')

    axes[0].set_title('Non Disaster Tweets')

    axes[1].set_xlabel('Word Count')

    axes[1].set_title('Disaster Tweets')

    

    fig.suptitle('Words Per Tweet', fontsize=24, va='baseline')

    

    fig.tight_layout()
plot_word_number_histogram(trainv[trainv['target'] == 0]['text'],

                           trainv[trainv['target'] == 1]['text'])
def plot_word_len_histogram(textno, textye):

    

    """A function for comparing average word length"""

    

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)

    sns.distplot(textno.str.split().apply(lambda x: [len(i) for i in x]).map(

        lambda x: np.mean(x)),

                 ax=axes[0], color='#e74c3c')

    sns.distplot(textye.str.split().apply(lambda x: [len(i) for i in x]).map(

        lambda x: np.mean(x)),

                 ax=axes[1], color='#e74c3c')

    

    axes[0].set_xlabel('Word Length')

    axes[0].set_ylabel('Frequency')

    axes[0].set_title('Non Disaster Tweets')

    axes[1].set_xlabel('Word Length')

    axes[1].set_title('Disaster Tweets')

    

    fig.suptitle('Mean Word Lengths', fontsize=24, va='baseline')

    fig.tight_layout()
plot_word_len_histogram(trainv[trainv['target'] == 0]['text'],

                        trainv[trainv['target'] == 1]['text'])
lis = [

    trainv[trainv['target'] == 0]['lemma_str'],

    trainv[trainv['target'] == 1]['lemma_str']

]
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes = axes.flatten()



for i, j in zip(lis, axes):

    try:

        new = i.str.split()

        new = new.values.tolist()

        corpus = [word.lower() for i in new for word in i]

        dic = defaultdict(int)

        for word in corpus:

            if word in stop:

                dic[word] += 1



        top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:15]

        x, y = zip(*top)

        df = pd.DataFrame([x, y]).T

        df = df.rename(columns={0: 'Stopword', 1: 'Count'})

        sns.barplot(x='Count', y='Stopword', data=df, palette='plasma', ax=j)

        plt.tight_layout()

    except:

        plt.close()

        print('No stopwords left in texts.')

        break
# Displaying most common words.



fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes = axes.flatten()



for i, j in zip(lis, axes):



    new = i.str.split()

    new = new.values.tolist()

    corpus = [word for i in new for word in i]



    counter = Counter(corpus)

    most = counter.most_common()

    x, y = [], []

    for word, count in most[:30]:

        if (word not in stop):

            x.append(word)

            y.append(count)



    sns.barplot(x=y, y=x, palette='plasma', ax=j)

axes[0].set_title('Non Disaster Tweets')



axes[1].set_title('Disaster Tweets')

axes[0].set_xlabel('Count')

axes[0].set_ylabel('Word')

axes[1].set_xlabel('Count')

axes[1].set_ylabel('Word')



fig.suptitle('Most Common Unigrams', fontsize=24, va='baseline')

plt.tight_layout()
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")

test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

sample = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
# For DistilBERT:

#model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')



## Want BERT instead of distilBERT? Uncomment the following line:

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')



# Load pretrained model/tokenizer

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)
train_df.head()
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
print(f'Number of training tweets: {train_df.shape[0]}\n')

print(f'Number of training tweets: {test_df.shape[0]}\n')
# Setting target variables, creating combined data and saving index for dividing combined data later.



labels = train_df['target'].values

idx = len(labels)

combined = pd.concat([train_df, test_df])

combined = combined.text.values
# Tokenizing the combined text data using bert tokenizer.



tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
# Print the original tweet.



print("Origina:",combined[0])



# Print the tweet split into tokens



print("Tokenized:", tokenizer.tokenize(combined[0]))



#print the sentence mapped to token ID's



print("Token IDs: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(combined[0])))
max_len = 0



# For every sentence...



for text in combined:



    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.

    

    input_ids = tokenizer.encode(text, add_special_tokens=True)



    # Update the maximum sentence length.

    

    max_len = max(max_len, len(input_ids))



print('Max sentence length: ', max_len)
# Making list of sentence lenghts:



token_lens = []



for text in combined:

    tokens = tokenizer.encode(text, max_length = 512)

    token_lens.append(len(tokens))
# Displaying sentence length dist.

import matplotlib.pyplot as plt

import seaborn as sns

fig, axes = plt.subplots(figsize=(14, 6))

sns.distplot(token_lens, color='#e74c3c')

plt.show()
# Splitting the train test data after tokenizing.



train= combined[:idx]

test = combined[idx:]

train.shape
def tokenize_map(sentence,labs='None'):

    

    """A function for tokenize all of the sentences and map the tokens to their word IDs."""

    

    global labels

    

    input_ids = []

    attention_masks = []



    # For every sentence...

    

    for text in sentence:

        #   "encode_plus" will:

        

        #   (1) Tokenize the sentence.

        #   (2) Prepend the `[CLS]` token to the start.

        #   (3) Append the `[SEP]` token to the end.

        #   (4) Map tokens to their IDs.

        #   (5) Pad or truncate the sentence to `max_length`

        #   (6) Create attention masks for [PAD] tokens.

        

        encoded_dict = tokenizer.encode_plus(

                            text,                      # Sentence to encode.

                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            truncation='longest_first', # Activate and control truncation

                            max_length = 84,           # Max length according to our text data.

                            pad_to_max_length = True, # Pad & truncate all sentences.

                            return_attention_mask = True,   # Construct attn. masks.

                            return_tensors = 'pt',     # Return pytorch tensors.

                       )



        # Add the encoded sentence to the id list. 

        

        input_ids.append(encoded_dict['input_ids'])



        # And its attention mask (simply differentiates padding from non-padding).

        

        attention_masks.append(encoded_dict['attention_mask'])



    # Convert the lists into tensors.

    

    input_ids = torch.cat(input_ids, dim=0)

    attention_masks = torch.cat(attention_masks, dim=0)

    

    if labs != 'None': # Setting this for using this definition for both train and test data so labels won't be a problem in our outputs.

        labels = torch.tensor(labels)

        return input_ids, attention_masks, labels

    else:

        return input_ids, attention_masks
# Tokenizing all of the train test sentences and mapping the tokens to their word IDs.



input_ids, attention_masks, labels = tokenize_map(train, labels)

test_input_ids, test_attention_masks= tokenize_map(test)
# Combine the training inputs into a TensorDataset.



dataset = TensorDataset(input_ids, attention_masks, labels)



# Create a 80-20 train-validation split.



# Calculate the number of samples to include in each set.



train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size



# Divide the dataset by randomly selecting samples.



train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



print('{:>5,} training samples'.format(train_size))

print('{:>5,} validation samples'.format(val_size))
# The DataLoader needs to know our batch size for training, so we specify it here. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.



batch_size = 32



# Create the DataLoaders for our training and validation sets.

# We'll take training samples in random order. 



train_dataloader = DataLoader(

            train_dataset,  # The training samples.

            sampler = RandomSampler(train_dataset), # Select batches randomly

            batch_size = batch_size # Trains with this batch size.

        )



# For validation the order doesn't matter, so we'll just read them sequentially.



validation_dataloader = DataLoader(

            val_dataset, # The validation samples.

            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.

            batch_size = batch_size # Evaluate with this batch size.

        )
prediction_data = TensorDataset(test_input_ids, test_attention_masks)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 



model = BertForSequenceClassification.from_pretrained(

    'bert-large-uncased', # Use the 124-layer, 1024-hidden, 16-heads, 340M parameters BERT model with an uncased vocab.

    num_labels = 2, # The number of output labels--2 for binary classification. You can increase this for multi-class tasks.   

    output_attentions = False, # Whether the model returns attentions weights.

    output_hidden_states = False, # Whether the model returns all hidden-states.

)



# Tell pytorch to run this model on the device which we set GPU in our case.



model.to(device)
# Get all of the model's parameters as a list of tuples:



params = list(model.named_parameters())



print('The BERT model has {:} different named parameters.\n'.format(len(params)))



print('==== Embedding Layer ====\n')



for p in params[0:5]:

    print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))



print('\n==== First Transformer ====\n')



for p in params[5:21]:

    print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))



print('\n==== Output Layer ====\n')



for p in params[-4:]:

    print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))
# Note: AdamW is a class from the huggingface library (as opposed to pytorch).



# The 'W' stands for 'Weight Decay fix' probably...



optimizer = AdamW(model.parameters(),

                  lr = 6e-6, # args.learning_rate

                  eps = 1e-8 # args.adam_epsilon

                )
# Number of training epochs. The BERT authors recommend between 2 and 4. 



# We chose to run for 3, but we'll see later that this may be over-fitting the training data.



epochs = 3



# Total number of training steps is [number of batches] x [number of epochs] (Note that this is not the same as the number of training samples).

total_steps = len(train_dataloader) * epochs



# Create the learning rate scheduler.



scheduler = get_linear_schedule_with_warmup(optimizer, 

                                            num_warmup_steps = 0, # Default value in run_glue.py

                                            num_training_steps = total_steps)
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
def format_time(elapsed):    

    

    """A function that takes a time in seconds and returns a string hh:mm:ss"""

    

    # Round to the nearest second.

    elapsed_rounded = int(round((elapsed)))

    

    # Format as hh:mm:ss

    return str(datetime.timedelta(seconds=elapsed_rounded))
# This training code is based on the `run_glue.py` script here:



# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128





# We'll store a number of quantities such as training and validation loss, validation accuracy, f1 score and timings.



training_stats = []



# Measure the total training time for the whole run.



total_t0 = time.time()



# For each epoch...



for epoch_i in range(0, epochs):

    

    # ========================================

    #               Training

    # ========================================

    

    # Perform one full pass over the training set.



    print('')

    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    print('Training...')



    # Measure how long the training epoch takes:

    

    t0 = time.time()



    # Reset the total loss for this epoch.

    

    total_train_loss = 0



    # Put the model into training mode. Don't be mislead--the call to `train` just changes the *mode*, it doesn't *perform* the training.

    

    # `dropout` and `batchnorm` layers behave differently during training vs. test ,

    # source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch

    

    model.train()



    # For each batch of training data...

    

    for step, batch in enumerate(train_dataloader):



        # Progress update every 50 batches.

        if step % 50 == 0 and not step == 0:

            # Calculate elapsed time in minutes.

            elapsed = format_time(time.time() - t0)

            

            # Report progress.

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))



        # Unpack this training batch from our dataloader. 

        #

        # As we unpack the batch, we'll also copy each tensor to the device(gpu in our case) using the `to` method.

        #

        # `batch` contains three pytorch tensors:

        #   [0]: input ids 

        #   [1]: attention masks

        #   [2]: labels 

        

        b_input_ids = batch[0].to(device).to(torch.int64)

        b_input_mask = batch[1].to(device).to(torch.int64)

        b_labels = batch[2].to(device).to(torch.int64)



        # Always clear any previously calculated gradients before performing a backward pass. PyTorch doesn't do this automatically because accumulating the gradients is 'convenient while training RNNs'. 

        # Source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

        

        model.zero_grad()        



        # Perform a forward pass (evaluate the model on this training batch).

        # The documentation for this `model` function is down here: 

        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers BertForSequenceClassification.

        

        # It returns different numbers of parameters depending on what arguments given and what flags are set. For our useage here, it returns the loss (because we provided labels),

        # And the 'logits' (the model outputs prior to activation.)

        

        loss, logits = model(b_input_ids, 

                             token_type_ids=None, 

                             attention_mask=b_input_mask, 

                             labels=b_labels)



        # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end, 

        # `loss` is a tensor containing a single value; the `.item()` function just returns the Python value from the tensor.

        

        total_train_loss += loss.item()



        # Perform a backward pass to calculate the gradients.

        

        loss.backward()



        # Clip the norm of the gradients to 1.0 This is to help prevent the 'exploding gradients' problem.

        

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)



        # Update parameters and take a step using the computed gradient.

        

        # The optimizer dictates the 'update rule'(How the parameters are modified based on their gradients, the learning rate, etc.)

        

        optimizer.step()



        # Update the learning rate.

        

        scheduler.step()



    # Calculate the average loss over all of the batches.

    

    avg_train_loss = total_train_loss / len(train_dataloader)            

    

    # Measure how long this epoch took.

    

    training_time = format_time(time.time() - t0)



    print('')

    print('  Average training loss: {0:.2f}'.format(avg_train_loss))

    print('  Training epcoh took: {:}'.format(training_time))

        

    # ========================================

    #               Validation

    # ========================================

    # After the completion of each training epoch, measure our performance on our validation set.



    print('')

    print('Running Validation...')



    t0 = time.time()



    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.

    

    model.eval()



    # Tracking variables:

    

    total_eval_accuracy = 0

    total_eval_loss = 0

    total_eval_f1 = 0

    nb_eval_steps = 0



    # Evaluate data for one epoch.

    

    for batch in validation_dataloader:

        

        # Unpack this training batch from our dataloader. 

        

        # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method.

        

        # `batch` contains three pytorch tensors:

        #   [0]: input ids 

        #   [1]: attention masks

        #   [2]: labels 

        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)

        

        # Tell pytorch not to bother with constructing the compute graph during the forward pass, since this is only needed for backprop (training part).

        

        with torch.no_grad():        



            # Forward pass, calculate logit predictions.

            # token_type_ids is the same as the 'segment ids', which differentiates sentence 1 and 2 in 2-sentence tasks.

            # The documentation for this `model` function is down here: 

            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers BertForSequenceClassification.

            # Get the 'logits' output by the model. The 'logits' are the output values prior to applying an activation function like the softmax.

            

            (loss, logits) = model(b_input_ids, 

                                   token_type_ids=None, 

                                   attention_mask=b_input_mask,

                                   labels=b_labels)

            

        # Accumulate the validation loss.

        

        total_eval_loss += loss.item()



        # Move logits and labels to CPU:

        

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()



        # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches:

        

        total_eval_accuracy += flat_accuracy(logits, label_ids)

        total_eval_f1 += flat_f1(logits, label_ids)

        



    # Report the final accuracy for this validation run.

    

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

    print('  Accuracy: {0:.2f}'.format(avg_val_accuracy))

    

    # Report the final f1 score for this validation run.

    

    avg_val_f1 = total_eval_f1 / len(validation_dataloader)

    print('  F1: {0:.2f}'.format(avg_val_f1))



    # Calculate the average loss over all of the batches.

    

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    

    

    

    # Measure how long the validation run took:

    

    validation_time = format_time(time.time() - t0)

    

    print('  Validation Loss: {0:.2f}'.format(avg_val_loss))

    print('  Validation took: {:}'.format(validation_time))



    # Record all statistics from this epoch.

    

    training_stats.append(

        {

            'epoch': epoch_i + 1,

            'Training Loss': avg_train_loss,

            'Valid. Loss': avg_val_loss,

            'Valid. Accur.': avg_val_accuracy,

            'Val_F1' : avg_val_f1,

            'Training Time': training_time,

            'Validation Time': validation_time

        }

    )



print('')

print('Training complete!')



print('Total training took {:} (h:mm:ss)'.format(format_time(time.time()-total_t0)))

# Display floats with two decimal places.



pd.set_option('precision', 2)



# Create a DataFrame from our training statistics.



df_stats = pd.DataFrame(data=training_stats)



# Use the 'epoch' as the row index.



df_stats = df_stats.set_index('epoch')



# Display the table.



display(df_stats)
# Increase the plot size and font size:



fig, axes = plt.subplots(figsize=(12,8))



# Plot the learning curve:



plt.plot(df_stats['Training Loss'], 'b-o', label='Training')

plt.plot(df_stats['Valid. Loss'], 'g-o', label='Validation')



# Label the plot:



plt.title('Training & Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')



plt.legend()

plt.xticks([1, 2, 3])



plt.show()
# Prediction on test set:



print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))



# Put model in evaluation mode:



model.eval()



# Tracking variables :



predictions = []



# Predict:



for batch in prediction_dataloader:

    

  # Add batch to GPU



  batch = tuple(t.to(device) for t in batch)

  

  # Unpack the inputs from our dataloader:

    

  b_input_ids, b_input_mask, = batch

  

  # Telling the model not to compute or store gradients, saving memory and speeding up prediction:



  with torch.no_grad():

      # Forward pass, calculate logit predictions:

    

      outputs = model(b_input_ids, token_type_ids=None, 

                      attention_mask=b_input_mask)



  logits = outputs[0]



  # Move logits and labels to CPU:

    

  logits = logits.detach().cpu().numpy()

 

  

  # Store predictions and true labels:

    

  predictions.append(logits)





print('    DONE.')
# Getting list of predictions and then choosing the target value with using argmax on probabilities.



flat_predictions = [item for sublist in predictions for item in sublist]

flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
len(flat_predictions)


submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission['target'] = flat_predictions

submission.head(10)
submission.to_csv("submission.csv")