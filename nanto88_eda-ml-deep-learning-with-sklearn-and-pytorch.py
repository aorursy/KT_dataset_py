# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import html

import ftfy



import nltk

from nltk import word_tokenize 

from nltk.util import ngrams

from collections import defaultdict

from nltk.corpus import stopwords

import string, re

from scipy.stats import norm



from tqdm.notebook import tqdm



import wandb

warnings.filterwarnings("ignore")
np.random.seed(42)
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
test.head()
all_data = pd.concat([train, test], ignore_index=True)
all_data.head(10)
y_values = train.target.value_counts().values

percent_y_values = [(y_values[0] / sum(y_values)),  (y_values[1] / sum(y_values))]

sns.barplot(x=['Not Disaster', 'Real Disaster'], y=percent_y_values, 

            palette="coolwarm").set_title('Distribution Target in train data')

plt.show()
def null_value_counts(df, col, null=True):

    if null:

        return len(df[df[col].isnull()][col].to_list())

    else:

        return len(df[~df[col].isnull()][col].to_list())

    
count_null_values = [null_value_counts(all_data, 'location', null=True), null_value_counts(all_data, 'location', null=False)]

percent_null_values = [(count_null_values[0] / sum(count_null_values)), (count_null_values[1] / sum(count_null_values))]

sns.barplot(x=['NaN', 'Not NaN'], y=percent_null_values, 

            palette="coolwarm").set_title('Distribution Null and Not Null Values in Location')

plt.show()
count_null_values = [null_value_counts(all_data, 'keyword', null=True), null_value_counts(all_data, 'keyword', null=False)]

percent_null_values = [(count_null_values[0] / sum(count_null_values)), (count_null_values[1] / sum(count_null_values))]

sns.barplot(x=['NaN', 'Not NaN'], y=percent_null_values, 

            palette="coolwarm").set_title('Distribution Null and Not Null Values in Keyword')

plt.show()


print('null_values in keyword', count_null_values[0])

print('percent_null_values in keyword', percent_null_values[0])
sns.barplot(y=all_data[all_data.target == 1].location.value_counts()[:15].index, 

            x=all_data[all_data.target == 1].location.value_counts()[:15].values,

            palette="coolwarm").set_title('Top 15 Locations in Real Disaster Tweets')

plt.show()
sns.barplot(y=all_data[all_data.target == 0].location.value_counts()[:15].index, 

            x=all_data[all_data.target == 0].location.value_counts()[:15].values,

            palette="coolwarm").set_title('Top 15 Locations in Not Disaster Tweets')

plt.show()
sns.barplot(y=all_data[all_data.target == 1].keyword.value_counts()[:15].index, 

            x=all_data[all_data.target == 1].keyword.value_counts()[:15].values,

           palette='coolwarm').set_title('Top 15 Keywords in Real Disaster Tweets')

plt.show()
sns.barplot(y=all_data[all_data.target == 0].keyword.value_counts()[:15].index, 

            x=all_data[all_data.target == 0].keyword.value_counts()[:15].values, 

            palette='coolwarm').set_title('Top 15 Keywords in Not Disaster Tweets')

plt.show()
len_w_tweet_disaster = train[train.target == 1]['text'].apply(lambda x: len(x.split()))

len_w_tweet_not_disaster = train[train.target == 0]['text'].apply(lambda x: len(x.split()))



fig,ax = plt.subplots(2, figsize=(10,25))



ax1,ax2 = ax.flatten()

sns.distplot(len_w_tweet_disaster,color='red', fit=norm, ax=ax1, 

             axlabel='n_word').set_title('Tweet Word Length Distribution from Disaster Tweets', fontsize=15)

sns.distplot(len_w_tweet_not_disaster,color='blue', fit=norm, ax=ax2, 

             axlabel='n_word').set_title('Tweet Word Length Distribution from Not Disaster Tweets', fontsize=15)



plt.subplots_adjust(left=0.2, right=2)

plt.show()
plot_dist = sns.distplot(len_w_tweet_disaster,color='red', fit=norm)

plot_not_dist = sns.distplot(len_w_tweet_not_disaster,color='blue', fit=norm, axlabel='n_word').set_title(

    'Combine of Tweet Word Length Distribution from Disaster Tweets and Not Disaster Tweets', fontsize=15)



plt.subplots_adjust(left=0.2, right=3)

plt.show()
len_c_tweet_disaster = train[train.target == 1]['text'].apply(lambda x: len(x))

len_c_tweet_not_disaster = train[train.target == 0]['text'].apply(lambda x: len(x))



fig,ax = plt.subplots(2, figsize=(10,25))



ax1,ax2 = ax.flatten()

sns.distplot(len_c_tweet_disaster,color='red', fit=norm, ax=ax1, 

             axlabel='n_character').set_title('Tweet Character Length Distribution from Disaster Tweets', fontsize=15)

sns.distplot(len_c_tweet_not_disaster,color='blue', fit=norm, ax=ax2, 

             axlabel='n_character').set_title('Tweet Character Length Distribution from Not Disaster Tweets', fontsize=15)



plt.subplots_adjust(left=0.7, right=2)

plt.show()
plot_dist = sns.distplot(len_c_tweet_disaster,color='red', fit=norm)

plot_not_dist = sns.distplot(len_c_tweet_not_disaster,color='blue', fit=norm, axlabel='n_character').set_title(

    'Combine of Tweet Character Length Distribution from Disaster Tweets and Not Disaster Tweets', fontsize=15)



plt.subplots_adjust(left=0.7, right=3)

plt.show()
def punctuations_checker(texts, get_top_15=True):

    punctuations = ['\\' + p for p in list(string.punctuation) if p not in list(',.?"\'@#')]

    dirty_words = defaultdict()

    for text in texts:

        for word in text.split():

            if all([ (len(re.findall(r'|'.join(punctuations), word)) > 1), (len(word) > 1) ]):

                try:

                    dirty_words[word] += 1

                except:

                    dirty_words[word] = 1

    if get_top_15:

        dirty_words = {k: v for k, v in sorted(dirty_words.items(), key=lambda item: item[1], reverse=True)[:15]}

    else:

        dirty_words = {k: v for k, v in sorted(dirty_words.items(), key=lambda item: item[1], reverse=True)}

    return dirty_words

            
dirty_words_real_disaster = punctuations_checker(all_data.text, get_top_15=True)

sns.barplot(y=list(dirty_words_real_disaster.keys()), x=list(dirty_words_real_disaster.values()),

           palette='coolwarm').set_title('Top 15 Words/Tokens Contain Punctuation Mark')



plt.show()
print( all_data[all_data.text.str.contains(';')].text.to_list()[:30] ) # show potentially dirty text
def preprocess(texts):

    clean_text = []

    for text in texts:

        text = html.unescape(text)

        text = ftfy.fix_text(text)

        text = re.sub(r'(?:http(?:s)?://)?\bt\b\.\bco\b(?:/[\w/-]+)?', ' ', text) # remove urls t.co (http://t.co/asd https://t.co/asd t.co/asd) show regex highlight : https://regex101.com/r/rGAI2w/1

        text = re.sub(r'&gt;', '>', text)

        text = re.sub(r'&lt;', '<', text)

        text = re.sub(r'&amp;', '&', text)

        text = re.sub('\\\n', '\n', text) # not raw string

        text = re.sub(r'\x89Û', '', text) # unicode

        text = re.sub(r'‰|_|Ï|Ò|ª|÷|å|©|£|À|Ì|Û|Ê', ' ', text) #special character

        text = text.replace('...', '')

        text = text.replace('%20', ' ')

        for p in list(string.punctuation):

            text = text.replace(p, f' {p} ') # replace in python more faster than regex re.sub

        text = re.sub('\s+', ' ', text).strip().lower() # remove double or more whitespace, remove in first and/or last whitespace, and transform to lowercase

        clean_text.append(text)

    

    return clean_text
all_data['text'] = preprocess(all_data.text)

# print(preprocess(all_data.text.to_list()[:10]))
def words_generator(texts, n_grams=2, get_top_50=True):

    gram2idx = defaultdict()

    for line in texts:

    #     token = nltk.word_tokenize(line)

        word_list = line.split(' ')

        filtered_words = [word.strip() for word in word_list if word not in stopwords.words('english')]

        grams = list(ngrams(filtered_words, n_grams)) 

#         if n_grams > 1:

        # merge list of tuple

        grams_merge = []

        for g in grams:

            g = ' '.join(map(str,g))

            grams_merge.append(g)

        grams = grams_merge

        for g in grams:

            if len(g.strip()) > 3 and any([char_g.isalpha() for char_g in set(g.strip())]):

                try:

                    gram2idx[g] += 1

                except:

                    gram2idx[g] = 1

                

    if get_top_50:

#         print(len(sorted(gram2idx.items(), key=lambda item: item[1])))

        gram2idx = {k: v for k, v in sorted(gram2idx.items(), key=lambda item: item[1], reverse=True)[:50]}



    else:

        gram2idx = {k: v for k, v in sorted(gram2idx.items(), key=lambda item: item[1], reverse=True)}



    return gram2idx
realdis_trigram_count = words_generator(all_data[all_data.target == 1].text[::], n_grams=3, get_top_50=True)

notdis_trigram_count = words_generator(all_data[all_data.target == 0].text[::], n_grams=3, get_top_50=True)

realdis_bigram_count = words_generator(all_data[all_data.target == 1].text[::], n_grams=2, get_top_50=True)

notdis_bigram_count = words_generator(all_data[all_data.target == 0].text[::], n_grams=2, get_top_50=True)

realdis_words_count = words_generator(all_data[all_data.target == 1].text[::], n_grams=1, get_top_50=True)

notdis_words_count = words_generator(all_data[all_data.target == 0].text[::], n_grams=1, get_top_50=True)
fig, axes = plt.subplots(ncols=2, figsize=(7, 25), dpi=300)



sns.barplot(y=list(realdis_trigram_count.keys()), x=list(realdis_trigram_count.values()), 

            palette='coolwarm', ax=axes[0]).set_title('Top 50 Trigrams in Real Disaster Tweets')



sns.barplot(y=list(notdis_trigram_count.keys()), x=list(notdis_trigram_count.values()), 

            palette='coolwarm', ax=axes[1]).set_title('Top 50 Trigrams in Not Disaster Tweets')



plt.subplots_adjust(left=0.2, right=2)

plt.show()
fig, axes = plt.subplots(ncols=2, figsize=(7, 25), dpi=300)



sns.barplot(y=list(realdis_bigram_count.keys()), x=list(realdis_bigram_count.values()), 

            palette='coolwarm', ax=axes[0]).set_title('Top 50 Bigrams in Real Disaster Tweets')



sns.barplot(y=list(notdis_bigram_count.keys()), x=list(notdis_bigram_count.values()), 

            palette='coolwarm', ax=axes[1]).set_title('Top 50 Bigrams in Not Disaster Tweets')



plt.subplots_adjust(left=0.2, right=2)

plt.show()
fig, axes = plt.subplots(ncols=2, figsize=(7, 25), dpi=300)



sns.barplot(y=list(realdis_words_count.keys()), x=list(realdis_words_count.values()), 

            palette='coolwarm', ax=axes[0]).set_title('Top 50 Words Count in Real Disaster Tweets')



sns.barplot(y=list(notdis_words_count.keys()), x=list(notdis_words_count.values()), 

            palette='coolwarm', ax=axes[1]).set_title('Top 50 Words Count in Not Disaster Tweets')



plt.subplots_adjust(left=0.2, right=2)

plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Create and generate a word cloud image:

def show_word_cloud(text):

    wordcloud = WordCloud().generate(text)



    # Display the generated image:

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()

words_real_disaster = ' '.join(all_data[all_data.target == 1].text.to_list())

words_not_disaster = ' '.join(all_data[all_data.target == 0].text.to_list())
show_word_cloud(words_real_disaster)
show_word_cloud(words_not_disaster)
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# from sklearn.metrics import f1_score

# from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
# train test split with stratify

train['clean_text'] = preprocess(train.text)

train = train.drop_duplicates(subset='clean_text', keep='first')

print('length train data', len(train))

print('split 80:20')

X = train.clean_text

y = train.target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# summarize train and test composition

train_0, train_1 = len(y_train[y_train==0]), len(y_train[y_train==1])

test_0, test_1 = len(y_val[y_val==0]), len(y_val[y_val==1])

print('Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

cat_0 = train_0 + test_0

cat_1 = train_1 + test_1

print('Train: 0=%f, 1=%f '% (train_0/cat_0, train_1/cat_1))

print('Test: 0=%f, 1=%f'% (test_0/cat_0, test_1/cat_1))
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier



from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import FunctionTransformer
np.random.seed(42)
model_sgd = SGDClassifier()

model_lsvm = LinearSVC() # linear support vector classifier

model_rf = RandomForestClassifier()

model_lr = LogisticRegression()

model_r = RidgeClassifier()



eclf = VotingClassifier(estimators=[('lr', model_lr), ('sgd', model_sgd), ('lsvm', model_lsvm)], voting='hard')



for model, label in zip([model_sgd, model_lsvm, model_rf, model_r, model_lr, eclf], 

                        ['SGD', 'LinearSVC', 'RandomForest', 'RidgeClassifier', 'LogisticRegression', 'VotingEnsemble']):

    clf = Pipeline([

            ('vect', CountVectorizer(ngram_range=(1,1))),

            ('tfidf', TfidfTransformer()),

            ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)), 

            ('model', model)

            ])

    clf.fit(X_train,y_train)

    print("Acc: %f using" % clf.score(X_val, y_val), f'[{label}]')
train_acc = clf.score(X_train, y_train)

val_acc = clf.score(X_val, y_val)

print('train_acc:', train_acc)

print('val_acc:', val_acc)
# download glove vector

!wget -O "/kaggle/working/glove.twitter.27B.zip" "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
!unzip -j "/kaggle/working/glove.twitter.27B.zip" "glove.twitter.27B.200d.txt"
import torch

import torch.nn as nn





import torch.optim as optim

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



import random
random.seed(360)

np.random.seed(360)

torch.manual_seed(360)

torch.backends.cudnn.deterministic = True

torch.cuda.manual_seed_all(360)
word2idx = {}

idx2word = {}



all_text = preprocess(train.text) + preprocess(test.text)

idx = 1 # index start from one because zero for padding

for text in all_text:

    words = nltk.word_tokenize(text)

    for word in words:

        if word not in word2idx.keys():

            word2idx[word] = idx

            if idx not in idx2word.keys():

                idx2word[idx] = word

                idx += 1
print(len(word2idx), len(idx2word))
# load glove

GLOVE_DIR = '/kaggle/working/'

embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.200d.txt'))

for line in tqdm(f, desc='load glove embedding'):

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

    

f.close()
# build embedding matrix

embedding_matrix = np.zeros((len(word2idx) + 2, 200)) # +2 for padding and unknown words

for word, idx in tqdm(word2idx.items(), desc='build embedding matrix'):

    try:

        embedding_matrix[idx] = embeddings_index[word]

    except:

        embedding_matrix[-1] = np.random.rand(1, 200)
def generate_data_loader(x, y, maxlen=128, with_target=True, bs=16):

    all_data = []



    for text, label in zip(x, y): 

        tokens = word_tokenize(text)

        token_ids = [word2idx[t] for t in tokens]

        # padding or truncate to maxlen

        if len(token_ids) > maxlen:

            token_ids = token_ids[:maxlen]

        else:

            token_ids = token_ids + ([0] * (maxlen - len(token_ids)))

        

        token_ids_tensor = torch.LongTensor(token_ids)

        

        target = torch.tensor(label)

        data = {

            'token_ids': token_ids_tensor,

            'target' : target

        }

            

        all_data.append(data)

    

    data_loader = DataLoader(all_data, batch_size=bs, num_workers=5)

    

    return data_loader
train_dataloader = generate_data_loader(X_train, y_train)

val_dataloader = generate_data_loader(X_val, y_val)

# all_dataloader = generate_data_loader(X, y)
def get_n_correct(y_true, y_pred):

    """ 

    y_true : tensor,

    y_pred : tensor,

    example: 

    y_true = torch.tensor([0.0, 1.0, 1.0])

    y_pred = torch.tensor([0.3, 0.6, 0.9])

    torch.eq is means equal of each tensor and return True or False, 

    and then the tensor of booleans turn to floats in 1.0 or 0.0

    torch.eq(y_true, torch.round(y_pred)).float() = torch.tensor([1.0, 1.0, 1.0])

    """

    y_pred = torch.sigmoid(y_pred) # apply logits y_pred to sigmoid activation function 

    

    with torch.no_grad():

        n_correct = torch.sum(torch.eq(y_true, torch.round(y_pred)).float()).item()

    return n_correct

def evaluate(data_loader, criterion, model):

    n_val_total, n_val_correct, val_loss_total = 0, 0, 0

    model.eval()

    with torch.no_grad():

        for sample_batched in data_loader:



            inputs = sample_batched['token_ids'].to(device)

            outputs = model(inputs)



            targets = sample_batched['target'].to(device).float().unsqueeze(1)



            loss = criterion(outputs, targets) #y_pred, y



            n_val_correct += get_n_correct(targets, outputs)

            n_val_total += len(outputs)

            val_loss_total += loss.item() * len(outputs)

    

    val_acc = n_val_correct / n_val_total

    val_loss = val_loss_total / n_val_total

    return val_acc, val_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

    print(torch.cuda.get_device_name())

    print(device)

else:

    print(device)
class LSTMModel(nn.Module):

    def __init__(self, embedding_matrix, output_size=1, embedding_dim=200, hidden_dim=300, hidden_dim_2=100, 

                 n_layers=1, drop_prob=0.5, bidirectional=False, pooling=False):

        super().__init__()

        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

        

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        self.dropout = nn.Dropout(drop_prob)

        self.pooling = pooling

        self.bidirectional = bidirectional

        self.fc_b_pool = nn.Linear(hidden_dim*4, hidden_dim*2)

        if self.bidirectional or self.pooling:

            self.fc = nn.Linear(hidden_dim*2, hidden_dim)

            self.fc_out = nn.Linear(hidden_dim, output_size)

        else:

            self.fc = nn.Linear(hidden_dim, hidden_dim_2)

            self.fc_out = nn.Linear(hidden_dim_2, output_size)

    

    def forward(self, x):

        embeds = self.embedding(x)

        len_seq = torch.as_tensor((x != 0).sum(dim=1), dtype=torch.int64) # sum of token ids without padding ids (0s)



        packed_seq = pack_padded_sequence(embeds, len_seq, batch_first=True, enforce_sorted=False)



        out_packed_lstm, (h_lstm, c_lstm) = self.lstm(packed_seq)

        

        if self.pooling:

            out_lstm, _ = pad_packed_sequence(out_packed_lstm, batch_first=True)

            mean_out = torch.mean(out_lstm, dim=1) # lstm: batch_size, 1, hidden | bilstm: batch_size, 2, hidden

            max_out, _ = torch.max(out_lstm, dim=1) # lstm: batch_size, 1, hidden | bilstm: batch_size, 2, hidden

            h_lstm = torch.cat([mean_out, max_out], dim=1) # lstm: batch_size, 2, hidden | bilstm: batch_size, 4, hidden

            h_lstm = h_lstm.view(h_lstm.shape[0], -1) # batch_size, seq_len, hidden > batch_size, hidden

            if self.bidirectional:

                h_lstm = self.fc_b_pool(h_lstm)

        else:

            h_lstm = h_lstm.transpose(0,1) # seq_len, batch_size, hidden > batch_size, seq_len, hidden (lstm: bs, 1, hidden | bilstm or pool: bs, 2, hidden)

            h_lstm = h_lstm.contiguous().view(h_lstm.shape[0], -1) # batch_size, seq_len, hidden > batch_size, hidden

        

        out = self.dropout(self.fc(h_lstm))

        out = self.fc_out(out)

        return out

    
# train model

def train_model(model, train_dataloader, val_dataloader, learning_rate=1e-3, epochs=50, early_stopping=10, model_name='model_disaster_lstm.pt'):

    # don't worry with higher epochs, we set early stopping

#     wandb.init(project="disaster-tweet-classification", name=model_name)

#     wandb.config.lr = learning_rate



    PATH_OUTPUT_MODEL = '/kaggle/working'

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    criterion = nn.BCEWithLogitsLoss()

    min_val_loss = None

    patience = 0

    list_lr, list_acc, list_loss, list_val_acc, list_val_loss = [], [], [], [], []

    model = model.to(device)

    for i in tqdm(range(epochs), desc='epochs'):

        n_total, n_correct, loss_total = 0, 0, 0



        model.train()



        for sample_batched in tqdm(train_dataloader, desc='batch loader'):

            optimizer.zero_grad()



            inputs = sample_batched['token_ids'].to(device)

            outputs = model(inputs)



            targets = sample_batched['target'].to(device).float().unsqueeze(1)



            loss = criterion(outputs, targets) #y_pred, y_true. y_pred already applied sigmoid by loss function



            loss.backward()

            optimizer.step()



            with torch.no_grad():

                n_correct += get_n_correct(targets, outputs)

                n_total += len(outputs)

                loss_total += loss.item() * len(outputs)



        train_acc = n_correct / n_total

        train_loss = loss_total / n_total



        learning_rate = optimizer.param_groups[0]['lr']

        print(f'Epoch {i} :')

        print('LR:', learning_rate)

        print('Acc:', train_acc)

        print('Loss:', train_loss)



        val_acc, val_loss = evaluate(val_dataloader, criterion, model)



        if min_val_loss == None:

            min_val_loss = val_loss



        if min_val_loss > val_loss:

            min_val_loss = val_loss

            print('Found Best Val Loss...')

            torch.save(model.state_dict(), os.path.join(PATH_OUTPUT_MODEL, model_name))

            patience = 0 # reset patience

        else:

            patience += 1

            scheduler.step()





        print('Val Acc:', val_acc)

        print('Val Loss:', val_loss)



#         wandb.log({"lr":learning_rate, "loss": train_loss, "val_loss": val_loss, 

#                    "acc":train_acc, "val_acc":val_acc})

        

        list_lr.append(learning_rate)

        list_loss.append(train_loss)

        list_val_loss.append(val_loss)

        list_acc.append(train_acc)

        list_val_acc.append(val_acc)

        

        if patience == early_stopping:

            print(f'Validation Loss not decreasing {early_stopping} times in a row..')

            print('Early Stopping..')

            print('Best Val Loss:', min_val_loss)

            break   

    

    return min_val_loss, (list_lr, list_acc, list_loss, list_val_acc, list_val_loss)
# pd.DataFrame({'train_acc':[0.5, 0.6, 0.9, 0.2, 0.4], 'loss':[], 'lr':[], 'metrics':['a','a','a','b','b'], 'epochs':[1,2,3,1,2], })

class MetricsBuilder(object):

    def __init__(self):

        self.df_metrics = pd.DataFrame({})

        

    def add_metrics(self, tuple_metrics, name='LSTM'):

        list_lr, list_acc, list_loss, list_val_acc, list_val_loss = tuple_metrics

        metrics_data = pd.DataFrame({'acc':list_acc, 'val_acc':list_val_acc, 'loss':list_loss, 'val_loss':list_val_loss, 'lr':list_lr,

                                     'metrics': [name for i in range(len(list_lr))], 'epochs':[i for i in range(len(list_lr))]})

        self.df_metrics = pd.concat([self.df_metrics, metrics_data])

    

    def __get_df_metrics__(self):

        return self.df_metrics



metrics_builder = MetricsBuilder()
lstm_model = LSTMModel(embedding_matrix, bidirectional=False)

min_val_loss_lstm, metrics_lstm = train_model(lstm_model, train_dataloader, val_dataloader, model_name='lstm.pt')
lstm_model_pool = LSTMModel(embedding_matrix, bidirectional=False, pooling=True)

min_val_loss_lstm_pool, metrics_lstm_pool = train_model(lstm_model_pool, train_dataloader, val_dataloader, model_name='lstm_pool.pt')
bilstm_model = LSTMModel(embedding_matrix, bidirectional=True)

min_val_loss_bilstm, metrics_bilstm = train_model(bilstm_model, train_dataloader, val_dataloader, model_name='bilstm.pt')
bilstm_model_pool = LSTMModel(embedding_matrix, bidirectional=True, pooling=True)

min_val_loss_bilstm_pool, metrics_bilstm_pool = train_model(bilstm_model_pool, train_dataloader, val_dataloader, model_name='bilstm_pool.pt')
print('min_val_loss_bilstm',min_val_loss_bilstm)

print('min_val_loss_bilstm_pool',min_val_loss_bilstm_pool)

print('min_val_loss_lstm',min_val_loss_lstm)

print('min_val_loss_lstm_pool',min_val_loss_lstm_pool)
metrics_builder.add_metrics(metrics_lstm, name='LSTM')

metrics_builder.add_metrics(metrics_lstm_pool, name='LSTM_Pool')

metrics_builder.add_metrics(metrics_bilstm, name='BiLSTM')

metrics_builder.add_metrics(metrics_bilstm_pool, name='BiSTM_Pool')
df_metrics = metrics_builder.__get_df_metrics__()
fig, ax = plt.subplots(5, figsize=(15,35))

ax1, ax2, ax3, ax4, ax5 = ax

sns.lineplot(x='epochs', y='acc', data=df_metrics, hue="metrics", style='metrics',

                  markers=True, ax=ax1).set_title('Training Accuracy', fontsize=15)

sns.lineplot(x='epochs', y='val_acc', data=df_metrics, hue="metrics", style='metrics',

                  markers=True, ax=ax2).set_title('Val Accuracy', fontsize=15)

sns.lineplot(x='epochs', y='loss', data=df_metrics, hue="metrics", style='metrics',

                  markers=True, ax=ax3).set_title('Training Loss', fontsize=15)

sns.lineplot(x='epochs', y='val_loss', data=df_metrics, hue="metrics", style='metrics',

                  markers=True, ax=ax4).set_title('Val Loss', fontsize=15)

sns.lineplot(x='epochs', y='lr', data=df_metrics, hue="metrics", style='metrics',

                  markers=True, ax=ax5).set_title('Learning Rate', fontsize=15)



plt.subplots_adjust(left=3, right=4)



plt.show()
best_bilstm_pool = LSTMModel(embedding_matrix, bidirectional=True, pooling=True).to(device)

best_bilstm_pool.load_state_dict(torch.load('/kaggle/working/bilstm_pool.pt', map_location=device))
def predict_test_set(data_loader, model):

    model.eval()

    all_outputs = []

    with torch.no_grad():

        for sample_batched in data_loader:



            inputs = sample_batched.to(device)

            outputs = model(inputs)

            outputs = (torch.sigmoid(outputs).squeeze(1) > 0.5).long().tolist()



            all_outputs.extend(outputs)

    

    return all_outputs

def generate_test_data_loader(x, maxlen=128, bs=16):

    all_data = []



    for text in x: 

        tokens = word_tokenize(text)

        token_ids = [word2idx[t] for t in tokens]

        # padding or truncate to maxlen

        if len(token_ids) > maxlen:

            token_ids = token_ids[:maxlen]

        else:

            token_ids = token_ids + ([0] * (maxlen - len(token_ids)))

        

        token_ids_tensor = torch.LongTensor(token_ids)           

        all_data.append(token_ids_tensor)

    

    data_loader = DataLoader(all_data, batch_size=bs, num_workers=5)

    

    return data_loader
test_dataloader = generate_test_data_loader(preprocess(test['text']))

predicted_target = predict_test_set(test_dataloader, best_bilstm_pool)

# predicted_target = blending_test_set(test_dataloader, lstm_model_pool, lstm_model, bilstm_model_pool)

# Use the model to make predictions



my_submission = pd.DataFrame({'Id': test.id, 'target': predicted_target})

my_submission.to_csv('submission.csv', index=False)



print(predicted_target[:3])