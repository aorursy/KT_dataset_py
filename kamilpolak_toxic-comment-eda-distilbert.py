from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from datetime import datetime



from scipy.misc import imread

from scipy import sparse

import scipy.stats as ss



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec 

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

from PIL import Image

import matplotlib_venn as venn



import string

import re    #for regex

import nltk

from nltk.corpus import stopwords

import spacy

from nltk import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

# Tweet tokenizer does not split at apostophes which is what we want

from nltk.tokenize import TweetTokenizer



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.decomposition import TruncatedSVD



import gc

import time

import warnings



#settings

start_time=time.time()

color = sns.color_palette()

sns.set_style("dark")

eng_stopwords = set(stopwords.words("english"))

warnings.filterwarnings("ignore")



lem = WordNetLemmatizer()

tokenizer=TweetTokenizer()



%matplotlib inline
!pip install transformers
import sys

import copy

import torch 

from scipy.sparse import *

from sklearn.metrics import roc_auc_score

import pyarrow as pa



import torch.nn as nn

from torch.optim import lr_scheduler

import torch.nn.functional as F

from torchvision import datasets, models, transforms

from torch.utils.data import Dataset,DataLoader

from transformers import DistilBertConfig,DistilBertTokenizer,DistilBertModel





if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
train=pd.read_csv("../input/toxic-comments/train.csv")

test=pd.read_csv("../input/toxic-comments/test.csv")
train.head()
print("Check for missing values in Train dataset")

null_check=train.isnull().sum()

print(null_check)

print("Check for missing values in Test dataset")

null_check=test.isnull().sum()

print(null_check)

print("filling NA with \"unknown\"")

train["comment_text"].fillna("unknown", inplace=True)

test["comment_text"].fillna("unknown", inplace=True)
x=train.iloc[:,2:].sum()

#marking comments without any tags as "clean"

rowsums=train.iloc[:,2:].sum(axis=1)

train['clean']=(rowsums==0)

#count number of clean entries

train['clean'].sum()

print("Total comments = ",len(train))

print("Total clean comments = ",train['clean'].sum())

print("Total tags =",x.sum())
x=train.iloc[:,2:].sum()

#plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("# per class")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Type ', fontsize=12)

#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
x=rowsums.value_counts()



#plot

plt.figure(figsize=(8,4))

ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[2])

plt.title("Multiple tags per comment")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('# of tags ', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
merge=pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])

df=merge.reset_index(drop=True)
#Sentense count in each comment:

    #  '\n' can be used to count the number of sentences in each comment

df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

#Word count in each comment:

df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))

#Unique word count

df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))

#Letter count

df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))

#punctuation count

df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#upper case words count

df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

#title case words count

df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Number of stopwords

df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#Average length of the words

df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#derived features

#Word count percent in each comment:

df['word_unique_percent']=df['count_unique_word']*100/df['count_word']

#derived features

#Punct percent in each comment:

df['punct_percent']=df['count_punctuations']*100/df['count_word']
#serperate train and test features

train_feats=df.iloc[0:len(train),]

test_feats=df.iloc[len(train):,]

#join the tags

train_tags=train.iloc[:,2:]

train_feats=pd.concat([train_feats,train_tags],axis=1)
train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10 

plt.figure(figsize=(12,6))

## sentenses

plt.subplot(121)

sns.violinplot(y='count_sent',x='clean', data=train_feats,split=True)

plt.xlabel('Clean?', fontsize=12)

plt.ylabel('# of sentences', fontsize=12)

plt.title("Number of sentences in each comment", fontsize=15)

# words

train_feats['count_word'].loc[train_feats['count_word']>200] = 200

plt.subplot(122)

sns.violinplot(y='count_word',x='clean', data=train_feats,split=True,inner="quart")

plt.xlabel('Clean?', fontsize=12)

plt.ylabel('# of words', fontsize=12)

plt.title("Number of words in each comment", fontsize=15)



plt.show()
train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200

#prep for split violin plots

#For the desired plots , the data must be in long format

temp_df = pd.melt(train_feats, value_vars=['count_word', 'count_unique_word'], id_vars='clean')

#spammers - comments with less than 40% unique words

spammers=train_feats[train_feats['word_unique_percent']<30]
plt.figure(figsize=(16,12))

gridspec.GridSpec(2,2)

plt.subplot2grid((2,2),(0,0))

sns.violinplot(x='variable', y='value', hue='clean', data=temp_df,split=True,inner='quartile')

plt.title("Absolute wordcount and unique words count")

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Count', fontsize=12)



plt.subplot2grid((2,2),(0,1))

plt.title("Percentage of unique words of total words in comment")

#sns.boxplot(x='clean', y='word_unique_percent', data=train_feats)

ax=sns.kdeplot(train_feats[train_feats.clean == 0].word_unique_percent, label="Bad",shade=True,color='r')

ax=sns.kdeplot(train_feats[train_feats.clean == 1].word_unique_percent, label="Clean")

plt.legend()

plt.ylabel('Number of occurances', fontsize=12)

plt.xlabel('Percent unique words', fontsize=12)



x=spammers.iloc[:,-7:].sum()

plt.subplot2grid((2,2),(1,0),colspan=2)

plt.title("Count of comments with low(<30%) unique words",fontsize=15)

ax=sns.barplot(x=x.index, y=x.values,color=color[3])



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.xlabel('Threat class', fontsize=12)

plt.ylabel('# of comments', fontsize=12)

plt.show()
## Feature engineering to prepare inputs for BERT....





Y = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].astype(float)

X = train['comment_text']





X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)
print('train_x shape is {}' .format({X_train.shape}))

print('test_x shape is {}' .format({X_test.shape}))

print('train_y shape is {}' .format({y_train.shape}))
X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
def accuracy_thresh(y_pred, y_true, thresh:float=0.4, sigmoid:bool=True):

    "Compute accuracy when `y_pred` and `y_true` are the same size."

    if sigmoid: y_pred = y_pred.sigmoid()

#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()

    return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()

#Expected object of scalar type Bool but got scalar type Double for argument #2 'other'



config = DistilBertConfig(vocab_size_or_config_json_file=32000,dropout=0.1, num_labels=6, intermediate_size=3072 )




class DistilBertForSequenceClassification(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.num_labels = config.num_labels



        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.seq_classif_dropout)



        nn.init.xavier_normal_(self.classifier.weight)



    def forward(self, input_ids=None, attention_mask=None, head_mask=None, labels=None):

        distilbert_output = self.distilbert(input_ids=input_ids,

                                            attention_mask=attention_mask,

                                            head_mask=head_mask)

        hidden_state = distilbert_output[0]                    

        pooled_output = hidden_state[:, 0]                   

        pooled_output = self.pre_classifier(pooled_output)   

        pooled_output = nn.ReLU()(pooled_output)             

        pooled_output = self.dropout(pooled_output)        

        logits = self.classifier(pooled_output) 

        return logits







max_seq_length = 256

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')





class text_dataset(Dataset):

    def __init__(self,x,y, transform=None):

        

        self.x = x

        self.y = y

        self.transform = transform

        

    def __getitem__(self,index):

        

        tokenized_comment = tokenizer.tokenize(self.x[index])

        

        if len(tokenized_comment) > max_seq_length:

            tokenized_comment = tokenized_comment[:max_seq_length]

            

        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_comment)



        padding = [0] * (max_seq_length - len(ids_review))

        

        ids_review += padding

        

        assert len(ids_review) == max_seq_length

        

        #print(ids_review)

        ids_review = torch.tensor(ids_review)

        

        hcc = self.y[index] # toxic comment        

        list_of_labels = [torch.from_numpy(hcc)]

        

        

        return ids_review, list_of_labels[0]

    

    def __len__(self):

        return len(self.x)

 



text_dataset(X_train, y_train).__getitem__(6)[1]   ### Testing index 6 to see output
batch_size = 32





training_dataset = text_dataset(X_train,y_train)



test_dataset = text_dataset(X_test,y_test)



dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False),

                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                   }

dataset_sizes = {'train':len(X_train),

                'val':len(X_test)}



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = DistilBertForSequenceClassification(config)

model.to(device)



print(device)
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):

    model.train()

    since = time.time()

    print('starting')

    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 100



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            

            beta_score_accuracy = 0.0

            

            micro_roc_auc_acc = 0.0

            

            

            # Iterate over data.

            for inputs, hcc in dataloaders_dict[phase]:

                

                inputs = inputs.to(device) 

                hcc = hcc.to(device)

            

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    

                    loss = criterion(outputs,hcc.float())

                    

                    if phase == 'train':

                        

                        loss.backward()

                        optimizer.step()



                running_loss += loss.item() * inputs.size(0)

                

                micro_roc_auc_acc +=  accuracy_thresh(outputs.view(-1,6),hcc.view(-1,6))

                

                #print(micro_roc_auc_acc)



                

            epoch_loss = running_loss / dataset_sizes[phase]



            

            epoch_micro_roc_acc = micro_roc_auc_acc / dataset_sizes[phase]



            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))

            print('{} micro_roc_auc_acc: {:.4f}'.format( phase, epoch_micro_roc_acc))



            if phase == 'val' and epoch_loss < best_loss:

                print('saving with loss of {}'.format(epoch_loss),

                      'improved over previous {}'.format(best_loss))

                best_loss = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), 'distilbert_model_weights.pth')

         



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(float(best_loss)))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model

 

print('done')
lrlast = .001

lrmain = 3e-5

#optim1 = torch.optim.Adam(

#    [

#        {"params":model.parameters,"lr": lrmain},

#        {"params":model.classifier.parameters(), "lr": lrlast},

#       

#   ])



optim1 = torch.optim.Adam(model.parameters(),lrmain)



optimizer_ft = optim1

criterion = nn.BCEWithLogitsLoss()



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=8)
#y_test = test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

x_test = test['comment_text']

y_test = np.zeros(x_test.shape[0]*6).reshape(x_test.shape[0],6)



test_dataset = text_dataset(x_test,y_test)

prediction_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)



def preds(model,test_loader):

    predictions = []

    for inputs, sentiment in test_loader:

        inputs = inputs.to(device) 

        sentiment = sentiment.to(device)

        with torch.no_grad():

            outputs = model(inputs)

            outputs = torch.sigmoid(outputs)

            predictions.append(outputs.cpu().detach().numpy().tolist())

    return predictions
predictions = preds(model=model_ft1,test_loader=prediction_dataloader)

predictions = np.array(predictions)[:,0]
submission = pd.DataFrame(predictions,columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])

test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]=submission

final_sub = test[['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']]

final_sub.head()