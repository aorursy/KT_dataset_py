from IPython.display import Image

Image(filename='../input/bert-sentiment-architecture/sentiment_arch_generalised.png')
import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, f1_score
data = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)

data.columns = ('target','uid', 'time', 'query', 'user', 'text')
data.head(3)
def preprocess_text(tweet):

    tweet = tweet.lower()

    # replace links with 'url'

    tweet = re.sub(r'((https?:\/\/)|(www\.))[A-Za-z0-9.\/]+', 'url',  tweet)

    tweet = re.sub(r'[A-Za-z0-9]+.com', 'url',tweet)

    # remove , @users, if any

    tweet = re.sub(r'[@][A-Za-z0-9]+', '',tweet)

    # remove non-ascii characters

    tweet = ''.join([w for w in tweet if ord(w)<128])

    #get hastags

    """

    # bert tokenizer takes care of such tokens, which have some punctuation attached to it

    # hastags will be taken care of 

    tags = ' '.join([w.strip("#") for w in tweet.split() if w.startswith("#")])

    tweet = re.sub(r'[#][A-Za-z0-9]+', '',tweet)

    """

    tweet = tweet.strip()

    # return tweet, tags

    return tweet
# Creating processed dataframe

sent_df = pd.DataFrame(None, columns=('target', 'text'))

sent_df['target'] = data['target']

sent_df['text'] = data['text'].apply(preprocess_text)

sent_df['tweet_size'] = data['text'].apply(lambda x:len(x.split()))
sent_df.head(3)
sent_df['tweet_size'].hist(bins = 4)
# max tweet size

np.max(sent_df['tweet_size'])
import transformers

from transformers import BertModel, BertTokenizer, AdamW

from transformers import get_linear_schedule_with_warmup

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader
class SentConfig():

    """

    Constants, config used during the project.

    """

    BERT_PRETRAINED = 'bert-base-uncased'

    TOKENIZER = BertTokenizer.from_pretrained(BERT_PRETRAINED)

    BERT_HIDDEN_SIZE = 768

    MAX_TOKENS_LEN = 128

    TRAIN_BATCH_SIZE = 64

    VALID_BATCH_SIZE = 32

    EPOCHS = 3

    SEED = 10
# Select random subset of 3L samples, 1.50L from each category. Also tweets must have more than 10 words in it. 



sent_df_sample = sent_df[(sent_df['tweet_size']>10) & (sent_df['target']==0)].sample(n=250000, random_state=SentConfig.SEED)

sent_df_sample = sent_df_sample.append(sent_df[(sent_df['tweet_size']>10) & (sent_df['target']==4)].sample(n=250000, random_state=SentConfig.SEED))

class SentimentDL():

    """

    DataLoader class, it employs mechanishm of tokenisation using bert, truncation and padding .

    :param modified_df: any dataframe train, test, validataion

    :return: DataLoader type object for the provided dataframe

    """

    def __init__(self, modified_df):

        self.mdf = modified_df

    

    def __len__(self):

        return self.mdf.shape[0]

    

    def __getitem__(self, index_num):

        """

        :return: dictionary of the BERT Tokenizer representaion, and the corresponding sentiment represented in long tensor (target varible)

        """

        row = self.mdf.iloc[index_num]

        tweet = row['text']

        target = 0 if int(row['target']) is 0 else 1

        # {0:'positive class', 1:'negative class'}

    

        tw_bert_tok = SentConfig.TOKENIZER(tweet)



        tw_input_ids = tw_bert_tok['input_ids']

        tw_mask = tw_bert_tok['attention_mask']

        tw_tt_ids = tw_bert_tok['token_type_ids']

    

        len_ = len(tw_input_ids)

        if len_ > SentConfig.MAX_TOKENS_LEN:

          tw_input_ids = tw_input_ids[:SentConfig.MAX_TOKENS_LEN-1]+[102]

          tw_mask = tw_mask[:SentConfig.MAX_TOKENS_LEN]

          tw_tt_ids = tw_tt_ids[:SentConfig.MAX_TOKENS_LEN]

        elif len_ < SentConfig.MAX_TOKENS_LEN:

          pad_len = SentConfig.MAX_TOKENS_LEN - len_

          tw_input_ids = tw_input_ids + ([0] * pad_len)

          tw_mask = tw_mask + ([0] * pad_len)

          tw_tt_ids = tw_tt_ids + ([0] * pad_len)

        return {

            'input_ids':torch.tensor(tw_input_ids, dtype=torch.long),

            'attention_mask':torch.tensor(tw_mask, dtype=torch.long),

            'token_type_ids':torch.tensor(tw_tt_ids, dtype=torch.long),

            'target':torch.tensor(target, dtype=torch.long)

        }
# get train, test, validation set from our preprocessed sample of 3L.



train, test = train_test_split(sent_df_sample, test_size=0.1)

train, val = train_test_split(train, test_size=0.05)
# create necessary data loader objects



train_dl = SentimentDL(train)

val_dl = SentimentDL(val)

test_dl = SentimentDL(test)



train_loader = DataLoader(train_dl, batch_size=SentConfig.TRAIN_BATCH_SIZE, shuffle=True)

validation_loader = DataLoader(val_dl, batch_size=SentConfig.VALID_BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_dl, batch_size=SentConfig.VALID_BATCH_SIZE, shuffle=True)
class SentimentModel(nn.Module):

    def __init__(self):

        super(SentimentModel, self).__init__()

        self.bert = BertModel.from_pretrained(SentConfig.BERT_PRETRAINED)

        self.drop_out = nn.Dropout(0.30)  # more dropout value for regularisation

        self.linear1 = nn.Linear(SentConfig.BERT_HIDDEN_SIZE, 2)

    

    def forward(self, input_ids, attention_mask, tt_ids):

        """

        This is forward step of our model. It takes, BERT Tokenizer generated representation of sentence, 

        and passes it through the bert -> dropout -> linear layers repectively.

    

        Note: out_ = token wise output (ignored) from the bert layer, as in our task (sentiment analysis) we need aggregated output 

        """

        out_, pooled_out = self.bert(input_ids, attention_mask, tt_ids)

        out = self.drop_out(pooled_out)

        out = self.linear1(out)

        return out
# get device type and accordingly move generated, model object to device.



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = SentimentModel()

model.to(device)


loss_function = nn.CrossEntropyLoss()





# Adam optimizer with weight decay AdamW from transformers

# do not apply weight decay in AdamW  to, bias layer and normalization terms

no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']  # taken from https://huggingface.co/transformers/training.html 

# more named parameteres in model.named_parameters()

optimizer_grouped_parameters = [

    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

]



# optimizer object

optim = AdamW(optimizer_grouped_parameters, lr=4e-5)



# learning rate scheduling

num_train_steps = int((train_dl.__len__()/SentConfig.TRAIN_BATCH_SIZE)*SentConfig.EPOCHS)

num_warmup_steps = int(0.05*num_train_steps)

scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps, num_train_steps)
def train_function(data_loader, model, optimizer, scheduler, device):

    """

    Function to train single epoch on the data accessible with data_loader

    """

    epoch_loss = 0

    model.train()

    for batch in data_loader:

        optimizer.zero_grad()



        input_ids = batch['input_ids'].to(device)

        attention_mask = batch['attention_mask'].to(device)

        token_type_ids = batch['token_type_ids'].to(device)

        target = batch['target'].to(device)



        outputs = model(input_ids, attention_mask, token_type_ids)



        batch_loss = loss_function(outputs, target)

        batch_loss.backward()

        optimizer.step()

        scheduler.step()

        epoch_loss += batch_loss.item()

    epoch_loss = epoch_loss / len(data_loader)

    return epoch_loss





def evaluation_function(data_loader, model, device, inference=False):

    """

    Function to evaluate current model performance.

    """

    epoch_loss = 0

    model.eval()

    results = []

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch['input_ids'].to(device)

            attention_mask = batch['attention_mask'].to(device)

            token_type_ids = batch['token_type_ids'].to(device)

            target = batch['target'].to(device)



            outputs = model(input_ids, attention_mask, token_type_ids)

#             if not inference:

            batch_loss = loss_function(outputs, target)

            epoch_loss += batch_loss.item()

#             else:

            outputs = torch.argmax(outputs, dim=1).to('cpu').numpy()

            target = target.to('cpu').numpy()

            results.extend(list(zip(outputs, target)))

    epoch_loss = epoch_loss / len(data_loader)

    return epoch_loss, np.array(results)
%%time



# Training : done on the basis of validation loss vs training loss

scores = []

min_f1 = 0



for epoch in range(SentConfig.EPOCHS):

  _ = train_function(train_loader, model, optim, scheduler, device)

  _, results = evaluation_function(validation_loader, model, device)

  validation_f1 = round(f1_score(results[:,1], results[:,0]),4)

  accuracy = round(accuracy_score(results[:,1], results[:,0]), 4)

  

  print('epoch num: ', epoch, 'f1 score: ',validation_f1 , 'accuracy: ', accuracy)

  scores.append((validation_f1, accuracy))



  if validation_f1 > min_f1:

        # save  model if validation f1 score is 

        torch.save(model.state_dict(), "SentimentModel5L.bin")

        # update max loss

        min_f1 =  validation_f1
# plotting losses



scores = np.array(scores)

fig, ax = plt.subplots(1, 2, figsize=(14,6))

ax[0].plot(range(SentConfig.EPOCHS), scores[:,0], 'r')

ax[1].plot(range(SentConfig.EPOCHS), scores[:,1])

ax[0].set(xlabel='Epoch num', ylabel='F1 Score')

ax[1].set(xlabel='Epoch num', ylabel='Accuracy')

ax[0].set_title('validation set f1 score at each epoch')

ax[1].set_title('validation set accuracy at each apoch')
# Load model from the trained saved model

state_dict_ = torch.load('SentimentModel5L.bin')

model = SentimentModel()

model.load_state_dict(state_dict_)

model.to(device)



# Calculate F1 score report

_, results = evaluation_function(test_loader, model, device, inference=True)

print(classification_report(results[:,1], results[:,0]))
print(round(accuracy_score(results[:,1], results[:,0]),4))