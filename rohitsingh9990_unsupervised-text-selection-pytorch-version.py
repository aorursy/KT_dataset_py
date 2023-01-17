import numpy as np

import pandas as pd

import os

import tokenizers

import string

import torch

import re

import transformers

import torch.nn as nn

from torch.nn import functional as F

from tqdm import tqdm

from sklearn import metrics

import matplotlib.pyplot as plt



from transformers import AdamW

from transformers import get_linear_schedule_with_warmup



from IPython.core.display import display, HTML

train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').head(100).fillna('')

train.head()
import pandas as pd

from sklearn import model_selection





train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

train = train.dropna().reset_index(drop=True)

train["kfold"] = -1





kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



for fold, (trn_, val_) in enumerate(kf.split(X=train, y=train.sentiment.values)):

    print(len(trn_), len(val_))

    train.loc[val_, 'kfold'] = fold



train.to_csv("train_folds.csv", index=False)
MAX_LEN = 96

ROBERTA_PATH = '../input/roberta-base/'

TOKENIZER = tokenizers.ByteLevelBPETokenizer(

    vocab_file=ROBERTA_PATH+'vocab.json', 

    merges_file=ROBERTA_PATH+'merges.txt', 

    lowercase=True,

    add_prefix_space=True

)





EPOCHS = 2

TRAIN_BATCH_SIZE = 64

VALID_BATCH_SIZE = 64

TRAINING = False



sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

sentiment_tar = {'positive': 1, 'negative': 2, 'neutral': 0}

device = torch.device('cuda')

DO_QUES_ANS = False
def process_data(tweet, selected_text, sentiment, tokenizer, max_len, do_ques_ans=False):

    

    # FIND TEXT / SELECTED_TEXT OVERLAP

    tweet = " " + " ".join(str(tweet).split())

    selected_text =  " ".join(str(selected_text).split())

    

    

    start_idx = tweet.find(selected_text)

    end_idx = start_idx + len(selected_text)

    

    char_targets = [0] * len(tweet)

    if start_idx != None and end_idx != None:

        for ct in range(start_idx, end_idx):

            char_targets[ct] = 1  

            

    tok = tokenizer.encode(tweet)

    tweet_ids = tok.ids

    

    # OFFSETS, CHAR CENTERS

    tweet_offsets = tok.offsets

    char_centers = [(offset[0] + offset[1]) / 2 for offset in tweet_offsets]

    

    target_idx = []

    for j, (offset1, offset2) in enumerate(tweet_offsets):

        if sum(char_targets[offset1: offset2]) > 0:

            target_idx.append(j)

            

#     print(target_idx)

    

    targets_start = target_idx[0]

    targets_end = target_idx[-1]



    stok = [sentiment_id[sentiment]]

    

    

    input_ids = [0] + tweet_ids + [2]

    token_type_ids = [0] + [0] * (len(tweet_ids) + 1)

    mask = [1] * len(token_type_ids)

    tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]

    targets_start += 1

    targets_end += 1

        

    if do_ques_ans:

        input_ids = [0] + stok + [2] + [2] + tweet_ids + [2]

        token_type_ids = [0, 0, 0, 0] + [0] * (len(tweet_ids) + 1)

        mask = [1] * len(token_type_ids)

        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

        targets_start += 4

        targets_end += 4





    padding_length = max_len - len(input_ids)

    if padding_length > 0:

        input_ids = input_ids + ([1] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

#         print(char_centers)

        char_centers = char_centers + ([0] * padding_length)

    

    return {

        'ids': input_ids,

        'mask': mask,

        'token_type_ids': token_type_ids,

        'targets_start': targets_start,

        'targets_end': targets_end,

        'orig_tweet': tweet,

        'orig_selected': selected_text,

        'sentiment': sentiment,

        'offsets': tweet_offsets,

        'char_cent': char_centers,

        'sentiment_tar': sentiment_tar[sentiment]

    }





class TweetDataset:

    def __init__(self, tweet, sentiment, selected_text):

        self.tweet = tweet

        self.sentiment = sentiment

        self.selected_text = selected_text

        self.tokenizer = TOKENIZER

        self.max_len = MAX_LEN

    

    def __len__(self):

        return len(self.tweet)



    def __getitem__(self, item):

        data = process_data(

            self.tweet[item], 

            self.selected_text[item], 

            self.sentiment[item],

            self.tokenizer,

            self.max_len

        )



        return {

            'ids': torch.tensor(data["ids"], dtype=torch.long),

            'mask': torch.tensor(data["mask"], dtype=torch.long),

            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),

            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),

            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),

            'orig_tweet': data["orig_tweet"],

            'orig_selected': data["orig_selected"],

            'sentiment': data["sentiment"],

            'offsets': torch.tensor(data["offsets"], dtype=torch.long),

            'char_cent': torch.tensor(data['char_cent'], dtype=torch.long),

            'sentiment_tar': data['sentiment_tar']

        }
class TweetModel(transformers.BertPreTrainedModel):

    """BERT model for QA and classification tasks.

    

    Parameters

    ----------

    config : transformers.BertConfig. Configuration class for BERT.

    Returns

    -------

    classifier_logits : torch.Tensor with shape (batch_size, num_classes).

        Classification scores of each labels.

    """



    def __init__(self, conf):

        super(TweetModel, self).__init__(conf)

        self.roberta = transformers.RobertaModel.from_pretrained(ROBERTA_PATH, config=conf)

        self.drop_out = nn.Dropout(0.1)

        self.classifier = nn.Linear(768, 3)



    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        hidden_states, pooled_output, _ = self.roberta(input_ids,

                            attention_mask=attention_mask,

                            token_type_ids=token_type_ids)

        



        # classification

        pooled_output = self.drop_out(pooled_output)

        classifier_logits = self.classifier(pooled_output)



        return hidden_states, classifier_logits
import torch.nn as nn

import torch

import numpy as np



def loss_fn(output, target):

    loss_func = nn.CrossEntropyLoss()

    loss = loss_func(output, target)

    return loss



# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels, device):

    pred_flat = np.argmax(preds, axis=1).flatten() 

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)





class AverageMeter():

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

        

        

class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.0003):

        self.patience = patience

        self.counter = 0

        self.mode = mode

        self.best_score = None

        self.early_stop = False

        self.delta = delta

        if self.mode == "min":

            self.val_score = np.Inf

        else:

            self.val_score = -np.Inf



    def __call__(self, epoch_score, model, model_path):



        if self.mode == "min":

            score = -1.0 * epoch_score

        else:

            score = np.copy(epoch_score)



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

            self.counter = 0



    def save_checkpoint(self, epoch_score, model, model_path):

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:

            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))

            torch.save(model.state_dict(), model_path)

        self.val_score = epoch_score
def train_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    losses = AverageMeter()

    accuracy_score = AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))



    accr = 0

    

    for bi, d in enumerate(tk0):

        ids = d['ids']

        token_type_ids = d['token_type_ids']

        mask = d['mask']

        targets_start = d['targets_start']

        targets_end = d['targets_end']

        sentiment = d['sentiment']

        orig_selected = d['orig_selected']

        orig_tweet = d['orig_tweet']

        targets_start = d['targets_start']

        targets_end = d['targets_end']

        offsets = d['offsets']

        targets = d['sentiment_tar']



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)

        targets_end = targets_end.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.long)



        model.zero_grad()

        _, outputs = model(

            input_ids=ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )



        loss = loss_fn(outputs, targets)

        loss.backward()

        optimizer.step()

        if scheduler is not None:

            scheduler.step()

            

        outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()

        outputs = np.argmax(outputs, axis=1)

        targets = targets.cpu().detach().numpy()



        accuracy = metrics.accuracy_score(targets, outputs)

        accuracy_score.update(accuracy.item(), ids.size(0))

        losses.update(loss.item(), ids.size(0))

        tk0.set_postfix(loss=losses.avg, accuracy = accuracy_score.avg)

    return losses.avg, accuracy_score.avg

        



def eval_fn(data_loader, model, device):

    model.eval()

    losses = AverageMeter()

    accuracy_score = AverageMeter()



    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        for bi, d in enumerate(tk0):

            ids = d['ids']

            token_type_ids = d['token_type_ids']

            mask = d['mask']

            targets_start = d['targets_start']

            targets_end = d['targets_end']

            sentiment = d['sentiment']

            orig_selected = d['orig_selected']

            orig_tweet = d['orig_tweet']

            targets_start = d['targets_start']

            targets_end = d['targets_end']

            offsets = d['offsets'].numpy()

            targets = d['sentiment_tar']



            ids = ids.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets_start = targets_start.to(device, dtype=torch.long)

            targets_end = targets_end.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.long)



            _, outputs = model(

                input_ids=ids,

                attention_mask=mask,

                token_type_ids=token_type_ids

            )



            loss = loss_fn(outputs, targets)

            outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()

            outputs = np.argmax(outputs, axis=1)

            targets = targets.cpu().detach().numpy().tolist()



            accuracy = metrics.accuracy_score(targets, outputs)

            accuracy_score.update(accuracy.item(), ids.size(0))

            losses.update(loss.item(), ids.size(0))

            tk0.set_postfix(loss=losses.avg, accuracy = accuracy_score.avg)





    return losses.avg, accuracy_score.avg
def run(fold):

    dfx = pd.read_csv('train_folds.csv')



    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)

    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)



    train_dataset = TweetDataset(

        tweet=df_train.text.values,

        sentiment=df_train.sentiment.values,

        selected_text=df_train.selected_text.values

    )



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        num_workers=0

    )



    valid_dataset = TweetDataset(

        tweet=df_valid.text.values,

        sentiment=df_valid.sentiment.values,

        selected_text=df_valid.selected_text.values

    )



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=VALID_BATCH_SIZE,

        num_workers=0

    )

    

    

    device = torch.device('cuda')

    model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)

    model_config.output_hidden_states = True

    model = TweetModel(conf=model_config)

    model.to(device)



    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]



    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(

        optimizer, 

        num_warmup_steps=0, 

        num_training_steps=num_train_steps

    )

    

    es = EarlyStopping(patience=2, mode='max')

    print(f"Training is Starting for fold={fold}")



    for epoch in range(EPOCHS):

        train_loss, train_acc = train_fn(train_data_loader, model, optimizer, device, scheduler)

#         print(f'Training loss is {train_loss}, Training Accuracy is {train_acc}')

        val_loss, val_acc = eval_fn(valid_data_loader, model, device)

        print(f'Validation loss is {val_loss}, Validation Accuracy is {val_acc}')

        es(val_acc, model, model_path=f'model_{fold}.bin')

        if es.early_stop:

            print('Early Stopping')

            break
# training only for fold = 0



if TRAINING:

    run(0)

model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)

model_config.output_hidden_states = True

model = TweetModel(conf=model_config)



if TRAINING:

    model.load_state_dict(torch.load('model_0.bin'))

else:

    model.load_state_dict(torch.load('../input/roberta-base-cam/model_0.bin'))

    

model.to(device)
pr = {0:'NEUTRAL',1:'POSITIVE',2:'NEGATIVE'}
import os

import argparse



import cv2

import torch

import numpy as np

from torch.nn import functional as F

import torchvision.transforms as transforms







def create_cam(model, device):

    dfx = pd.read_csv('train_folds.csv')

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True).tail(90)



    train_dataset = TweetDataset(

        tweet=df_train.text.values,

        sentiment=df_train.sentiment.values,

        selected_text=df_train.selected_text.values

    )



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=VALID_BATCH_SIZE,

        num_workers=0

    )

    

    

    finalconv_name = 'classifier'

    # hook

    feature_blobs = []

    def hook_feature(module, input, output):

        feature_blobs.append(output.cpu().data.numpy())



    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    params = list(model.parameters())

    # get weight only from the last layer(linear)

    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    



    

    for bi, d in enumerate(train_data_loader):

        

        ids = d['ids']

        token_type_ids = d['token_type_ids']

        mask = d['mask']

        sentiment = d['sentiment']

        orig_tweet = d['orig_tweet']

        targets = d['sentiment_tar']

        orig_selected = d['orig_selected']

        char_cent = d['char_cent']



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.long)

        char_cent = char_cent.to(device, dtype=torch.long)



        hidden_states, outputs = model(

            input_ids=ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )

    

        mask = mask.cpu().detach().numpy().tolist()  

        char_cent = char_cent.cpu().detach().numpy().tolist()  

    

        for index in range(0, ids.shape[0]):



            last_conv_output = hidden_states[index]

            pred_vec = outputs[index]

            last_conv_output = np.squeeze(last_conv_output)

            last_conv_output = torch.sigmoid(last_conv_output).cpu().detach().numpy().tolist() 



            pred_vec = torch.sigmoid(pred_vec).cpu().detach().numpy().tolist()  

            pred = np.argmax(pred_vec)

            layer_weights = weight_softmax[pred, :]

            final_output = np.dot(last_conv_output, layer_weights)

            

            

            if len(orig_tweet[index])>95: continue #too wide for display

            if pr[pred]!=sentiment[index].upper(): continue #skip misclassified



            # PLOT INFLUENCE VALUE

            print()

            plt.figure(figsize=(20,3))

            

            idx = np.sum(mask[index])

            v = np.argsort(final_output[:idx-1])



            mx = final_output[v[-1]]; x = max(-10,-len(v))

            mn = final_output[v[x]]

            

            plt.plot(char_cent[index][:idx-2],final_output[1:idx-1],'o-')

            plt.plot([1,95],[mn,mn],':')

            plt.xlim((0,95))

            plt.yticks([]); plt.xticks([])

            plt.title(f'Predict label is {pr[pred]} True label is {sentiment[index]}',size=16)

            plt.show()



            # DISPLAY ACTIVATION TEXT

            html = ''

            for j in range(1,idx):

                x = (final_output[j]-mn)/(mx-mn)

                html += "<span style='background:{};font-family:monospace'>".format('rgba(255,255,0,%f)'%x)

                tokenizer = TOKENIZER

                html += tokenizer.decode( [ids[index][j]] )

                html += "</span>"

            html += " (predict)"

            display(HTML(html))





            # DISPLAY TRUE SELECTED TEXT

            tweet = " ".join(orig_tweet[index].lower().split()) 

            selected_text = " ".join(orig_selected[index].lower().split())

            sp = tweet.split(selected_text)

            html = "<span style='font-family:monospace'>"+sp[0]+"</span>"

            for j in range(1,len(sp)):

                html += "<span style='background:yellow;font-family:monospace'>"+selected_text+'</span>'

                html += "<span style='font-family:monospace'>"+sp[j]+"</span>"

            html += " (true)"

            display(HTML(html))

            print()

create_cam(model, device)