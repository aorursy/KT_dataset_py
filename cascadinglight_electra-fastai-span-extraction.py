from transformers import AutoModelForQuestionAnswering, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import os
from itertools import compress
from torch.utils.data import DataLoader
from functools import partial
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from fastai.core import *
from fastai.text import *
file_dir, electra_dir = [Path(f'/kaggle/input/{i}') for i in ['tweet-sentiment-extraction', 'electrabase']]

train_df = pd.read_csv(file_dir/'train.csv')
train_df['text'] = train_df['text'].apply(lambda x: str(x))
train_df['sentiment'] = train_df['sentiment'].apply(lambda x: str(x))
train_df['selected_text'] = train_df['selected_text'].apply(lambda x: str(x))

test_df = pd.read_csv(file_dir/'test.csv')
test_df['text'] = test_df['text'].apply(lambda x: str(x))
test_df['sentiment'] = test_df['sentiment'].apply(lambda x: str(x))
max_len = 128
bs = 64
tokenizer = BertWordPieceTokenizer(str(electra_dir/'vocab.txt'), lowercase=True)
def preprocess(sentiment, tweet, selected, tokenizer, max_len):
    _input = tokenizer.encode(sentiment, tweet)
    _span = tokenizer.encode(selected, add_special_tokens=False)
    
    len_span = len(_span.ids)
    start_idx = None
    end_idx = None
    
    for ind in (i for i, e in enumerate(_input.ids) if e == _span.ids[0]):
        if _input.ids[ind: ind + len_span] == _span.ids:
            start_idx = ind
            end_idx = ind + len_span - 1
            break
    
    # Handles cases where Wordpiece tokenizing input & span separately produces different outputs
    if not start_idx:
        idx0 = tweet.find(selected)
        idx1 = idx0 + len(selected)
        
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1):
                char_targets[ct] = 1
                
        tweet_offsets = list(compress(_input.offsets, _input.type_ids))[0:-1]
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
                
        start_idx, end_idx = target_idx[0] +3 , target_idx[-1] + 3
        
    _input.start_target = start_idx
    _input.end_target = end_idx
    _input.tweet = tweet
    _input.sentiment = sentiment
    _input.selected = selected
    
    _input.pad(max_len)
    
    return _input
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss    

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction
    
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return torch.lerp(nll, loss/c, self.ε) 
class TweetDataset(Dataset):
    def __init__(self, dataset, test = None):
        self.df = dataset
        self.test = test
        
    def __getitem__(self, idx):
        if not self.test:
            sentiment, tweet, selected = (self.df[col][idx] for col in ['sentiment', 'text', 'selected_text'])
            _input = preprocess(sentiment, tweet, selected, tokenizer, max_len)
            
            yb = [torch.tensor(_input.start_target), torch.tensor(_input.end_target)]
            
        else:
            _input = tokenizer.encode(self.df.sentiment[idx], self.df.text[idx])
            _input.pad(max_len)
            
            yb = 0

        xb = [torch.LongTensor(_input.ids),
              torch.LongTensor(_input.attention_mask),
              torch.LongTensor(_input.type_ids),
              np.array(_input.offsets)]

        return xb, yb     

    def __len__(self):
        return len(self.df)
pt_model = AutoModel.from_pretrained(electra_dir)
class SpanModel(nn.Module):
    def __init__(self,pt_model):
        super().__init__()
        self.model = pt_model
        self.drop_out = nn.Dropout(0.5)
        self.qa_outputs1c = torch.nn.Conv1d(768*2, 128, 2)
        self.qa_outputs2c = torch.nn.Conv1d(768*2, 128, 2)
        self.qa_outputs1 = nn.Linear(128, 1)
        self.qa_outputs2 = nn.Linear(128, 1)
        
#         self.qa_outputs = nn.Linear(768 * 2, 2) # update hidden size

    # could pass offsets here and not use - can grab in last_input
    def forward(self, input_ids, attention_mask, token_type_ids, offsets = None):
        
        _, hidden_states = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        out = self.drop_out(out)
        out = torch.nn.functional.pad(out.transpose(1,2), (1, 0))
        
        out1 = self.qa_outputs1c(out).transpose(1,2)
        out2 = self.qa_outputs2c(out).transpose(1,2)

        start_logits = self.qa_outputs1(self.drop_out(out1)).squeeze(-1)
        end_logits = self.qa_outputs2(self.drop_out(out2)).squeeze(-1)
        return start_logits, end_logits
class CELoss(Module):
    def __init__(self, loss_fn = nn.CrossEntropyLoss()): 
        self.loss_fn = loss_fn
        
    def forward(self, inputs, start_targets, end_targets):
        start_logits, end_logits = inputs # assumes tuple input
        
        logits = torch.cat([start_logits, end_logits]).contiguous()
        
        targets = torch.cat([start_targets, end_targets]).contiguous()
        
        return self.loss_fn(logits, targets)
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
# Note that validation ds is by default not shuffled in fastai - so indexing like this will work for Callback
# https://forums.fast.ai/t/how-to-set-shuffle-false-of-train-and-val/33730

class JaccardScore(Callback):
    "Stores predictions and targets to perform calculations on epoch end."
    def __init__(self, valid_ds): 
        self.valid_ds = valid_ds
        self.context_text = valid_ds.df.text
        self.answer_text = valid_ds.df.selected_text
        
    def on_epoch_begin(self, **kwargs):
        self.jaccard_scores = []  
        self.valid_ds_idx = 0
        
        
    def on_batch_end(self, last_input:Tensor, last_output:Tensor, last_target:Tensor, **kwargs):
              
        input_ids = last_input[0]
        attention_masks = last_input[1].bool()
        token_type_ids = last_input[2].bool()
        offsets = last_input[3]

        start_logits, end_logits = last_output
        
        # for id in batch of ids
        for i in range(len(input_ids)):
            
            _offsets = offsets[i]
            start_idx, end_idx = torch.argmax(start_logits[i]), torch.argmax(end_logits[i])
            _answer_text = self.answer_text[self.valid_ds_idx]
            original_start, original_end = _offsets[start_idx][0], _offsets[end_idx][1]
            pred_span = self.context_text[self.valid_ds_idx][original_start : original_end]
                
            score = jaccard(pred_span, _answer_text)
            self.jaccard_scores.append(score)

            self.valid_ds_idx += 1
            
    def on_epoch_end(self, last_metrics, **kwargs):        
        res = np.mean(self.jaccard_scores)
        return add_metrics(last_metrics, res)
model = SpanModel(pt_model)
tr_df, val_df = train_test_split(train_df, test_size = 0.2, random_state = 42)
tr_df, val_df  = [df.reset_index(drop=True) for df in [tr_df, val_df]]
train_ds, valid_ds = [TweetDataset(i) for i in [tr_df, val_df]]

test_ds = TweetDataset(test_df, test = True)
loss_fn = partial(CELoss, LabelSmoothingCrossEntropy())

data = DataBunch.create(train_ds, valid_ds, test_ds, path=".", bs = bs)
learner = Learner(data, model, loss_func = loss_fn(), path = electra_dir/'electra-conv', model_dir=f".")
preds = []
test_df_idx = 0

with torch.no_grad():
    
    for xb,yb in tqdm(learner.data.test_dl):
        model0 = learner.load(f'electra_conv_0').model.eval()
        start_logits0, end_logits0 = to_cpu(model0(*xb))
        start_logits0, end_logits0 = start_logits0.float(), end_logits0.float()
        
        model1 = learner.load(f'electra_conv_1').model.eval()
        start_logits1, end_logits1 = to_cpu(model1(*xb))
        start_logits1, end_logits1 = start_logits1.float(), end_logits1.float()
        
        model2 = learner.load(f'electra_conv_2').model.eval()
        start_logits2, end_logits2 = to_cpu(model2(*xb))
        start_logits2, end_logits2 = start_logits2.float(), end_logits2.float()
        
        model3 = learner.load(f'electra_conv_3').model.eval()
        start_logits3, end_logits3 = to_cpu(model3(*xb))
        start_logits3, end_logits3 = start_logits3.float(), end_logits3.float()
        
        model4 = learner.load(f'electra_conv_4').model.eval()
        start_logits4, end_logits4 = to_cpu(model4(*xb))
        start_logits4, end_logits4 = start_logits4.float(), end_logits4.float()
        
        input_ids = to_cpu(xb[0])
        attention_masks = to_cpu(xb[1].bool())
        token_type_ids = to_cpu(xb[2].bool())
        offsets = to_cpu(xb[3])
        
        start_logits = (start_logits0 + start_logits1 + start_logits2 + start_logits3 + start_logits4) / 5
        end_logits = (end_logits0 + end_logits1 + end_logits2 + end_logits3 + end_logits4) / 5
        
        for i in range(len(input_ids)):
            
            _offsets = offsets[i]
#             start_idx, end_idx = torch.argmax(start_logits[i]), torch.argmax(end_logits[i])

            start_idx, end_idx = get_best_start_end_idxs(start_logits[i], end_logits[i])
            original_start, original_end = _offsets[start_idx][0], _offsets[end_idx][1]
            pred_span = test_ds.df.text[test_df_idx][original_start : original_end]
            preds.append(pred_span)
            test_df_idx += 1
test_df['selected_text'] = preds
test_df['selected_text'] = test_df.apply(lambda o: o['text'] if len(o['text']) < 3 else o['selected_text'], 1)
subdf = test_df[['textID', 'selected_text']]
subdf.to_csv("submission.csv", index=False)