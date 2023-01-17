import numpy as np

import pandas as pd

import os

import re

import string

import tokenizers

import transformers



import torch

import torch.nn as nn



from sklearn import model_selection

from torch.nn import functional as F

from tqdm import tqdm

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup
MAX_LEN = 192

TEST_BATCH_SIZE = 32

EPOCHS = 5

ROBERTA_PATH = "../input/roberta-base"

TOKENIZER = tokenizers.ByteLevelBPETokenizer(

    vocab_file=f"{ROBERTA_PATH}/vocab.json", 

    merges_file=f"{ROBERTA_PATH}/merges.txt", 

    lowercase=True,

    add_prefix_space=True

)

DEVICE = 'cuda'

TEST_PATH = '../input/tweet-sentiment-extraction/test.csv'

FOLDS = 5
test_df = pd.read_csv(TEST_PATH)

test_df
class AverageMeter:

    """

    Computes and stores the average and current value

    """

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

    def __init__(self, patience=7, mode="max", delta=0.001):

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





def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
class TweetModel(transformers.BertPreTrainedModel):

    def __init__(self, conf):

        super(TweetModel, self).__init__(conf)

        # Let's take pretrained weights for our model

        config = transformers.RobertaConfig.from_pretrained(

            '../input/roberta-base/config.json', output_hidden_states=True)    

        self.roberta = transformers.RobertaModel.from_pretrained(

            '../input/roberta-base/pytorch_model.bin', config=config)

        self.drop_out = nn.Dropout(0.3)

        # The final layer will have two output features for start and end indexes.

        self.fc = nn.Linear(768, 2)

#         nn.init.normal_(self.l0.weight, std = 0.02)

        nn.init.normal_(self.fc.weight, std=0.02)

        nn.init.normal_(self.fc.bias, 0)

        

    def forward(self, ids, mask):

        _, _, out = self.roberta(

            ids,

            attention_mask = mask

        )

        out = torch.stack([out[-1], out[-2], out[-3], out[-4]])

        out = torch.mean(out, 0)

        out = self.drop_out(out)

        logits = self.fc(out)

        start_logits, end_logits = logits.split(1, dim = -1)

        

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

        

        return start_logits, end_logits
def process_data(tweet, sentiment, tokenizer, max_len):

    '''

    tweet - Tweet from which we have to perform sentiment extraction.

    selected_text - Expected output of sentiment extraction

    sentiment - sentiment to extract (positive, negative or neutral)

    tokenizer - tokenizer to be used for creating tokens

    max_len - max length of tweet.

    '''

    tweet = " " + " ".join(str(tweet).split())

#     selected_text = " " + " ".join(str(selected_text).split())

    

#     len_st = len(selected_text) - 1

#     idx0 = None

#     idx1 = None

    

#     for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):

#         if " " + tweet[ind: ind + len_st] == selected_text:

#             idx0 = ind

#             idx1 = ind + len_st - 1

#             break

            

#     char_targets = [0] * len(tweet)

#     if idx0 != None and idx1 != None:

#         char_targets[idx0: idx1 + 1] = [1]*(idx1 + 1 - idx0)



    tok_tweets = tokenizer.encode(tweet)

    input_ids_orig = tok_tweets.ids

    tweet_offsets = tok_tweets.offsets

    

#     target_idx = []

    

#     for j, (offset1, offset2) in enumerate(tweet_offsets):

#         if sum(char_targets[offset1: offset2]) > 0:

#             target_idx.append(j)

            

#     targets_start = target_idx[0]

#     targets_end = target_idx[-1]



    sentiment_id = {

        'positive': 1313,

        'negative': 2430,

        'neutral': 7974

    }

    

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]

    token_type_ids = [0, 0, 0, 0] + [1] * (len(input_ids_orig) + 1)

    mask = [1] * len(token_type_ids)

    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0,0)]

#     targets_start += 4

#     targets_end += 4

    

    padding_length = max_len - len(input_ids)

    if padding_length > 0:

        input_ids = input_ids + ([1] * padding_length)

        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        mask = mask + [0] * padding_length

        token_type_ids += [0] * padding_length

    

    return {

        'ids': input_ids,

        'mask': mask,

        'token_type_ids': token_type_ids,

        'orig_tweet': tweet,

        'sentiment': sentiment,

        'offsets': tweet_offsets

    }
class TweetDataset:

    def __init__(self, tweet, sentiment):

        self.tweet = tweet

        self.sentiment = sentiment

        self.tokenizer = TOKENIZER

        self.max_len = MAX_LEN

        

    def __len__(self):

        return len(self.tweet)

    

    def __getitem__(self, item):

        data = process_data(

            self.tweet[item],

            self.sentiment[item],

            self.tokenizer,

            self.max_len

        )

        

        return {

            'ids': torch.tensor(data['ids'], dtype = torch.long),

            'mask': torch.tensor(data['mask'], dtype = torch.long),

            'token_type_ids': torch.tensor(data['token_type_ids'], dtype = torch.long),

            'orig_tweet': data['orig_tweet'],

            'sentiment': data['sentiment'],

            'offsets': torch.tensor(data['offsets'], dtype = torch.long)

        }
def calculate_jaccard_score(

    original_tweet, 

    sentiment_val, 

    idx_start, 

    idx_end, 

    offsets,

    verbose=False):

    

    if idx_end < idx_start:

        idx_end = idx_start

    

    filtered_output  = ""

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:

        filtered_output = original_tweet

        

    else:

        for ix in range(idx_start, idx_end + 1):

            filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]

            if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:

                filtered_output += " "

    return filtered_output
model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)

model_config.output_hidden_states = True



model1 = TweetModel(conf=model_config)

model1.to(DEVICE)

model1.load_state_dict(torch.load("../input/roberta-sentiment-extraction/model_0.bin"))

model1.eval()



model2 = TweetModel(conf=model_config)

model2.to(DEVICE)

model2.load_state_dict(torch.load("../input/roberta-sentiment-extraction/model_1.bin"))

model2.eval()



model3 = TweetModel(conf=model_config)

model3.to(DEVICE)

model3.load_state_dict(torch.load("../input/roberta-sentiment-extraction/model_2.bin"))

model3.eval()



model4 = TweetModel(conf=model_config)

model4.to(DEVICE)

model4.load_state_dict(torch.load("../input/roberta-sentiment-extraction/model_3.bin"))

model4.eval()



model5 = TweetModel(conf=model_config)

model5.to(DEVICE)

model5.load_state_dict(torch.load("../input/roberta-sentiment-extraction/model_4.bin"))

model5.eval()
def eval_fn(data_loader, device):

    final_outputs = []

    

    with torch.no_grad():

        tk0 = tqdm(data_loader, total = len(data_loader))

        for bi, d in enumerate(tk0):

            ids = d["ids"]

            mask = d["mask"]

            token_type_ids = d["token_type_ids"]

            orig_tweet = d["orig_tweet"]

            sentiment = d["sentiment"]

            offsets = d["offsets"]



            ids = ids.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            

            predicted_start1, predicted_end1 = model1(ids, mask)

            predicted_start2, predicted_end2 = model2(ids, mask)

            predicted_start3, predicted_end3 = model3(ids, mask)

            predicted_start4, predicted_end4 = model4(ids, mask)

            predicted_start5, predicted_end5 = model5(ids, mask)

            

            predicted_start = (predicted_start1 + predicted_start2 + predicted_start3 + predicted_start4 + predicted_start5) / 5

            predicted_end = (predicted_end1 + predicted_end2 + predicted_end3 + predicted_end4 + predicted_end5) / 5

            

            predicted_start = torch.softmax(predicted_start, dim = 1).cpu().detach().numpy()

            predicted_end = torch.softmax(predicted_end, dim = 1).cpu().detach().numpy()



            for i, tweet in enumerate(orig_tweet):

                tweet_sentiment = sentiment[i]

                output_sentence = calculate_jaccard_score(

                                    original_tweet = tweet,

                                    sentiment_val = tweet_sentiment,

                                    idx_start = np.argmax(predicted_start[i, :]),

                                    idx_end = np.argmax(predicted_end[i, :]),

                                    offsets = offsets[i]

                                    )

                final_outputs.append(output_sentence)

    return final_outputs
def predict():

    df_test = test_df

    test_dataset = TweetDataset(

                    tweet = df_test.text.values,

                    sentiment = df_test.sentiment.values

                    )

    test_dataloader = torch.utils.data.DataLoader(

                            test_dataset,

                            batch_size = TEST_BATCH_SIZE,

                            num_workers = 0

                        )

    

    outputs = eval_fn(test_dataloader, DEVICE)

    

    sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

    sample.loc[:, 'selected_text'] = outputs

    sample.to_csv("submission.csv", index=False)
predict()