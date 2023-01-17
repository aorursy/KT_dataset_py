import os

import torch

import pandas as pd

import torch.nn as nn

import numpy as np

import torch.nn.functional as F

from torch.optim import lr_scheduler

from sklearn import model_selection

from sklearn import metrics

import transformers

import tokenizers

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

from tqdm.autonotebook import tqdm

import utils

from joblib import Parallel, delayed

import warnings

warnings.filterwarnings("ignore")
class config:

    LEARNING_RATE = 4e-5

    MAX_LEN = 192

    TRAINING_FILE = "../input/tweet-train-folds-v2/train_8folds.csv"

    ROBERTA_PATH = "../input/roberta-base"

    TOKENIZER = tokenizers.ByteLevelBPETokenizer(

        vocab_file=f"{ROBERTA_PATH}/vocab.json", 

        merges_file=f"{ROBERTA_PATH}/merges.txt", 

        lowercase=True,

        add_prefix_space=True

    )
# for sentiment in ["negative", "positive", "neutral"]:

#     print(config.TOKENIZER.encode(sentiment).ids)
# from transformers import *

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# tokenizer.encode_plus('hello', 'world')['input_ids']
def process_data(tweet, selected_text, sentiment, tokenizer, max_len):

    # We add a space in front for the Roberata tokenizer. Same thing fot the selected text. 

    # As turns out, doing this processing step could be improved. Check the many top solutions 

    # for better approaches.

    tweet = " " + " ".join(str(tweet).split()) # each word is spearated with space through join method

    selected_text = " " + " ".join(str(selected_text).split())



    len_st = len(selected_text) - 1

    idx0 = None # start of the selected text

    idx1 = None # ending of the selected text



    # Find the start and end indices of the span

    # assert 1 in the tweet whether selected_text remains

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]): # 0th position holds space

        if " " + tweet[ind: ind+len_st] == selected_text:

            idx0 = ind

            idx1 = ind + len_st - 1

            break



    # Assign a positive label for the characters within the selected span 

    # (based on the start and end indices)

    char_targets = [0] * len(tweet)

    if idx0 != None and idx1 != None:

        for ct in range(idx0, idx1 + 1):

            char_targets[ct] = 1

    

    # Tokenize the tweet text and get ids and offsets

    # One detail here: we need to use the tokenizer from the tokenizers

    # library since the one from transformers doesn't provide offsets

    # (or maybe I am wrong, please correct me in the comments if that is the case).

    tok_tweet = tokenizer.encode(tweet)

    

    # an instance of id and offset : 

    # 0 [0,3]: Four

    # here first 0 is the id and two braced numbers are the offsets

    # for more : http://morphadorner.northwestern.edu/morphadorner/techtalk/sentenceandtokenoffsets/

    input_ids_orig = tok_tweet.ids

    tweet_offsets = tok_tweet.offsets

    

    

    # the tokenized word is appended when it has at least one character

    # The indices of the "positive" tokens are stored in `target_idx`.

    target_idx = []

    for j, (offset1, offset2) in enumerate(tweet_offsets):

        if sum(char_targets[offset1: offset2]) > 0:

            target_idx.append(j)

    

    # Ommit the first and last tokens, which should be the [CLS] and [SEP] tokens

    targets_start = target_idx[0]

    targets_end = target_idx[-1]



    # id's are stored in roberta's pretrained token, which is shown at the beginning of this notebook

    sentiment_id = {

        'positive': 1313,

        'negative': 2430,

        'neutral': 7974

    }

    

    # Configuration of tokenizer has given earlier, check it out.

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]

    

    

    '''

    RoBERTa doesn’t have token_type_ids, you don’t need to indicate which token 

    belongs to which segment.

    

    before input_ids_orig, we can see 4 individual segment, as there is no token_type_ids

    in roberta, 4 positions are added with 0 value.

    '''

    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)

    mask = [1] * len(token_type_ids)

    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

    targets_start += 4

    targets_end += 4



    # How much to pad the text to have the same sequence lengths. 

    padding_length = max_len - len(input_ids)

    if padding_length > 0:

        input_ids = input_ids + ([1] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    

    # Return processed tweet as a dictionary

    return {

        'ids': input_ids,

        'mask': mask,

        'token_type_ids': token_type_ids,

        'targets_start': targets_start,

        'targets_end': targets_end,

        'orig_tweet': tweet,

        'orig_selected': selected_text,

        'sentiment': sentiment,

        'offsets': tweet_offsets

    }
class TweetDataset:

    def __init__(self, tweet, sentiment, selected_text):

        self.tweet = tweet

        self.sentiment = sentiment

        self.selected_text = selected_text

        

        # from another class

        self.tokenizer = config.TOKENIZER

        self.max_len = config.MAX_LEN

    

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



        # Return the processed data where the lists are converted to `torch.tensor`s

        return {

            'ids': torch.tensor(data["ids"], dtype=torch.long),

            'mask': torch.tensor(data["mask"], dtype=torch.long),

            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),

            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),

            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),

            'orig_tweet': data["orig_tweet"],

            'orig_selected': data["orig_selected"],

            'sentiment': data["sentiment"],

            'offsets': torch.tensor(data["offsets"], dtype=torch.long)

        }
class TweetModel(transformers.BertPreTrainedModel):

    def __init__(self, conf):

        super(TweetModel, self).__init__(conf)

        # Load the pretrained BERT model

        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)

        # Set 10% dropout to be applied to the BERT backbone's output

        self.drop_out = nn.Dropout(0.1)

        

        '''

        768 is the dimensionality of roberta_base's hidden representations

        Multiplied by 2 since the forward pass concatenates the last two hidden representation layers

        The output will have two dimensions ("start_logits", and "end_logits")

        '''

        self.l0 = nn.Linear(768 * 2, 2)

        torch.nn.init.normal_(self.l0.weight, std=0.02)

    

    

    

    # Return the hidden states from the BERT backbone

    def forward(self, ids, mask, token_type_ids):

        _, _, out = self.roberta(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        ) # bert_layers x bs x SL x (768)

        

        

        '''

        Concatenate the last two hidden states

        This is done since experiments have shown that just getting the last layer

        gives out vectors that may be too taylored to the original BERT training objectives (MLM + NSP)

        Sample explanation: https://bert-as-service.readthedocs.io/en/latest/section/faq.html

        why-not-the-last-hidden-layer-why-second-to-last

        '''

        out = torch.cat((out[-1], out[-2]), dim=-1) # bs x SL x (768 * 2)

        # Apply 10% dropout to the last 2 hidden states

        out = self.drop_out(out) # bs x SL x (768 * 2)

        # The "dropped out" hidden vectors are now fed into the linear layer to output two scores

        logits = self.l0(out) # bs x SL x 2



        # Splits the tensor into start_logits and end_logits

        # (bs x SL x 2) -> (bs x SL x 1), (bs x SL x 1)

        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1) # (bs x SL)

        end_logits = end_logits.squeeze(-1) # (bs x SL)



        return start_logits, end_logits
def calculate_jaccard_score(

    original_tweet, 

    target_string, 

    sentiment_val, 

    idx_start, 

    idx_end, 

    offsets,

    verbose=False):

    

    # A span's end index has to be greater than or equal to the start index

    # If this doesn't hold, the start index is set to equal the end index (the span is a single token)

    if idx_end < idx_start:

        idx_end = idx_start

    

    # Combine into a string the tokens that belong to the predicted span

    filtered_output  = ""

    

    

    '''

    If the token is not the last token in the tweet, and the ending offset of the current token is less

    than the beginning offset of the following token, add a space.

    Basically, add a space when the next token (word piece) corresponds to a new word

    '''

    for ix in range(idx_start, idx_end + 1):

        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]

        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:

            filtered_output += " "



    # Set the predicted output as the original tweet when the tweet's sentiment is 

    # "neutral", or the tweet only contains one word

    if len(original_tweet.split()) < 2:

        filtered_output = original_tweet



        

    # Calculate the jaccard score between the predicted span, and the actual span

    # The IOU (intersection over union) approach is detailed in the utils module's `jaccard` function:

    # https://www.kaggle.com/abhishek/utils

    jac = utils.jaccard(target_string.strip(), filtered_output.strip())

    return jac, filtered_output
df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

df_test.loc[:, "selected_text"] = df_test.text.values
ROBERTA_PATH = "../input/roberta-base"

model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)

model_config.output_hidden_states = True

device = torch.device("cuda")
%%time



model1 = TweetModel(conf=model_config)

model1.to(device)

model1.load_state_dict(torch.load("../input/train-tweet-fold/model_0.bin"))

model1.eval()



model2 = TweetModel(conf=model_config)

model2.to(device)

model2.load_state_dict(torch.load("../input/train-tweet-fold/model_1.bin"))

model2.eval()



model3 = TweetModel(conf=model_config)

model3.to(device)

model3.load_state_dict(torch.load("../input/train-tweet-fold/model_2.bin"))

model3.eval()



model4 = TweetModel(conf=model_config)

model4.to(device)

model4.load_state_dict(torch.load("../input/train-tweet-fold/model_3.bin"))

model4.eval()



model5 = TweetModel(conf=model_config)

model5.to(device)

model5.load_state_dict(torch.load("../input/train-tweet-fold/model_4.bin"))

model5.eval()
final_output = []



# Instantiate TweetDataset with the test data

TEST_BATCH_SIZE = 32



test_dataset = TweetDataset(

        tweet=df_test.text.values,

        sentiment=df_test.sentiment.values,

        selected_text=df_test.selected_text.values

    )



# Instantiate DataLoader with `test_dataset`

data_loader = torch.utils.data.DataLoader(

    test_dataset,

    shuffle=False,

    batch_size=TEST_BATCH_SIZE,

    num_workers=1

)



# Turn off gradient calculations

with torch.no_grad():

    tk0 = tqdm(data_loader, total=len(data_loader))

    

    # Predict the span containing the sentiment for each batch

    for bi, d in enumerate(tk0):

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]

        sentiment = d["sentiment"]

        orig_selected = d["orig_selected"]

        orig_tweet = d["orig_tweet"]

        targets_start = d["targets_start"]

        targets_end = d["targets_end"]

        offsets = d["offsets"].numpy()



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)

        targets_end = targets_end.to(device, dtype=torch.long)



        # Predict start and end logits for each of the models

        outputs_start1, outputs_end1 = model1(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start2, outputs_end2 = model2(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start3, outputs_end3 = model3(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start4, outputs_end4 = model4(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        outputs_start5, outputs_end5 = model5(

            ids=ids,

            mask=mask,

            token_type_ids=token_type_ids

        )

        

        # Get the average start and end logits across the five models and use these as predictions

        # This is a form of "ensembling"

        outputs_start = (outputs_start1 + outputs_start2 + outputs_start3 + outputs_start4 + 

                         outputs_start5 ) / 5

        outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 +

                       outputs_end5) / 5

        

        

        # Apply softmax to the predicted start and end logits

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()

        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        jaccard_scores = []

        

        # Convert the start and end scores to actual predicted spans (in string form)

        for px, tweet in enumerate(orig_tweet):

            selected_tweet = orig_selected[px]

            tweet_sentiment = sentiment[px]

            _, output_sentence = calculate_jaccard_score(

                original_tweet=tweet,

                target_string=selected_tweet,

                sentiment_val=tweet_sentiment,

                idx_start=np.argmax(outputs_start[px, :]),

                idx_end=np.argmax(outputs_end[px, :]),

                offsets=offsets[px]

            )

            final_output.append(output_sentence)
sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

sample.loc[:, 'selected_text'] = final_output

sample.to_csv("submission.csv", index=False)

sample.head()