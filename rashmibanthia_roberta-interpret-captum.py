#Turn on the internet and GPU

!pip install captum > /dev/null
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

#from tqdm.autonotebook import tqdm

from tqdm import tqdm

#import utils



from tokenizers import ByteLevelBPETokenizer #BertWordPieceTokenizer



from captum.attr import visualization as viz

from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients

from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer



def set_seeds(SEED=42):

  os.environ["PYTHONHASHSEED"] = str(SEED)

  np.random.seed(SEED)

  torch.manual_seed(SEED)

  torch.cuda.manual_seed(SEED)

  torch.backends.cudnn.deterministic = True

  torch.backends.cudnn.benchmark = False 



SEED=10

set_seeds(SEED)



#torch.cuda.get_device_name(0)


MODEL_PATH = "../input/roberta-models-38"

MODEL_PATH_0 = f'{MODEL_PATH}/model_0.bin'





MAX_LEN = 128

VALID_BATCH_SIZE = 1 #Do not change 





train_path = '../input/tweet-sentiment-extraction/train.csv'

test_path = '../input/tweet-sentiment-extraction/test.csv'

sample_submission_path = '../input/tweet-sentiment-extraction/sample_submission.csv'

ROBERTA_PATH = '../input/roberta-base/'

TOKENIZER = ByteLevelBPETokenizer(

              vocab_file=f"{ROBERTA_PATH}/vocab.json", 

              merges_file=f"{ROBERTA_PATH}/merges.txt", 

              lowercase=True,

              add_prefix_space=True )

ROBERTA_CONFIG_JSON = f'{ROBERTA_PATH}/config.json' 

TRAINING_FILE = '../input/tweet-sentimentextraction-8folds/R8_train_folds.csv'



class TweetModel(transformers.BertPreTrainedModel):

    def __init__(self, conf):

        super(TweetModel, self).__init__(conf)

        self.roberta = transformers.RobertaModel.from_pretrained(f'{ROBERTA_PATH}', config=conf)

        self.drop_out = nn.Dropout(0.1)

        self.l0 = nn.Linear(768 * 2, 2)

        torch.nn.init.normal_(self.l0.weight, std=0.02)

    

    def forward(self, ids, mask, token_type_ids):

        _, _, out = self.roberta(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )



        out = torch.cat((out[-1], out[-2]), dim=-1)

        out = self.drop_out(out)

        logits = self.l0(out)



        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits
def process_data(tweet, selected_text, sentiment, tokenizer, max_len):

    tweet = " " + " ".join(str(tweet).split())

    selected_text = " " + " ".join(str(selected_text).split())



    len_st = len(selected_text) - 1

    idx0 = None

    idx1 = None



    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):

        if " " + tweet[ind: ind+len_st] == selected_text:

            idx0 = ind

            idx1 = ind + len_st - 1

            break



    char_targets = [0] * len(tweet)

    if idx0 != None and idx1 != None:

        for ct in range(idx0, idx1 + 1):

            char_targets[ct] = 1

    

    tok_tweet = tokenizer.encode(tweet)

    input_ids_orig = tok_tweet.ids

    tweet_offsets = tok_tweet.offsets

    

    target_idx = []

    for j, (offset1, offset2) in enumerate(tweet_offsets):

        if sum(char_targets[offset1: offset2]) > 0:

            target_idx.append(j)

    

    targets_start = target_idx[0]

    targets_end = target_idx[-1]



    sentiment_id = {

        'positive': 1313,

        'negative': 2430,

        'neutral': 7974

    }

    

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]

    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)

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

            

        }


def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def calculate_jaccard_score(

    original_tweet, 

    target_string, 

    sentiment_val, 

    idx_start, 

    idx_end, 

    offsets,

    verbose=False):

    

    if idx_end < idx_start:

        idx_end = idx_start

    

    filtered_output  = ""

    for ix in range(idx_start, idx_end + 1):

        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]

        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:

            filtered_output += " "



    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:

        filtered_output = original_tweet



    jac = jaccard(target_string.strip(), filtered_output.strip())

    return jac, filtered_output

def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):

    pred = predict(inputs,

                   token_type_ids=token_type_ids,mask=attention_mask)

                   #position_ids=position_ids,

                   

    pred = pred[position]

    return pred.max(1).values





#A helper function to perform forward pass of the model and make predictions.

def predict(inputs, token_type_ids=None, position_ids=None, mask=None):

#     return model1(inputs, token_type_ids=token_type_ids,

#                  #position_ids=position_ids, 

#                  mask=mask, )

     return model1(inputs, token_type_ids=token_type_ids,mask=mask)

    

def summarize_attributions(attributions):

            attributions = attributions.sum(dim=-1).squeeze(0)

            attributions = attributions / torch.norm(attributions)

            return attributions
fold = 0

final_output = []

set_seeds(SEED)

dfx = pd.read_csv(TRAINING_FILE)

df_train = dfx[dfx.kfold != fold].reset_index(drop=True)

df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)



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





device = torch.device("cuda")

model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_CONFIG_JSON)

model_config.output_hidden_states = True



model1 = TweetModel(conf=model_config)

model1.to(device)

model1.load_state_dict(torch.load(MODEL_PATH_0))

model1.eval()

data = []

incorrect = 0 

with torch.no_grad():

    for bi, d in enumerate(valid_data_loader):

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

         

        start_scores, end_scores = predict(ids, \

                                   token_type_ids=token_type_ids, \

                                   position_ids=None, \

                                  mask=mask)

        

        outputs_start = torch.softmax(start_scores, dim=1).cpu().detach().numpy()

        outputs_end = torch.softmax(end_scores, dim=1).cpu().detach().numpy()

        

        #What are Integrated Gradients - https://captum.ai/docs/extension/integrated_gradients

        lig = LayerIntegratedGradients(squad_pos_forward_func, model1.roberta.embeddings)

       

        attributions_start, delta_start = lig.attribute(inputs=ids,

                                baselines=None, 

                                additional_forward_args=(token_type_ids, None, mask),

                                return_convergence_delta=True)

        

        attributions_end, delta_end = lig.attribute(inputs=ids,  

                              additional_forward_args=(token_type_ids, None, mask, 1),

                              return_convergence_delta=True)

      

        attributions_start_sum = summarize_attributions(attributions_start[0])

        attributions_end_sum = summarize_attributions(attributions_end[0])

        ground_truth_start_ind = targets_start.cpu().detach().numpy()



        # storing couple samples in an array for visualization purposes

        tokenizer = TOKENIZER        

        

        #todo: remove Ġ 

        all_tokens = ['<s>'] + tokenizer.encode(sentiment[0]).tokens  + ['</s>'] + ['</s>'] + tokenizer.encode(orig_tweet[0]).tokens + ['<s>']

        

        start_position_vis = viz.VisualizationDataRecord(

                          attributions_start_sum,  #word_attributions

                          torch.max(torch.softmax(start_scores[0], dim=0)), #pred_prob

                          torch.argmax(start_scores[0]), #pred_class

                          torch.argmax(start_scores[0]), #true_class

                          str(ground_truth_start_ind), #attr_class

                          attributions_start_sum.sum(),     #attr_score

                          all_tokens, #raw input

                          delta_start) #convergence score

        

        ground_truth_end_ind = targets_end.cpu().detach().numpy()

        end_position_vis = viz.VisualizationDataRecord(

                        attributions_end_sum,

                        torch.max(torch.softmax(end_scores[0], dim=0)),

                        torch.argmax(end_scores[0]),

                        torch.argmax(end_scores[0]),

                        str(ground_truth_end_ind),

                        attributions_end_sum.sum(),       

                        all_tokens,

                        delta_end)

        

        jscore, output_sentence = calculate_jaccard_score(

              original_tweet=orig_tweet[0],

              target_string=orig_selected[0],

              sentiment_val=sentiment[0],

              idx_start=np.argmax(outputs_start[0, :]),  

              idx_end=np.argmax(outputs_end[0, :]),   

              offsets=offsets[0]

          )

        

        print('\033[1m', 'Visualizations For Start Position', '\033[0m')

        viz.visualize_text([start_position_vis])

        print('\033[1m', 'Visualizations For End Position', '\033[0m')

        viz.visualize_text([end_position_vis])        

        

        print("True selected:",orig_selected[0], "\t\t Predicted:", output_sentence)

        #print("\n\n")

        #Todo: fix too much spacing ?? 

        

        if bi > 10: #Visualize just 10 samples 

            break



        #Todo: Continue with Captum tutorial 