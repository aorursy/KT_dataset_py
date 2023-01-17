use_own_text = True #for testing set

use_train_data = False #use train data for testing
import os

DATASET_PATH = '../input/bert-model-weights/'

os.listdir(DATASET_PATH)
BERT_MODEL_PATHS = [

 'bert_bert_sb_0.93405_out_1_cpa.pt',

 'bert_wss_boosted2_cpa_nlabels31_0.93774.pt', #LB 0.935 

 'bert_wss_aux_ot_rt_0.93803.pt', #LB 0.936

 'bert_sb1t9_no_identity_ot_rt_cont_rand_sb1t9_0.93574_out_13_cpa.pt',

 'bert_base_cont_boosted3_sge6_0.93672_out_12_cpa.pt',

 'bert_base_last_year_competition_cont_pp_no_identity_cont__0.94643_out_1_cpa.pt',  #LB 0.93525

 'bert_sb_mini_epoch_0.93451_out_8_cpa.pt',

 'bert_wss_aux_ot_0.93722.pt',

 'bert_public_0.93373.pt',

 'bert_subgroup_balanced_20p_target_ist_bert_ga2_0.93370.pt', 

 'bert_yuval_baseline_bert_model_cont_boost_0.93352.pt',

 'bert_base_0.93327_out_1_cpa.pt',

]



BERT_MODEL_PATHS = [DATASET_PATH+bmp for bmp in BERT_MODEL_PATHS]
DATASET_PATH2 = '../input/bertlstm-weights/'

os.listdir(DATASET_PATH2)
BERT_MODEL_PATHS.append(DATASET_PATH2+'bert_beLSTM_sb2t8_0.91725.pt')

BERT_MODEL_PATHS.append(DATASET_PATH2+'bert_beLSTM_aux_wss_bad_minor_0.91675.pt')

BERT_MODEL_PATHS.append(DATASET_PATH2+'bert_beLSTM_aux_wss_0.92162.pt')
import sys

#package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

package_dir = '../input/pytorch-pretrained-bert-master/pytorch-pretrained-bert-master/pytorch-pretrained-BERT-master/'

#custom_weight_dir = '../input/bert model weights/'

sys.path.append(package_dir)
args = {

    'device' : 'cuda',

    'learning_rate' : 25e-6,

    'warmup_proportion': 0.2, #0.1,

    'batch_size' : 32,

    'ebbs_epoch_samples' : 25e3,

    'num_eebs_epochs' : 15e6//100e3,

    'ebbs_test_multipl' : 1, # ebbs_epoch_samples*(mulitpl+1) = samples that get evaluated/tested 

    'eval_sample_size' : 10e3,

    'num_labels' : 1,

    'max_seq_length' : 250,

    'gradient_accumulation_steps' : 2,

    'fp16' : False, #apex #https://github.com/NVIDIA/apex/issues/131

    'seed' : 70135

}
label_clms = ['target','severe_toxicity', 'obscene','identity_attack', 

              'insult', 'threat','funny', 'wow','sad','sexual_explicit','rating']



identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

                    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']



#target_columns = ['severe_toxicity', 'obscene','identity_attack', 'insult', 'threat']

#offtarget_columns = ['funny', 'wow','sad', 'likes', 'disagree', 'sexual_explicit']



target_clms = [*label_clms,*identity_columns]
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import torch.utils.data

import numpy as np

import pandas as pd

from tqdm import tqdm

import os

import warnings

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertModel

from pytorch_pretrained_bert import BertConfig



warnings.filterwarnings(action='once')

device = torch.device('cuda')
def replace_underscore_with_backspace(txt):

    """ does what function name is ;) """

    return re.sub('_+',' ',txt)



def replace_newline_with_dot(txt):

    """ does what function name is ;) """

    return re.sub('\n+','. ',txt)

def replace_newline_with_space(txt):

    """ does what function name is ;) """

    return re.sub('\n+',' ',txt)



import re

def remove_urls(txt):

    """ Removal of HTTP links/URLs mixed up in any text: """

    return re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", " ", txt)

import string

def remove_multi_symbols(txt,symbols=string.punctuation):

    """ reduces symbols that occure multiple times to one symbol

    by default goes over those symbols: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    Note: call this after calling remove_urls

    """

    for symbol in symbols:

        txt = re.sub('\\'+symbol+'+',symbol,txt)

    return txt
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

if use_train_data:

    test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv").sample(10000)
if use_own_text:

    my_text = [

        'asdf',

        'feck-you',

        'trolling the nation',

        'I am proud to be gay',

        'Nothing to do here',

        'Hey, keep your Islamophobia out of public domains.', #should be 0

        'Well, the money the education system spent on you was wasted', #should be 0.5

        'This clown confuses the word "substance" with "abuse".', #should be 0

        '"Two things are infinite: the universe and human stupidity; and I´m not sure about the universe." - Albert Einstein', #should be 0

        'I have epilepsy',

        'I am a woman with epilepsy',

        'I am a person who uses a wheelchair',

        'I use a wheelchair',

        'I am gay and have to use a wheelchair',

        'I am blind',

        'I am a person who is deaf',

        'I am a woman who is deaf',

        'I am a non-binary-gender person who is deaf',

        'I am a black person who is deaf',

        'I am pround to be gay',

        'I am a proud homo',

        'I am a proud lesbian',

        'I am a proud to be black',

        'KKK',

        'a55hle',

        'a$$hole',

        'I am a very proud lesbian',

        'I am a blind homosexual.',

        "I don't feel ashamed for beeing lesbian",

        "Proud to be a fu**ing gay person",

        "Totally wrong to think that this wouldn't be gay as shit",

         ' ',

         '!',

         '',

         '???',

        'I hate trump',

        'I hate obama',

        'Trump',

        'I like that you are gay',

    ]

    test_df = pd.DataFrame({

        'id': range(len(my_text)),

        'comment_text': my_text

    })
test_df['comment_text_orig'] = test_df['comment_text']
functions = [

    remove_urls,

    replace_underscore_with_backspace,

    replace_newline_with_dot, #replace with dot like in prepro

    lambda k: remove_multi_symbols(k,symbols=['.','-','=']) # backspace should have also been added

    ]

for func in functions:

    print(func.__name__)

    test_df['comment_text'] = test_df['comment_text'].apply(func)
pd.set_option('max_colwidth', 500)

test_df[['comment_text','comment_text_orig']].sample(10)
def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = np.zeros((len(example),max_seq_length+2))

    token_len = np.zeros(len(example))

    for i,text in enumerate(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

        #test_df.loc[i,'tokens']=str(tokens_a)

        tokens = tokenizer.convert_tokens_to_ids(tokens_a)

        tokens = [101, *tokens, 102]

        token_len[i] = len(tokens)

        all_tokens[i,:len(tokens)] = np.asarray(tokens)

    return all_tokens, token_len
SEED = 1234

TOKENIZER_PATH = '../input/berttokenizer/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'



np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir=None,do_lower_case=True)
test_df['comment_text'] = test_df['comment_text'].astype(str)

X_test, token_len = convert_lines(test_df["comment_text"], args['max_seq_length'], tokenizer)
test_df['token_len'] = token_len
#X_test
import torch

from torch import Tensor

import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss



device = torch.device('cuda')



class BertForMultiLabelSequenceClassification(BertForSequenceClassification):

    """BERT model for classification.

    This module is composed of the BERT model with a linear layer on top of

    the pooled output.

    Params:

        `config`: a BertConfig class instance with the configuration to build a new model.

        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:

        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]

            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts

            `extract_features.py`, `run_classifier.py` and `run_squad.py`)

        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token

            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to

            a `sentence B` token (see BERT paper for more details).

        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices

            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max

            input sequence length in the current batch. It's the mask that we typically use for attention when

            a batch has varying length sentences.

        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]

            with indices selected in [0, ..., num_labels].

    Outputs:

        if `labels` is not `None`:

            Outputs the CrossEntropy classification loss of the output with the labels.

        if `labels` is `None`:

            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:

    ```python

    # Already been converted into WordPiece token ids

    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,

        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)

    logits = model(input_ids, token_type_ids, input_mask)

    ```

    """



    def __init__(self, config, num_labels=2):

        super(BertForMultiLabelSequenceClassification,

              self).__init__(config, num_labels)

        self.num_labels = num_labels

        self.bert = BertModel(config)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, return_pooled_output=False, return_last_unpooled_layer=False):

      full_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

      #pooled_output = self.dropout(pooled_output)

      logits = self.classifier(pooled_output)

      

      if labels is not None:

        #BCE: class has to be binary!

        #CE: choose one of n-classes

          loss_fct = BCEWithLogitsLoss() #weight=loss_weight

          #weight=loss_weight.view(1,-1)

          # The size of tensor a (32) must match the size of tensor b (19) at non-singleton dimension 1



          mask = (labels != -1).to(dtype=torch.float)

          logits = logits * mask

          label_tensor = labels * mask



          loss = loss_fct(logits, labels) #output, target

          return loss, logits

      elif return_pooled_output: return logits, pooled_output

      elif return_last_unpooled_layer: return logits, full_output

      return logits



    def forward_old(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, loss_weight=None):

        _, pooled_output = self.bert(

            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)



        if labels is not None:

          #BCE: class has to be binary!

          #CE: choose one of n-classes

            loss_fct = BCEWithLogitsLoss() #weight=loss_weight

            #weight=loss_weight.view(1,-1)

            # The size of tensor a (32) must match the size of tensor b (19) at non-singleton dimension 1



            mask = (labels != -1).to(dtype=torch.float)

            logits = logits * mask

            labels = labels * mask

            

            loss = loss_fct(logits.view(-1, self.num_labels), #output, 

                            labels.view(-1, self.num_labels)) #target

            return loss

        else:

            return logits



    def freeze_bert_encoder(self):

        for param in self.bert.parameters():

            param.requires_grad = False



    def unfreeze_bert_encoder(self):

        for param in self.bert.parameters():

            param.requires_grad = True

            

from torch import nn

class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)    # (N, T, 1, K)

        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)

        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked

        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)

        x = x.squeeze(2)  # (N, T, K)

        return x

      

class BertLSTM1(nn.Module):

    """#TODO"""

    #https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version

    def __init__(self, embedding_matrix, num_labels, n_aux_units = 7, max_features=args['max_seq_length'],LSTM_UNITS=128,DENSE_HIDDEN_UNITS=4*128):

        super(BertLSTM1, self).__init__()

        self.num_labels = num_labels

        embed_size = embedding_matrix.shape[1]



        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        #self.embedding.weight.requires_grad = False #may set to true!!

        self.embedding_dropout = SpatialDropout(0.3)



        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)



        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)



        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)

        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, n_aux_units)





    def forward(self, input_ids, attention_mask=None, labels=None, loss_weight=None, return_pooled_output=False):

        h_embedding = self.embedding(input_ids)

        h_embedding = self.embedding_dropout(h_embedding)



        h_lstm1, _ = self.lstm1(h_embedding)

        h_lstm2, _ = self.lstm2(h_lstm1)



        # global average pooling

        avg_pool = torch.mean(h_lstm2, 1)

        # global max pooling

        max_pool, _ = torch.max(h_lstm2, 1)



        h_conc = torch.cat((max_pool, avg_pool), 1)

        h_conc_linear1  = F.relu(self.linear1(h_conc))

        h_conc_linear2  = F.relu(self.linear2(h_conc))



        hidden = h_conc + h_conc_linear1 + h_conc_linear2



        result = self.linear_out(hidden)

        #aux_result = self.linear_aux_out(hidden)

        #out = torch.cat([result, aux_result], 1)



        return result#out.view(-1, self.num_labels) 

    

class BertLSTM2(nn.Module):

    """#TODO"""

    #https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version

    def __init__(self, embedding_matrix, num_labels, n_aux_units = 7, max_features=args['max_seq_length'],LSTM_UNITS=128,DENSE_HIDDEN_UNITS=4*128):

        super(BertLSTM2, self).__init__()

        self.num_labels = num_labels

        embed_size = embedding_matrix.shape[1]



        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        #self.embedding.weight.requires_grad = False #may set to true!!

        self.embedding_dropout = SpatialDropout(0.3)



        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)



        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)



        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, num_labels)

        #self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, n_aux_units)





    def forward(self, input_ids, attention_mask=None, labels=None, loss_weight=None, return_pooled_output=False):

        h_embedding = self.embedding(input_ids)

        h_embedding = self.embedding_dropout(h_embedding)



        h_lstm1, _ = self.lstm1(h_embedding)

        h_lstm2, _ = self.lstm2(h_lstm1)



        # global average pooling

        avg_pool = torch.mean(h_lstm2, 1)

        # global max pooling

        max_pool, _ = torch.max(h_lstm2, 1)



        h_conc = torch.cat((max_pool, avg_pool), 1)

        h_conc_linear1  = F.relu(self.linear1(h_conc))

        h_conc_linear2  = F.relu(self.linear2(h_conc))



        hidden = h_conc + h_conc_linear1 + h_conc_linear2



        result = self.linear_out(hidden)



        return result#out.view(-1, self.num_labels) 
def load_bert_model(path, num_labels=args['num_labels']):

    state_dict = torch.load(path)

    if path.find('../input/bertlstm-weights/')==0:

        print('loading bertLSTM model')

        if num_labels is None:

            num_labels = int(state_dict[list(state_dict.keys())[-1]].shape[0])

            print('Number of Labels set to',num_labels)

        try:

            model = BertLSTM1(np.zeros((30522, 768)),num_labels)

            print(model.load_state_dict(state_dict))

        except:

            print('trying BertLSTM2')

            model = BertLSTM2(np.zeros((30522, 768)),num_labels)

            print(model.load_state_dict(state_dict))

        model = model.to('cuda')

        del state_dict

        return model

    

    config = BertConfig(30522)

    if num_labels is None:

        if 'classifier.bias' in state_dict.keys():

            num_labels = int(state_dict['classifier.bias'].shape[0])

        else: 

            print('No classifier.bias in state_dict --> assuming num_labels = 1')

            num_labels = 1



    model = BertForMultiLabelSequenceClassification(config,num_labels=num_labels)

    model.load_state_dict(state_dict)

    model.to(device)

    del state_dict

    return model
out_clms = []

for ii, BERT_MODEL_PATH in  enumerate(BERT_MODEL_PATHS):

    print('\nusing model',BERT_MODEL_PATH)

    model = load_bert_model(BERT_MODEL_PATH,num_labels=None)

    for param in model.parameters():

        param.requires_grad = False

    model.eval()

    

    buckets = 30

    old_thld = -1

    for b in range(buckets):

        thld = int(np.percentile(test_df.token_len,(b+1)*100//buckets))

        mask = (test_df.token_len>old_thld) & (test_df.token_len<=thld)

        if mask.sum()>0:

            batch_size = args['batch_size'] if thld<=220 else 16

            old_thld = thld

            print('\rBucketNr',b,'thld:',thld,'samples:',mask.sum(),end='')



            preds = None

            test = torch.utils.data.TensorDataset(torch.tensor(X_test[mask,:thld], dtype=torch.long))

            test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

            for i, (x_batch,) in enumerate(test_loader):

                logits = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

                #pred2 = model2(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)



                y_pred = logits.sigmoid().detach().cpu().numpy()

                if preds is None: preds = y_pred

                else: preds = np.concatenate((preds,y_pred),axis=0)

            test_df.loc[mask,'bo'+str(ii)] = preds[:,0]

    out_clms.append('bo'+str(ii))
weights = np.ones(len(out_clms))



#overweight model nr2

#if len(weights)>=0: weights[0] = 0.4

#if len(weights)>=1: weights[1] = 1.1

#if len(weights)>=2: weights[2] = 1.2

#if len(weights)>=4: weights[4] = 1.0

#if len(weights)>=10: weights[9] = 1.0

#if len(weights)>=11: weights[10] = 0.2

#if len(weights)>=12: weights[11] = 0.2

#if len(weights)>=13: weights[12] = 0.3

    

print([(BERT_MODEL_PATHS[i].split('/')[-1],weights[i]) for i in range(len(weights))])

test_df.fillna(0,inplace=True)

test_df['prediction'] = np.average(test_df[out_clms].values, weights=weights, axis=1)
use_ranked_sum = False # was worse on test data by 0.02

if use_ranked_sum:

    for key in out_clms:

        test_df[key + '_rank'] = test_df[key].rank()

    test_df['rank_sum'] = np.sum(

            test_df[col] for col in test_df.columns if '_rank' in col)

    test_df['rank_sum_prediction'] = test_df['rank_sum']/(len(out_clms) *

            test_df.shape[0])
if use_train_data:

    from sklearn import metrics

    import numpy as np

    import pandas as pd

    # List all identities

    identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

                        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    target_columns = ['severe_toxicity', 'obscene','identity_attack', 'insult', 'threat']

    offtarget_columns = ['funny', 'wow','sad', 'likes', 'disagree', 'sexual_explicit']



    TOXICITY_COLUMN = 'target'

    SUBGROUP_AUC = 'subgroup_auc'

    BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative

    BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive



    # Convert taget and identity columns to booleans

    def convert_to_bool(df, col_name):

        df[col_name] = np.where(df[col_name] >= 0.5, True, False)



    def convert_dataframe_to_bool(df):

        bool_df = df.copy()

        for col in ['target'] + identity_columns: #+target_columns+offtarget_columns

            convert_to_bool(bool_df, col)

        return bool_df



    def compute_auc(y_true, y_pred):

        try:

            return metrics.roc_auc_score(y_true, y_pred)

        except ValueError:

            return np.nan



    def compute_subgroup_auc(df, subgroup, label, model_name):

        subgroup_examples = df[df[subgroup]]

        return compute_auc(subgroup_examples[label], subgroup_examples[model_name])



    def compute_bpsn_auc(df, subgroup, label, model_name):

        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""

        subgroup_negative_examples = df[df[subgroup] & ~df[label]]

        non_subgroup_positive_examples = df[~df[subgroup] & df[label]]

        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

        return compute_auc(examples[label], examples[model_name])



    def compute_bnsp_auc(df, subgroup, label, model_name):

        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""

        subgroup_positive_examples = df[df[subgroup] & df[label]]

        non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]

        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

        return compute_auc(examples[label], examples[model_name])



    def compute_bias_metrics_for_model(dataset,subgroups,model,label_col,include_asegs=False):

        """Computes per-subgroup metrics for all subgroups and one model."""

        dataset = convert_dataframe_to_bool(dataset)

        records = []

        for subgroup in subgroups:

            record = {

                'subgroup': subgroup,

                'subgroup_size': len(dataset[dataset[subgroup]])

            }

            record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)

            record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)

            record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)

            records.append(record)

        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)



    def calculate_overall_auc(df, model_name, TOXICITY_COLUMN):

        true_labels = df[TOXICITY_COLUMN]>=0.5

        predicted_labels = df[model_name]

        return metrics.roc_auc_score(true_labels, predicted_labels)



    def power_mean(series, p):

        total = sum(np.power(series, p))

        return np.power(total / len(series), 1 / p)



    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):

        bias_score = np.average([

            power_mean(bias_df[SUBGROUP_AUC], POWER),

            power_mean(bias_df[BPSN_AUC], POWER),

            power_mean(bias_df[BNSP_AUC], POWER)

        ])

        #print(bias_score)

        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score), bias_score



    def get_scores(in_df, model_name,TOXICITY_COLUMN='target',fillna=True,consolidation_method='mean'):

        # groupby ID to remove dublicates --> merge splitted data

        df = in_df[[model_name,'id',*identity_columns, TOXICITY_COLUMN]].groupby('id')

        if consolidation_method=='mean': df = df.mean()

        if consolidation_method=='max': df = df.max()  

        if consolidation_method=='min': df = df.min() 



        bias_metrics_df = compute_bias_metrics_for_model(df, identity_columns, model_name, TOXICITY_COLUMN)

        if fillna: bias_metrics_df.fillna(0.5,inplace=True)

        try:

          overall_auc = calculate_overall_auc(df, model_name, TOXICITY_COLUMN)

        except ValueError:

          print('Error - only one class present in df')

          overall_auc = np.nan

        final_metric, bias_score = get_final_metric(bias_metrics_df, overall_auc)

        bias_metrics_df['overall_auc']=overall_auc

        bias_metrics_df['final_bias_score'] = bias_score

        bias_metrics_df['final_metric'] = final_metric

        bias_metrics_df.index = bias_metrics_df.subgroup

        return bias_metrics_df
if use_train_data:

    score = get_scores(test_df,'prediction')
if use_train_data:

    for i,clm in enumerate(out_clms):

        print(i, str(get_scores(test_df,clm).final_metric.mean())[:5],weights[i],BERT_MODEL_PATHS[i].split('/')[-1],sep='\t')
for out_clm in ['prediction',*out_clms]:

    submission = pd.DataFrame.from_dict({

        'id': test_df['id'],

        'prediction': test_df[out_clm]

    })

    if out_clm=='prediction': out_clm = 'blend'

    submission.to_csv('submission_'+out_clm+'.csv', index=False)
submission.head()
test_df.to_csv('test_df.csv', index=False)
pd.concat((test_df['comment_text'],(test_df[[*out_clms,'prediction']]*100).astype(np.int)),axis=1)