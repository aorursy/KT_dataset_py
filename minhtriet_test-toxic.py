VERSION = "20200220" 
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py!python pytorch-xla-env-setup.py --version $VERSION
language_dict = {0: 'base', 1: 'spanish', 2: 'english', 3: 'german', 4: 'dutch'}
import os, time
import pandas as pd
from kaggle_datasets import KaggleDatasets
import torch
from sklearn import metrics
from transformers.data.processors import InputFeatures

from tqdm import tqdm
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

BASE_FILE_PATH = "/kaggle/input/cleaned-data"
MODEL_NAME = 'xlm-roberta-base'
TRAIN_EPOCH = 20

torch.manual_seed(12)
# Detect hardware, return appropriate distribution strategy
#try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
#    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#    print('Running on TPU ', tpu.master())
#except ValueError:
#    tpu = None
#tpu = None
#if tpu:
#    tf.config.experimental_connect_to_cluster(tpu)
#    tf.tpu.experimental.initialize_tpu_system(tpu)
#    strategy = tf.distribute.experimental.TPUStrategy(tpu)
#else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
#    strategy = tf.distribute.get_strategy()

#print("REPLICAS: ", strategy.num_replicas_in_sync)
whole_train = pd.read_csv(f'{BASE_FILE_PATH}/cleaned_train.csv')

#positive_200 = whole_train[whole_train["toxic"]==1][0:199]
#negative_200 = whole_train[whole_train["toxic"]==0][0:199]

part_train = pd.read_csv(f'{BASE_FILE_PATH}/cleaned_train.csv')
part_valid = pd.read_csv(f'{BASE_FILE_PATH}/cleaned_validation.csv')
part_test = pd.read_csv(f'{BASE_FILE_PATH}/cleaned_test.csv')


SEQUENCE_LENGTH = 32

def multilingual_model(max_seq_length=SEQUENCE_LENGTH, trainable=False):
    """Build and return a multilingual BERT model and tokenizer."""
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = 2, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    return model
#def convert_df_to_tensors(tokenizer, df):
#    sentences = df['comment_text'].values
#    labels = df['toxic'].values
#    batch_encoding = tokenizer.batch_encode_plus(sentences, max_length=SEQUENCE_LENGTH, pad_to_max_length=True)
#    return torch.tensor(batch_encoding['input_ids'], dtype=torch.long), \
#           torch.tensor(batch_encoding['attention_mask'], dtype=torch.long), \
#           torch.tensor(labels)

def convert_df_to_tensors(tokenizer, df):
    sentences = df['comment_text'].values
    labels = df['toxic'].values

    features = []
    print("Starting...")
    for i in tqdm(range(len(sentences))):
        encoding = tokenizer.encode_plus(sentences[i], max_length=SEQUENCE_LENGTH, pad_to_max_length=True)
        inputs = {k: encoding[k] for k in encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features
print('Loading XLMRoberta tokenizer...')
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
#part_train_dataset = convert_df_to_tensors(tokenizer, part_train)
#part_valid_dataset = convert_df_to_tensors(tokenizer, part_valid)

#torch.save(part_train_dataset, 'train.pt')
#torch.save(part_valid_dataset, 'valid.pt')
model = multilingual_model()
from transformers import EvalPrediction
from typing import Dict
import numpy as np

def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"auc": metrics.roc_auc_score(preds, p.label)} #p.label_ids)}


from torch.utils.data.dataset import Dataset
#> xsv select toxic cleaned_train.csv | xsv frequency
#field,value,count
#toxic,0,1952248
#toxic,1,173495
#1952248 / 173495

weighted_dict = {0:1, 1:11.2524741347}

class ToxicDataset(Dataset):
    def __init__(self, input_path):
        features = torch.load(input_path)
        self.input_id = torch.tensor([x.input_ids for x in tqdm(features)], dtype=torch.long)
        self.attention_mask = torch.tensor([x.attention_mask for x in tqdm(features)], dtype=torch.long)
        self.labels = torch.tensor([x.label for x in tqdm(features)], dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def get_tensordataset(self):
        return TensorDataset(self.input_id, self.attention_mask, self.labels)

    def __getitem__(self, i):
        return self.features[i]
    
    def get_weights(self):
        return [weighted_dict[x.label] for x in self.features]

    def get_labels(self):
        return [x.label for x in self.features]
    
    def read_csv(self, csv_file_path):
        part_train = pd.read_csv("")
        #part_valid = torch.load(f'{BASE_FILE_PATH}/valid.pt')

        #part_train_data = TensorDataset(*convert_df_to_tensors(tokenizer, part_train))
        #part_valid_data = TensorDataset(*convert_df_to_tensors(tokenizer, part_valid))

        part_train_data = part_train_dataset
        part_valid_data = part_valid_dataset

        weighted_list = part_train['toxic'].map(weighted_dict)

        #train_dataloader = DataLoader(part_train_data, sampler=WeightedRandomSampler(weighted_list, part_train_data), batch_size=BATCH_SIZE)
        #valid_dataloader = DataLoader(part_valid_data)
    

part_train_dataset = ToxicDataset(f'{BASE_FILE_PATH}/train.pt')
part_valid_dataset = ToxicDataset(f'{BASE_FILE_PATH}/valid.pt').get_tensordataset();
SEQUENCE_LENGTH = 32
BATCH_SIZE = 16
MODEL_NAME = 'xlm-roberta-large'
device = torch.device('cpu')
LEARNING_RATE = 2e-5

def multilingual_model(max_seq_length=SEQUENCE_LENGTH):
    """Build and return a multilingual BERT model and tokenizer."""
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = 2, # The number of output labels--2 for binary classification.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    return model

def fit(model, num_epocs=TRAIN_EPOCH):
    global_step = 0
    model.train()
    for i_ in tqdm(range(num_epocs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, label_ids = batch
            loss, logit = model(input_ids, labels=label_ids)
#            if n_gpu > 1:
#                loss = loss.mean() # mean() to average on multi-gpu.
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = LEARNING_RATE
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        print('Loss after epoc {}'.format(tr_loss / nb_tr_steps))

model = multilingual_model(SEQUENCE_LENGTH)
print('Loading XLMRoberta tokenizer...')
# tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

args={}
args['fp16'] = False
args['gradient_accumulation_steps'] = 1
args['warmup_proportion'] = 0.01
args['learning_rate'] = 2e-4




#
train_dataloader = DataLoader(part_train_dataset.get_tensordataset(), sampler=WeightedRandomSampler(part_train_dataset.get_weights(), num_samples=BATCH_SIZE))
valid_dataloader = DataLoader(part_valid_dataset.get_tensordataset())

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters,LEARNING_RATE)
fit(model)
!ls output/kaggle/working
def eval():
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predicts = []
    with torch.no_grad():
        for input_ids, attention_mask, label in valid_dataloader:
            loss, logits = model(input_ids, labels=label)
            predicts.append(np.argmax(logits, axis=1))
    return predicts
predicts = eval()
tmp_eval_accuracy = metrics.roc_auc_score(part_valid['toxic'], predicts)
print(tmp_eval_accuracy)
n_wrong = 0
for p in range(len(predicts)):
    if part_valid['toxic'].iloc[p] != predicts[p]:
        print(part_valid.iloc[p]['comment_text'])
        print(part_valid.iloc[p]['toxic'])
        n_wrong += 1
    if n_wrong == 20:
        break
torch.save(model, 'model.pt')