import tensorflow as tf

try:

   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

   print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

except ValueError:

   tpu = None

if tpu:

   tf.config.experimental_connect_to_cluster(tpu)

   tf.tpu.experimental.initialize_tpu_system(tpu)

   strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

   strategy = tf.distribute.get_strategy()
!pip install nlp

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly  --apt-packages libomp5 libopenblas-dev
%%time

%autosave 60



import os

os.environ['XLA_USE_BF16'] = "1"

os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'



import gc

gc.enable()

import time



import numpy as np

import pandas as pd

from tqdm import tqdm 



import nlp

import transformers

from transformers import (AdamW, 

                          XLMRobertaTokenizer, 

                          XLMRobertaModel, 

                          get_cosine_schedule_with_warmup)



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.distributed import DistributedSampler



import torch_xla

import torch_xla.core.xla_model as xm

import torch_xla.debug.metrics as met

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.utils.serialization as xser

import torch_xla.version as xv



import warnings

warnings.filterwarnings("ignore")



print('PYTORCH:', xv.__torch_gitrev__)

print('XLA:', xv.__xla_gitrev__)
test = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')

sample_submission = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')
TRAIN_BATCH_SIZE = 16

VALID_BATCH_SIZE = 16

EPOCHS = 3

MAX_LEN = 80

# Scale learning rate to 8 TPU's

LR = 2e-5 * xm.xrt_world_size() 

METRICS_DEBUG = True

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
# encoding

def convert_to_features(batch):

    input_pairs = list(zip(batch['premise'], batch['hypothesis']))

    encodings = tokenizer.batch_encode_plus(input_pairs, 

                                            add_special_tokens=True, 

                                            pad_to_max_length=True, 

                                            max_length=MAX_LEN, 

                                            truncation=True, 

                                            return_attention_mask=True, 

                                            return_token_type_ids=True)

    return encodings
test_dataset = nlp.load_dataset('csv', data_files=['../input/contradictory-my-dear-watson/test.csv'])['train']

drop_columns = test_dataset.column_names

encoded_test_dataset = test_dataset.map(convert_to_features, batched=True, remove_columns=drop_columns)

encoded_test_dataset.set_format("torch", columns=['attention_mask', 'input_ids', 'token_type_ids']) 
class XLMRoberta(nn.Module):

    def __init__(self, num_labels, multisample):

        super(XLMRoberta, self).__init__()

        output_hidden_states = False

        self.num_labels = num_labels

        self.multisample= multisample

        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-large", 

                                                       output_hidden_states=output_hidden_states, 

                                                       num_labels=1)

        self.layer_norm = nn.LayerNorm(1024*2)

        self.dropout = nn.Dropout(p=0.2)

        self.high_dropout = nn.Dropout(p=0.5)        

        self.classifier = nn.Linear(1024*2, self.num_labels)

    

    def forward(self,

        input_ids=None,

        attention_mask=None,

        token_type_ids=None,

        position_ids=None,

        head_mask=None,

        inputs_embeds=None):

        outputs = self.roberta(input_ids,

                               attention_mask=attention_mask,

                               token_type_ids=token_type_ids,

                               position_ids=position_ids,

                               head_mask=head_mask,

                               inputs_embeds=inputs_embeds)

        average_pool = torch.mean(outputs[0], 1)

        max_pool, _ = torch.max(outputs[0], 1)

        concatenate_layer = torch.cat((average_pool, max_pool), 1)

        normalization = self.layer_norm(concatenate_layer)

        if self.multisample:

            # Multisample Dropout

            logits = torch.mean(

                torch.stack(

                    [self.classifier(self.dropout(normalization)) for _ in range(5)],

                    dim=0,

                ),

                dim=0,

            )

        else:

            logits = self.dropout(normalization)

            logits = self.classifier(logits)       

        outputs = logits

        return outputs  
TEST_BATCH_SIZE = 32



test_data_loader = DataLoader(

    encoded_test_dataset, 

    batch_size=TEST_BATCH_SIZE,

    drop_last=False,

    num_workers=4,

    shuffle=False

)



WRAPPED_MODEL = xmp.MpModelWrapper(XLMRoberta(num_labels=3, multisample=False))



device = xm.xla_device()

model = WRAPPED_MODEL.to(device).eval()

model.load_state_dict(torch.load("../input/contradictorywatsonpublicxlmroberta/model.bin"))
test_preds = []



for i, data in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):

    ids = data["input_ids"]

    mask = data["attention_mask"]

    type_ids = data["token_type_ids"]

    ids = ids.to(device, dtype=torch.long)

    mask = mask.to(device, dtype=torch.long)

    type_ids = type_ids.to(device, dtype=torch.long)

    outputs = model(

        input_ids = ids,

        attention_mask = mask,

        token_type_ids = type_ids

    )

    outputs_np = outputs.cpu().detach().numpy().tolist()

    test_preds.extend(outputs_np)
test_preds = torch.FloatTensor(test_preds)

top1_prob, top1_label = torch.topk(test_preds, 1)

y = top1_label.cpu().detach().numpy()

sample_submission.prediction = y

sample_submission.to_csv('submission.csv', index=False)