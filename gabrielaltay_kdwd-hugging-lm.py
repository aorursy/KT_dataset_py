from collections import Counter

import json

import os

from pprint import pprint

import string

import time



import matplotlib.pyplot as plt

import numpy as np

import spacy

import seaborn as sns

from tqdm import tqdm

from unidecode import unidecode



sns.set()

sns.set_context('talk')



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# split corpus file

!split -l 100000 -d /kaggle/input/kdwd-make-sentences/wikipedia_intros_sentences.txt wikipedia_intros_sentences_chunks_
!ls -lh /kaggle/*/*
!pip uninstall -y tensorflow

!pip install transformers==2.5.1

!pip install tokenizers==0.5.2
from tokenizers import ByteLevelBPETokenizer

from tokenizers.processors import BertProcessing
VOCAB_SIZE = 50_000

MAX_FREQUENCY = 2

MAX_LENGTH = 512

WORK_DIR = 'WikipediaRoberta'
!mkdir {WORK_DIR}
!ls -lh /kaggle/*/*
corpus_fname = '/kaggle/input/kdwd-make-sentences/wikipedia_intros_sentences.txt'
!head /kaggle/input/kdwd-make-sentences/wikipedia_intros_sentences.txt
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(

    files=[corpus_fname], 

    vocab_size=VOCAB_SIZE, 

    min_frequency=MAX_FREQUENCY, 

    special_tokens=[

        "<s>",

        "<pad>",

        "</s>",

        "<unk>",

        "<mask>",

])
tokenizer.save(WORK_DIR)
!ls -lh /kaggle/*/*
encoded = tokenizer.encode('I am a sentence.')

print(encoded.ids)

print(encoded.tokens)

print(encoded.type_ids)

print(encoded.offsets)

print(encoded.attention_mask)

print(encoded.special_tokens_mask)

print(encoded.overflowing)

print(encoded.original_str)

print(encoded.normalized_str)
vocab = json.load(open(os.path.join(WORK_DIR, 'vocab.json'), 'r'))
list(vocab.keys())[0:20]
!head {WORK_DIR}/merges.txt
tokenizer = ByteLevelBPETokenizer(

    f"{WORK_DIR}/vocab.json",

    f"{WORK_DIR}/merges.txt",

)

tokenizer._tokenizer.post_processor = BertProcessing(

    ("</s>", tokenizer.token_to_id("</s>")),

    ("<s>", tokenizer.token_to_id("<s>")),

)

tokenizer.enable_truncation(max_length=MAX_LENGTH)



encoded = tokenizer.encode(unidecode("My name is Gabriel."))

print(encoded)

print(encoded.tokens)
import torch

torch.cuda.is_available()
!wget -c https://raw.githubusercontent.com/huggingface/transformers/v2.5.1/examples/run_language_modeling.py
!ls -lh /kaggle/*/*
import json

config = {

    "architectures": ["RobertaForMaskedLM"],

    "attention_probs_dropout_prob": 0.1,

    "hidden_act": "gelu",

    "hidden_dropout_prob": 0.1,

    "hidden_size": 768,

    "initializer_range": 0.02,

    "intermediate_size": 3072,

    "layer_norm_eps": 1e-05,

    "max_position_embeddings": 514,

    "model_type": "roberta",

    "num_attention_heads": 12,

    "num_hidden_layers": 6,

    "type_vocab_size": 1,

    "vocab_size": VOCAB_SIZE

}

with open(os.path.join(WORK_DIR, 'config.json'), 'w') as fp:

    json.dump(config, fp)



tokenizer_config = {

    "max_len": MAX_LENGTH

}

with open(os.path.join(WORK_DIR, 'tokenizer_config.json'), 'w') as fp:

    json.dump(tokenizer_config, fp)
!ls -lh /kaggle/*/*
cmd = """

  python run_language_modeling.py

  --train_data_file {}

  --output_dir {}

  --model_type roberta

  --mlm

  --config_name {}

  --tokenizer_name {}

  --do_train

  --line_by_line

  --learning_rate 2.5e-5

  --num_train_epochs 1

  --save_total_limit 2

  --save_steps 2000

  --per_gpu_train_batch_size 4

  --seed 42

""".replace("\n", " ").format(

    '/kaggle/working/wikipedia_intros_sentences_chunks_00', 

    WORK_DIR + '-small-v1', 

    WORK_DIR, 

    WORK_DIR

)
!rm -rf WikipediaRoberta-small-v1
%%time

!{cmd}
!ls -lh /kaggle/*/*
from transformers import pipeline



fill_mask = pipeline(

    "fill-mask",

    model = WORK_DIR + '-small-v1',

    tokenizer = WORK_DIR + '-small-v1'

)
result = fill_mask("The sun <mask>.")

result
result = fill_mask("This is the beginning of a beautiful <mask>.")

result
result = fill_mask("Playing music is <mask> for your ears.")

result