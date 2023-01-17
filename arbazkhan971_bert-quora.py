# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Install dependencies

!pip uninstall -y tensorflow

!pip install transformers

# transformers version at notebook creation --- 2.5.1

# tokenizers version at notebook creation --- 0.5.2
from pathlib import Path
# paths = [str(x) for x in Path(".").glob("/kaggle/input/quoratext/file_text.txt")]

# paths
%%time 

from pathlib import Path



from tokenizers import ByteLevelBPETokenizer





# Initialize a tokenizer

tokenizer = ByteLevelBPETokenizer()



# Customize training

tokenizer.train(files="/kaggle/input/quoratext/file_text.txt", vocab_size=52_000, min_frequency=2, special_tokens=[

    "<s>",

    "<pad>",

    "</s>",

    "<unk>",

    "<mask>",

])
!mkdir EsperBERTo

tokenizer.save("EsperBERTo")
from tokenizers.implementations import ByteLevelBPETokenizer

from tokenizers.processors import BertProcessing





tokenizer = ByteLevelBPETokenizer(

    "./EsperBERTo/vocab.json",

    "./EsperBERTo/merges.txt",

)
tokenizer._tokenizer.post_processor = BertProcessing(

    ("</s>", tokenizer.token_to_id("</s>")),

    ("<s>", tokenizer.token_to_id("<s>")),

)

tokenizer.enable_truncation(max_length=512)
tokenizer.encode("Mi estas Julien.")
!nvidia-smi
# Check that PyTorch sees it

import torch

torch.cuda.is_available()
# Get the example scripts.

!wget -c https://raw.githubusercontent.com/huggingface/transformers/master/examples/run_language_modeling.py
import json

config = {

	"architectures": [

		"RobertaForMaskedLM"

	],

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

	"vocab_size": 520

}

with open("./EsperBERTo/config.json", 'w') as fp:

    json.dump(config, fp)



tokenizer_config = {

	"max_len": 512

}

with open("./EsperBERTo/tokenizer_config.json", 'w') as fp:

    json.dump(tokenizer_config, fp)
cmd =	"""

  python run_language_modeling.py

  --train_data_file ../input/quoratext/file_text.txt

  --output_dir ./EsperBERTo-small-v1

	--model_type roberta

	--mlm

	--config_name ./EsperBERTo

	--tokenizer_name ./EsperBERTo

	--do_train

	--line_by_line

	--learning_rate 1e-4

	--num_train_epochs 1

	--save_total_limit 2

	--save_steps 500

	--per_gpu_train_batch_size 16

	--seed 42

""".replace("\n", " ")
%%time

!{cmd}
from transformers import pipeline



fill_mask = pipeline(

    "fill-mask",

    model="../working/EsperBERTo-small-v1",

    tokenizer="../working/EsperBERTo-small-v1"

)



# The sun <mask>.

# =>



result = fill_mask("Data Science <mask>.")
result = fill_mask("Data Science in <mask>.")

result
import re
from nltk.translate.bleu_score import sentence_bleu

def evaluation(actual,autocomplete):

    result = fill_mask(autocomplete)

    sentances = [i['sequence'] for i in result]

    list_of_sentances = []

    print(sentances)

    for i in sentances:

        line = re.sub('<s>', '', i)

        line = re.sub("</s>","",line)

        list_of_sentances.append(line.split())

    print(list_of_sentances)

    print(actual)

    score = sentence_bleu(list_of_sentances,actual.split())

    print(score)

    return score

    

    
print(evaluation("what is datascience","what is <mask>"))