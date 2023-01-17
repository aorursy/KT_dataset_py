# we don't want Weights And Biases Logging, the Trainer class by 🤗 Transformers seems to need login credentials which I don't have.

# so bye-bye wandb

!pip uninstall -y wandb
import numpy as np

import pandas as pd

import os

from pathlib import Path

import warnings

import random

import torch 

from torch import nn

import torch.optim as optim

from sklearn.model_selection import train_test_split

import tokenizers

from transformers import RobertaModel, RobertaConfig



warnings.filterwarnings('ignore')



import torch

torch.cuda.is_available()
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False



seed = 80085

seed_everything(seed)





train_split = 0.9

max_length = 128

vocab_size = 8000  # we didn't choose 8k, 8k chose us!





# create required directories

lm_data_dir = "/kaggle/working/lm_data"

model_dir = "/kaggle/working/kidBERTa"

!mkdir {lm_data_dir}

!mkdir {model_dir}
train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train_df.head()
test_df.head()
data = train_df['text'].values.tolist() + test_df['text'].values.tolist()

print(len(data), 'total tweets (train + test)')



train_data_size = int(len(data)*train_split)

train_data = data[:train_data_size]

eval_data = data[train_data_size:]



def dump2file(d, fp):

    with open(fp, 'w') as f:

        for item in d:

            f.write("%s\n" % item)



# we need to train the tokernizer with everything we got

dump2file(data, os.path.join(lm_data_dir,'everything.txt'))



# the Language Model training data

dump2file(train_data, os.path.join(lm_data_dir,'train.txt'))



# the Language Model eval data

dump2file(eval_data, os.path.join(lm_data_dir,'eval.txt'))
from tokenizers import ByteLevelBPETokenizer



tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=[f'{lm_data_dir}/everything.txt'], vocab_size=vocab_size, min_frequency=2, special_tokens=[

    "<s>",

    "<pad>",

    "</s>",

    "<unk>",

    "<mask>",

])



# tokenizer_config = {

#     "max_len": 512

# }

# import json

# with open(f"{model_dir}/tokenizer_config.json", 'w+') as fp:

#     json.dump(tokenizer_config, fp)



tokenizer.save(model_dir)
from tokenizers.implementations import ByteLevelBPETokenizer

from tokenizers.processors import BertProcessing





tokenizer = ByteLevelBPETokenizer(

    f"{model_dir}/vocab.json",

    f"{model_dir}/merges.txt",

)

tokenizer._tokenizer.post_processor = BertProcessing(

    ("</s>", tokenizer.token_to_id("</s>")),

    ("<s>", tokenizer.token_to_id("<s>")),

)

tokenizer.enable_truncation(max_length=max_length)







tokenizer.encode("the kid shall not overfit!").tokens
# let's reload, else we'll get complains.



from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, max_len=max_length)
from transformers import RobertaConfig



config = RobertaConfig(

    vocab_size=vocab_size,

    intermediate_size=256,

    max_position_embeddings=256+2,

    num_attention_heads=1,

    num_hidden_layers=2,

    type_vocab_size=1,

    hidden_size=128,

)



# save the config for later use

config.to_json_file(f"{model_dir}/config.json")
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

model
from transformers import DataCollatorForLanguageModeling



data_collator = DataCollatorForLanguageModeling(

    tokenizer=tokenizer, mlm=True, mlm_probability=0.15

)
%%time

from transformers import LineByLineTextDataset



train_dataset = LineByLineTextDataset(

    tokenizer=tokenizer,

    file_path=f'{lm_data_dir}/train.txt',

    block_size=128,

)



eval_dataset = LineByLineTextDataset(

    tokenizer=tokenizer,

    file_path=f'{lm_data_dir}/eval.txt',

    block_size=128,

)
from transformers import Trainer, TrainingArguments



EPOCHS = 20



training_args = TrainingArguments(

    learning_rate=1e-3,

    output_dir=model_dir,

    overwrite_output_dir=True,

    num_train_epochs=EPOCHS,

    per_gpu_train_batch_size=128,

    save_steps=0,

    save_total_limit=1,

    do_eval=True,

    logging_steps=200,

    evaluate_during_training=True,

    seed=seed

)



trainer = Trainer(

    model=model,

    args=training_args,

    data_collator=data_collator,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    prediction_loss_only=True,

)
%%time

trainer.train()
trainer.evaluate(eval_dataset)
trainer.save_model(model_dir)
!ls {model_dir}
kidBERTa_config = RobertaConfig.from_pretrained(f'{model_dir}/config.json', output_hidden_states=True)    

kidBERTa = RobertaModel.from_pretrained(f'{model_dir}/pytorch_model.bin', config=kidBERTa_config)

kidBERTa