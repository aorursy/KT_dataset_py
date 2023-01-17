import pandas as pd

df = pd.read_csv('../input/articles/articles.csv')

df.columns = ['index','text']
!pip install git+https://github.com/NVIDIA/apex --global-option="--cpp_ext" --global-option="--cuda_ext"

!pip install -U fast-bert



# !pip install -U spacy

# !pip install -U spacy-lookups-data

!pip install -U transformers

!pip install -U tokenizers

# !python3 -m spacy download en_core_web_md
import os

from fast_bert import BertLMDataBunch

from fast_bert import BertLMLearner

from fast_bert import accuracy

from pathlib import Path

from transformers import BertTokenizer

import logging

import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case='True')

logger = logging.getLogger()

device = torch.device('cuda')

if torch.cuda.device_count() > 1:

    multi_gpu = True

else:

    multi_gpu = False

metrics = []

metrics.append({'name': 'accuracy', 'function': accuracy})    

dir = Path('../working')
# from tokenizers import BertWordPieceTokenizer

# tok4 = BertWordPieceTokenizer()


# tok3 = BertWordPieceTokenizer('../input/bertvocab/bert-large-uncased-vocab.txt')
# import spacy

# import en_core_web_md

# nlp = en_core_web_md.load()
# entities = []

# for row in df.text:

#     doc = nlp(row)

#     for ent in doc.ents: 

#         if(ent.label_ in ['PERSON','FAC','ORG','GPE','LOC','EVENT','WORK_OF_ART','LAW']):

#             entities.append(ent.text)

            

# ent = pd.Series(entities)
# ent = ent.drop_duplicates()
# ent.sort_values()
# pd.options.display.max_rows = 15000

# ent
# ent.apply(lambda x: tok2.encode(x).tokens)
# tok2.add_tokens(ent.tolist())
# ent.apply(lambda x: tok2.encode(x).tokens)
#ent.to_csv('ent.csv',sep = ' ',index=False)

# tok.train('./ent.csv')
# tok.save('./','tok_trained')
# ent.apply(lambda x: tok.encode(x).tokens)
# tok3
# trained_tokens = list(tok.get_vocab().keys())
# pd.Series(trained_tokens).sort_values()
# tok3.add_tokens(trained_tokens)
# tok3
# combined_tokenz = list(tok3.get_vocab().keys())
# comb = [x for x in combined_tokenz if 'unused' not in x]
# pd.Series(comb).to_csv('ent2.csv',sep=' ',index=False)
# tok4.train('./ent2.csv')
# tok4.get_vocab()
# ent.apply(lambda x: tok4.encode(x).tokens)
# # text = df.text[0]

# ' '.join(tok4.encode(text).tokens)
# ent2 = ent.apply(lambda x:x.split(' '))
# ent2 = ent2.explode()
from box import Box

args = Box({

    "seed": 42,

    "task_name": 'news_lm',

    "model_name": 'distilbert-base-uncased',

    "model_type": 'bert',

    "train_batch_size": 16,

    "learning_rate": 4e-5,

    "num_train_epochs": 20,

    "fp16": True,

    "fp16_opt_level": "O2",

    "warmup_steps": 1000,

    "logging_steps": 0,

    "max_seq_length": 512,

    "multi_gpu": True if torch.cuda.device_count() > 1 else False

})
 
%%time

databunch_lm = BertLMDataBunch.from_raw_corpus(

                        dir,df.text.tolist(),

                        tokenizer='bert-base-uncased',

                        batch_size_per_gpu=args.train_batch_size,

                        max_seq_length=args.max_seq_length,

                        multi_gpu=args.multi_gpu,

                        model_type=args.model_type,

                        logger=logger)
learner = BertLMLearner.from_pretrained_model(

                                    dataBunch=databunch_lm,

                                    pretrained_path=args.model_name,

                                    output_dir=dir,

                                    metrics=metrics,

                                    device=device,

                                    logger=logger,

                                    multi_gpu=args.multi_gpu,

                                    logging_steps=args.logging_steps,

                                    fp16_opt_level=args.fp16_opt_level)
learner.fit(epochs=10,

            lr=5e-5,

            validate=True, 

            schedule_type="warmup_cosine")
learner.save_model()