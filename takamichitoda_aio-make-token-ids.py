!apt install aptitude -y

!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y

!pip install mecab-python3==0.996.6rc2



!pip install unidic-lite
import gc

import os

import pandas as pd

import numpy as np



import MeCab



import tqdm

import pickle



from transformers import *
class config:

    OUTPUT = "/kaggle/working"

    MAX_LEN = 512

    TOKENIZER = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")



len(config.TOKENIZER)
!wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/train_questions.json 2> /dev/null

!wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev1_questions.json 2> /dev/null

!wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev2_questions.json 2> /dev/null

!wget https://www.nlp.ecei.tohoku.ac.jp/projects/aio/labeled_entities.txt 2> /dev/null

!wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/candidate_entities.json.gz 2> /dev/null

!wget https://www.nlp.ecei.tohoku.ac.jp/projects/AIP-LB/static/aio_leaderboard.json 2> /dev/null

!gzip -d candidate_entities.json.gz

!ls
df_train_questions = pd.read_json('train_questions.json', orient='records', lines=True)

df_train_questions.head(1)
df_dev1_questions = pd.read_json('dev1_questions.json', orient='records', lines=True)

df_dev1_questions.head(1)
df_dev2_questions = pd.read_json('dev2_questions.json', orient='records', lines=True)

df_dev2_questions.head(1)
df_candidate_entities = pd.read_json('candidate_entities.json', orient='records', lines=True)

df_candidate_entities.head(1)
df_aio_leaderboard = pd.read_json("aio_leaderboard.json", orient='records', lines=True)

df_aio_leaderboard.head(1)
def tokenize_dataframe(df):

    inputs = []

    for idx in tqdm.notebook.tqdm(range(len(df))):

        question = df.loc[idx, "question"]

        answer_candidates = df.loc[idx, "answer_candidates"]



        _ids, _mask, _id_type = [], [], [] 

        for c in answer_candidates:

          context = df_candidate_entities[df_candidate_entities["title"]==c]["text"].iloc[0]

      

          tok = config.TOKENIZER.encode_plus(context,

                                             question,

                                             add_special_tokens=True,

                                             max_length=config.MAX_LEN,

                                             truncation_strategy="only_first",

                                             pad_to_max_length=True)

          _ids.append(tok["input_ids"])

          _mask.append(tok["attention_mask"])

          _id_type.append(tok["token_type_ids"])



        answer_entity = df.loc[idx, "answer_entity"]

        

        try:

            label = answer_candidates.index(answer_entity)

        except ValueError:

            label = 999  # テストデータのラベルはとりあえず999としておく



        d = {

            "input_ids": _ids,

            "attention_mask": _mask,

            "token_type_ids": _id_type, 

            "label": label

        }

        

        inputs.append(d)

    return inputs
dev1 = tokenize_dataframe(df_dev1_questions)



del df_dev1_questions

gc.collect()



with open(f"{config.OUTPUT}/dev1.pkl", "wb") as f:

    pickle.dump(dev1, f)
dev2 = tokenize_dataframe(df_dev2_questions)



del df_dev2_questions 

gc.collect()



with open(f"{config.OUTPUT}/dev2.pkl", "wb") as f:

    pickle.dump(dev2, f)
test = tokenize_dataframe(df_aio_leaderboard)



del df_aio_leaderboard 

gc.collect()



with open(f"{config.OUTPUT}/test.pkl", "wb") as f:

    pickle.dump(test, f)
train = tokenize_dataframe(df_train_questions)



del df_train_questions, df_candidate_entities

gc.collect()



with open(f"{config.OUTPUT}/train.pkl", "wb") as f:

  pickle.dump(train, f)