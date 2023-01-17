!yes | pip install transformers==2.10.0



!apt install aptitude -y

!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y

!pip install mecab-python3==0.996.6rc2



!pip install unidic-lite
import pandas as pd



import MeCab

import tqdm

import pickle



import torch

import torch.nn as nn



from transformers import *

import tokenizers
INPUT = "/kaggle/input/squad-japanese/"

MAX_LEN = 512



TOKENIZER = BertJapaneseTokenizer.from_pretrained("bert-base-japanese")

TOKENIZER.save_pretrained("./")

TOKENIZER = tokenizers.BertWordPieceTokenizer("./vocab.txt", lowercase=True)
valid_df = pd.read_json(f"{INPUT}/valid.jsonl", orient='records', lines=True)

valid_df = valid_df[valid_df.apply(lambda x: x["end"] - x["start"], axis=1) != 0].reset_index(drop=True)



valid_df["question_join"] = valid_df["question"].map(lambda x: "".join(x.split()))

valid_df["context_join"] = valid_df["context"].map(lambda x: "".join(x.split()))

valid_df["answer"] = valid_df.apply(lambda x: "".join(x["context"].split()[x["start"]:x["end"]]), axis=1)



valid_df = valid_df[["question_join", "context_join", "answer", "id"]]

valid_df.columns = ["question", "context", "answer", "uuid"]

valid_df.head(1)
df0 = pd.read_csv("/kaggle/input/jasquad-train-bert/result_xla:0.csv")

df1 = pd.read_csv("/kaggle/input/jasquad-train-bert/result_xla:1.csv")

pred_df = pd.concat([df0, df1], axis=0)

pred_df.head(1)
result_df = pred_df.merge(valid_df, on='uuid')
result_df = result_df.sample(frac=1).reset_index(drop=True)



for idx in range(5):

    question = result_df.loc[idx, "question"]

    answer = result_df.loc[idx, "answer"]

    predict_text = result_df.loc[idx, "predict text"]

    print("Q:", question)

    print("[Answer]")

    print(answer)

    print("[predict]")

    print(predict_text)

    print("----------------------")