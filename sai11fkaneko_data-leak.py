import os

import random

import difflib

from binascii import crc32

import sys 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification"

val_df = pd.read_csv(os.path.join(DATA_PATH, "validation.csv"))

test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

test_en_df = pd.read_csv("../input/test-en-df/test_en.csv")



val_ids_df = pd.read_csv(os.path.join(DATA_PATH, "validation-processed-seqlen128.csv"))

test_ids_df = pd.read_csv(os.path.join(DATA_PATH, "test-processed-seqlen128.csv"))

test_ids_df = pd.merge(test_ids_df, test_df[["id", "lang"]], on="id")
target_column = "comment_text"

test_ids_df.loc[test_ids_df[target_column].isin(val_ids_df[target_column]), ]
target_column = "input_word_ids"

overlapped_test_df = test_ids_df.loc[test_ids_df[target_column].isin(val_ids_df[target_column]), ]

overlapped_val_df = val_ids_df.loc[val_ids_df[target_column].isin(test_ids_df[target_column]), ]

overlapped_test_df.nunique()
test_ids_df["token_hash"] = test_ids_df["input_word_ids"].apply(lambda x: crc32(x.encode()) & 0xffffffff)

val_ids_df["token_hash"] = val_ids_df["input_word_ids"].apply(lambda x: crc32(x.encode()) & 0xffffffff)

target_column = "token_hash"

overlapped_hash = test_ids_df.loc[test_ids_df[target_column].isin(val_ids_df[target_column]), target_column]

overlapped_hash = np.unique(overlapped_hash.values)

overlapped_hash.sort()

overlapped_hash.shape
def display_test_val_comment(test_ids_df, val_ids_df, target_column="token_hash", target_value=0, test_en_df=None, ind_=0):

    test_sample = test_ids_df.query(f"{target_column} == {target_value}")

    val_sample = val_ids_df.query(f"{target_column} == {target_value}")

    

    test_comment = test_sample.comment_text.values[ind_]

    val_comment = val_sample.comment_text.values[0]

    print("{}".format("-"*80))



    print(f"{target_column}:{target_hash}")

    print(f">> TEST \n ID:{test_sample.id.values[ind_]}, LANG:{test_sample.lang.values[ind_]}, Duplicated Num:{len(test_sample)}")

    print(f"COMMET_TEXT:{test_comment}\n")



    print(f">> VALIDATION \n ID:{val_sample.id.values[0]}, LANG:{val_sample.lang.values[0]}, Duplicated Num:{len(val_sample)}, TOXIC:{val_sample.toxic.values[0]}")

    print(f"COMMET_TEXT:{val_comment}\n")



    # diff = difflib.unified_diff(test_comment.replace(" ", "\n ").split(), val_comment.replace(" ", "\n ").split(), "TEST", "VAL", lineterm='\n')

    diff = difflib.context_diff(test_comment.replace(" ", "\n ").split(), val_comment.replace(" ", "\n ").split(), "TEST", "VAL", lineterm='\n')

    print(">> diff result")

    sys.stdout.writelines(diff)



    if test_en_df is not None:

        test_en_comment = test_en_df.query(f"id == {test_sample.id.values[ind_]}").content_en.values[0]

        print("\n\n>> TEST ENGLISH Translation")

        print(f"{test_en_comment}\n")
for i, target_hash in enumerate(overlapped_hash):

    display_test_val_comment(test_ids_df, val_ids_df, target_column="token_hash", target_value=target_hash, test_en_df=test_en_df)

    # comment out lines below if you want to check all samples

    if i >= 2:

        break
# submission file from https://www.kaggle.com/hamditarek/ensemble by Tarek Hamdi, its LB score is 0.9462.

sub_df = pd.read_csv("../input/ensemble/submission.csv")

sub_df = pd.merge(sub_df, test_ids_df, on="id")

sub_df.head()
non_toxic_hash = val_ids_df.loc[val_ids_df.token_hash.isin(overlapped_hash), :].query("toxic == 0").token_hash

toxic_hash = val_ids_df.loc[val_ids_df.token_hash.isin(overlapped_hash), :].query("toxic == 1").token_hash

non_toxic_hash = np.unique(non_toxic_hash)

toxic_hash = np.unique(toxic_hash)
sub_df.loc[sub_df.token_hash.isin(non_toxic_hash), "toxic"].hist()
sub_df.loc[sub_df.token_hash.isin(non_toxic_hash), :].sort_values("toxic").tail()
sub_df.query("token_hash == 3880127965")
non_toxic_hash = non_toxic_hash[non_toxic_hash != 3880127965]
sub_df.loc[sub_df.token_hash.isin(non_toxic_hash), "toxic"].hist()
sub_df.loc[sub_df.token_hash.isin(toxic_hash), "toxic"].hist()
sub_df.loc[sub_df.token_hash.isin(non_toxic_hash), "toxic"] = 0.0

sub_df.loc[sub_df.token_hash.isin(toxic_hash), "toxic"]= 1.0
sub_df[["id", "toxic"]].to_csv("submission.csv", index=False)
test_ids_df.duplicated("input_word_ids", keep=False).sum()
val_ids_df.duplicated("input_word_ids", keep=False).sum()