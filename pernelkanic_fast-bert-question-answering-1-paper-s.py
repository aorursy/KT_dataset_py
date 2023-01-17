from transformers import BertTokenizer, BertForQuestionAnswering

from transformers import AutoTokenizer, AutoModel

from transformers import AutoModelForQuestionAnswering, TFAutoModelForQuestionAnswering

from transformers import pipeline

import transformers

import torch

import os

from tqdm import tqdm

import json

import re

import pandas as pd

import torch

import time

import tensorflow as tf
all_papers = []

for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge'):

    for filename in tqdm(filenames):

        if ".json" in filename:

            b = json.load(open(os.path.join(dirname, filename),"r"))

            whole_text = ""

            for t in b['body_text']:

                text = re.sub(r'\[.{0,}\]', '',t['text'])

                whole_text+=text

            if "corona" in whole_text.lower() or "covid" in whole_text.lower():

                all_papers.append(whole_text)
all_results = pd.DataFrame(columns=["answer","score","start","end"])
model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
for x in tqdm(range(len(all_papers)-7000)):

    paper = all_papers[x]

    start = time.time()

    inputs = tokenizer.encode_plus(paper, "What is known about transmission, incubation, and environmental stability?" , add_special_tokens=True, return_tensors="tf", max_length=512)

    input_ids = inputs["input_ids"].numpy()[0]



    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer_start_scores, answer_end_scores = model(inputs)



    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]

    answer_end = (tf.argmax(answer_end_scores, axis=1) + 1).numpy()[0] 



    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))



    all_results.loc[x] = answer, max(answer_start_scores.numpy()[0]), 0, 0

all_results.sort_values(by=['score']).tail(10)
for result in all_results.sort_values(by=['score']).tail(10).values:

    print("-"*10+" Score: {} ".format(result[1])+"-"*10)

    print(result[0])

    