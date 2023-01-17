!pip install transformers
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



data_path = "/kaggle/input/CORD-19-research-challenge/2020-03-13/"



sources = pd.read_csv(data_path + "all_sources_metadata_2020-03-13.csv")

sources = sources[["title", "abstract", "Microsoft Academic Paper ID"]].dropna(subset=['title', 'abstract'])

















from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import torch



tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")

model = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")



# todo: Probably a better way 

sources = sources[sources['abstract'].str.contains("ADE")]



# Just a test

# Put all the titles together and try answer extraction





def chunkTitles(titles, nb):

    total = len(titles)

    delta = int(total / nb)

    

    chunks = []

    for i in range(0, total, nb):

        chunks.append(titles[i: i + nb])

        

    return chunks





abstracts = sources['abstract'].astype(str)



question = "What are the methods for evaluating complication of Antibody-Dependent Enhancement?"





for abstracts_chunked in chunkTitles(abstracts, 1):

    abstracts_together = " ".join(("".join(abstracts_chunked)).split(" ")[:200])

    

    encoded_question = tokenizer.encode_plus(question, abstracts_together, add_special_tokens=True, return_tensors="pt")

    input_ids = encoded_question["input_ids"]

    answer_start_scores, answer_end_scores = model(input_ids)



    answer_start = torch.argmax(answer_start_scores)

    answer_end = torch.argmax(answer_end_scores)



    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start.data:answer_end.data].tolist()))

    

    

    if len(answer) > 0:

        print(answer)

        print()

        

        

#todo: Maybe take all the responses and make a summary?
