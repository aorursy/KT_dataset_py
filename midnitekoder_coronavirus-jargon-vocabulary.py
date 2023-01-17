# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import json

from multiprocessing import Pool

import random

import pickle

import re

from functools import reduce

# Any results you write to the current directory are saved as output.
filenames_list = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for each_filename in filenames:

        filenames_list.append(os.path.join(dirname, each_filename))
filenames_list[:20]
len(filenames_list)


# for filename in random.sample(filenames_list, 2):

#     if filename.split(".")[-1] == "json":

#         ifp = open(os.path.join(dirname, filename))

#         research_paper = json.load(ifp)

#         title = research_paper["metadata"]["title"]

#         print(title, "\n\n")

#         abstract_text = " ".join([each["text"] for each in research_paper["abstract"]])

#         print(abstract_text, "\n\n")

#         body_text = " ".join([each["text"] for each in research_paper["body_text"]])

#         print(body_text)



            
ifp = open("/kaggle/input/stopwords-compiled/stopwords_compiled.txt", "r")

stopwords = ifp.read().split("\n")
stopwords[:20]
def clean_phrase(phrase_str):

    phrase_split = phrase_str.strip().split(".")

    if len(phrase_split) == 1:

        return phrase_split[0]

    else:

        return phrase_split[-1]
def jargons_extractor(text):

    abbreviation_jargon_pairs = []

    potential_phrases = re.split(" " + " | ".join(list(stopwords)), text)

#     print(potential_phrases)

    for each_phrase in potential_phrases:

        phrase_jargon_splits = re.split("\(|\)", each_phrase)

        if len(phrase_jargon_splits) >= 2 and len(phrase_jargon_splits[0]) > 1 and len(phrase_jargon_splits[1]) > 1 and len(phrase_jargon_splits[1].split(" ")) == 1 and len(clean_phrase(phrase_jargon_splits[0]).split(" ")) > 1 and phrase_jargon_splits[1].isnumeric() == False and phrase_jargon_splits[1][-1] != "%":

#             print(clean_phrase(phrase_jargon_splits[0]).strip(), " -> ", phrase_jargon_splits[1].strip())

            abbreviation_jargon_pairs.append((phrase_jargon_splits[1].strip(), clean_phrase(phrase_jargon_splits[0]).strip()))

    return abbreviation_jargon_pairs
def process_research_paper(filename_path):

    if filename_path.split(".")[-1] == "json":

#         print(filename_path)

        ifp = open(filename_path, "r")

        research_paper = json.load(ifp)

        title = research_paper["metadata"]["title"]

        abstract_text = " ".join([each["text"] for each in research_paper["abstract"]])

        body_text = " ".join([each["text"] for each in research_paper["body_text"]])

        all_text = " {} {} {} ".format(title, abstract_text, body_text)

#         print(all_text[:10])

        return jargons_extractor(all_text)
with Pool(processes=100) as pool:

    lists_of_jargon_lists = pool.map(process_research_paper, filenames_list)
len(lists_of_jargon_lists)
lists_of_jargon_lists[90:92]
short_hand_jargon_pairs = reduce(lambda x, y: x + y, [each_list for each_list in lists_of_jargon_lists if each_list])
len(short_hand_jargon_pairs)
random.sample(short_hand_jargon_pairs, 100)
short_hand_set = set([each_pair[0] for each_pair in short_hand_jargon_pairs])
len(short_hand_set)
short_hand_pair_dict = dict(zip(list(short_hand_set), [[]] * len(short_hand_set)))
# short_hand_pair_dict
for each_pair in short_hand_jargon_pairs:

    short_hand_pair_dict[each_pair[0]] = short_hand_pair_dict[each_pair[0]] + [each_pair[1]]
short_hand_pair_dict['gp41']
for each_key in random.sample(short_hand_pair_dict.keys(), 100):

    print(each_key, " -> ", short_hand_pair_dict[each_key])
short_hand_pair_dict["aa"]