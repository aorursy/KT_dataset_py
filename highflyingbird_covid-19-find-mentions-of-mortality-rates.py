import numpy as np

import pandas as pd

import json

import os



titles = []

abstracts = []

texts = []

for dirname, _, filenames in os.walk("/kaggle/input/CORD-19-research-challenge/"):

    for filename in filenames:

        file = os.path.join(dirname, filename)

        if file.split(".")[-1] == "json":

            with open(file,"r")as f:

                doc = json.load(f)

                titles.append(doc["metadata"]["title"])

                abstracts.append(" ".join([item["text"] for item in doc["abstract"]]))

                texts.append(" ".join([item["text"] for item in doc["body_text"]]))
from nltk.tokenize import sent_tokenize



sentences = []

for text in texts:

    if "covid" in text.lower() or "sars-cov-2" in text.lower():

        for s in sent_tokenize(text):

            if "mortality rate" in s.lower():

                sentences.append(s)
print("-", "\n- ".join(sentences))