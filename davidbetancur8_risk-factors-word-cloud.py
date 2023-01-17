import os

import re

import json

from tqdm import tqdm

import pandas as pd

from collections import Counter

from stop_words import get_stop_words

from wordcloud import WordCloud

import matplotlib.pyplot as plt
cord_path = '../input/CORD-19-research-challenge'

dirs = ["biorxiv_medrxiv", "comm_use_subset", "noncomm_use_subset", "custom_license"]
docs = []

for d in dirs:

    for file in tqdm(os.listdir(f"{cord_path}/{d}/{d}")):

        file_path = f"{cord_path}/{d}/{d}/{file}"

        j = json.load(open(file_path, "rb"))



        title = j["metadata"]["title"]

        authors = j["metadata"]["authors"]



        try:

            abstract = j["abstract"][0]["text"].lower()

        except:

            abstract = ""



        full_text = ""

        for text in j["body_text"]:

            full_text += text["text"].lower() + "\n\n"

        docs.append([title, authors, abstract, full_text])



df = pd.DataFrame(docs, columns=["title", "authors", "abstract", "full_text"])
risk_df = df[df["full_text"].str.contains("risk")]
texts = risk_df.full_text.values

sw = get_stop_words('en')

sw.extend(["factor", "risk", "factors", "et"])



all_words = []

for text in texts:

    sentences = re.split('[. ] |\n',text)

    for sentence in sentences:

        sentence = sentence.replace(',', '')

        if ("risk" in sentence) and ("factor" in sentence) :

            words = sentence.split()

            words = [word for word in words if word not in sw]

            all_words.append(words)

            

all_words = [item for sublist in all_words for item in sublist]
word_dict = Counter(all_words)
wc = WordCloud(background_color="white",width=1000, height=1500).generate_from_frequencies(word_dict)

fig = plt.figure(figsize=(15,15))

plt.imshow(wc, interpolation="bilinear")

plt.axis("off")

plt.show()