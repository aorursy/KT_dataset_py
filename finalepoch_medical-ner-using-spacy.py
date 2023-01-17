import json

with open("/kaggle/input/medical-ner/Corona2.json") as f:

    annotation = json.load(f)
TRAIN_DATA  = []

for e in annotation["examples"]:

    content = e["content"]

    entities = []

    for an in e["annotations"]:        

        if len(an["value"]) == len(an["value"].strip()):          

            if len(an['human_annotations']) == 0:

                continue

            info = (an["start"],an["end"],an["tag_name"])

            entities.append(info)

            #print(an["start"],an["end"],an["tag_name"])

    if len(entities) > 0:

        TRAIN_DATA.append(([content,{"entities":entities}]))    
from __future__ import unicode_literals, print_function

import random

from pathlib import Path

from spacy.util import minibatch, compounding

import spacy

import sys
spacy.util.use_gpu(0)

def train_model(model=None, output_dir="/kaggle/working/medical-ner", n_iter=1000):

    if model is not None:

        nlp = spacy.load(model)  # load existing spaCy model

        print("Loaded model '%s'" % model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'en' model")



    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    else:

        ner = nlp.get_pipe("ner")



    for _, annotations in TRAIN_DATA:

        for ent in annotations.get("entities"):

            ner.add_label(ent[2])



    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # only train NER

        if model is None:

            nlp.begin_training(device=0)

        for itn in range(n_iter):

            random.shuffle(TRAIN_DATA)

            losses = {}

            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 64.0, 1.2))

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(

                    texts,  

                    annotations,  

                    drop=0.20, 

                    losses=losses

                   

                )

            print("Losses", losses)



    # save model to output directory

    if output_dir is not None:

        output_dir = Path(output_dir)

        if not output_dir.exists():

            output_dir.mkdir()

        nlp.to_disk(output_dir)

        print("Saved model to", output_dir)
train_model()
nlp2 = spacy.load("/kaggle/working/medical-ner")
import numpy as np

import pandas as pd

import os

import json

import random



files = []

for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge/'):

    for filename in filenames:

        if ".json" in filename:           

            fpath = os.path.join(dirname, filename)

            if len(files) < 300:

                files.append(fpath)

random.shuffle(files)
output = []

entities = []

for i in range(0,len(files)):

    if i%100 == 0:

        print('completed ', i)

    with open(files[i]) as f:

        file_data = json.load(f)        

    for o in file_data["body_text"]: 

            doc = nlp2(o["text"],disable=['parser','tagger'])

            for ent in doc.ents:

                if len(ent.text) > 2:

                    entities.append((ent.text, ent.label_))
from collections import Counter

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = [12, 6]

pathogens = [l[0] for l in entities if l[1] == 'Pathogen']

counts = Counter(pathogens)

counts = {x : counts[x] for x in counts if counts[x] >= 20}

plt.title("Pathogens detected so far !")

plt.xticks(rotation='vertical')

plt.bar(counts.keys(),counts.values())

plt.show()

plt.savefig('path.png')
medical_conds = [l[0] for l in entities if l[1] == 'MedicalCondition']

counts = Counter(medical_conds)

counts = {x : counts[x] for x in counts if counts[x] >=20 and len(x) > 4}

plt.xticks(rotation='vertical')

plt.title("Medical Conditions detected so far !")

plt.bar(counts.keys(),counts.values(),color ="g")

plt.show()

plt.savefig('mc.png')
medicines = [l[0] for l in entities if l[1] == 'Medicine']

counts = Counter(medicines)

counts = {x : counts[x] for x in counts if counts[x] >=35 and len(x) > 4}

#plt.xticks(counts.keys(),rotation='vertical')

plt.xticks(rotation='vertical')

plt.title("Medicines detected so far !")

plt.bar(counts.keys(),counts.values(),color="y")

plt.show()

plt.savefig('med.png')