#!pip install spac scispacy spacy_langdetect https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_lg-0.2.3.tar.gz
import spacy 

import scispacy

import pandas as pd 

import os

import numpy as np

#import scispacy

import json

from tqdm.notebook import tqdm

from scipy.spatial import distance

import ipywidgets as widgets

from scispacy.abbreviation import AbbreviationDetector

from spacy_langdetect import LanguageDetector

# UMLS linking will find concepts in the text, and link them to UMLS. 

from scispacy.umls_linking import UmlsEntityLinker

import time
#nlp = spacy.load("en_core_sci_lg")

nlp = spacy.load("en_core_sci_lg", disable=["tagger"])

# If you're on kaggle, load the model with the following, if you run into an error:

#nlp = spacy.load("/opt/conda/lib/python3.6/site-packages/en_core_sci_lg/en_core_sci_lg-0.2.3/", disable=["tagger"])



# We also need to detect language, or else we'll be parsing non-english text 

# as if it were English. 

nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)



# Add the abbreviation pipe to the spacy pipeline. Only need to run this once.

abbreviation_pipe = AbbreviationDetector(nlp)

nlp.add_pipe(abbreviation_pipe)



# Our linker will look up named entities/concepts in the UMLS graph and normalize

# the data for us. 

linker = UmlsEntityLinker(resolve_abbreviations=True)

nlp.add_pipe(linker)
from spacy.vocab import Vocab

new_vector = nlp(

               """Single‐stranded RNA virus, belongs to subgenus 

                   Sarbecovirus of the genus Betacoronavirus.5 Particles 

                   contain spike and envelope, virions are spherical, oval, or pleomorphic 

                   with diameters of approximately 60 to 140 nm.

                   Also known as severe acute respiratory syndrome coronavirus 2, 

                   previously known by the provisional name 2019 novel coronavirus 

                   (2019-nCoV), is a positive-sense single-stranded RNA virus. It is 

                   contagious in humans and is the cause of the ongoing pandemic of 

                   coronavirus disease 2019 that has been designated a 

                   Public Health Emergency of International Concern""").vector



vector_data = {"COVID-19": new_vector,

               "2019-nCoV": new_vector,

               "SARS-CoV-2": new_vector}



vocab = Vocab()

for word, vector in vector_data.items():

    nlp.vocab.set_vector(word, vector)
print(

    nlp("COVID-19").similarity(nlp("novel coronavirus")), "\n",

    nlp("SARS-CoV-2").similarity(nlp("severe acute respiratory syndrome")), "\n",

    nlp("COVID-19").similarity(nlp("sickness caused by a new virus")))
#nlp.to_disk('/home/acorn/Documents/covid-19-en_lg')
nlp.max_length=2000000
doc = nlp("Attention deficit disorcer (ADD) is treated using various medications. However, ADD is not...")



print("Abbreviation", "\t", "Definition")

for abrv in doc._.abbreviations[0:10]:

	print(f"{abrv} \t ({abrv.start_char}, {abrv.end_char}) {abrv._.long_form}")
def df_cleaner(df):

    df.fillna("Empty", inplace=True) # If we leave floats (NaN), spaCy will break.

    for i in df.index:

        for j in range(len(df.columns)):

            if " q q" in df.iloc[i,j]:

                df.iloc[i,j] = df.iloc[i,j].replace(" q q","") # Some articles are filled with " q q q q q q q q q"



# Convenience method for lemmatizing text. This will remove punctuation that isn't part of

# a word. 

def lemmatize_my_text(doc):

    lemma_column = []

    for i in df.index:

        if df.iloc[i]["language"] == "en":

            doc = nlp(str(df.iloc[i][column]), disable=["ner","linker", "language_detector"])

            lemmatized_doc = " ".join([token.lemma_ for token in doc])

            lemma_column.append(lemmatized_doc)

        else: 

            lemma_column.append("Non-English")

    return lemma_column



#Unnabreviate text. This should be done BEFORE lemmatiztion and vectorization. 

def unnabreviate_my_text(doc):

    if len(doc._.abbreviations) > 0 and doc._.language["language"] == "en":

        doc._.abbreviations.sort()

        join_list = []

        start = 0

        for abbrev in doc._.abbreviations:

            join_list.append(str(doc.text[start:abbrev.start_char]))

            if len(abbrev._.long_form) > 5: #Increase length so "a" and "an" don't get un-abbreviated

                join_list.append(str(abbrev._.long_form))

            else:

                join_list.append(str(doc.text[abbrev.start_char:abbrev.end_char]))

            start = abbrev.end_char

        # Reassign fixed body text to article in df.

        new_text = "".join(join_list)

        # We have new text. Re-nlp the doc for futher processing!

        doc = nlp(new_text)

        return(doc)

    

def pipeline(df, column, dataType, filename):

    create = pd.DataFrame(columns={"_id","language","section","sentence","startChar","endChar","entities","lemma","w2vVector"})

    create.to_csv(filename + "_text_processed" + ".csv", index=False)

    

    docs = nlp.pipe(df[column].astype(str))

    i = -1

    for doc in tqdm(docs):

        languages = []

        start_chars = []

        end_chars = []

        entities = []

        sentences = []

        vectors = []

        _ids = []

        columns = []

        lemmas = []

        i = i + 1

        

        if doc._.language["language"] == "en" and len(doc.text) > 5:

            for sent in doc.sents:

                languages.append(doc._.language["language"])

                sentences.append(sent.text)

                vectors.append(sent.vector)

                start_chars.append(sent.start_char)

                end_chars.append(sent.end_char)

                doc_ents = []

                for ent in sent.ents: 

                    if len(ent._.umls_ents) > 0:

                        poss = linker.umls.cui_to_entity[ent._.umls_ents[0][0]].canonical_name

                        doc_ents.append(poss)

                entities.append(doc_ents)

                _ids.append(df.iloc[i,0])

                if dataType == "tables":

                    columns.append(df.iloc[i]["figure"])

                elif dataType == "text":

                    columns.append(column)

                lemmatized_doc = " ".join([token.lemma_ for token in doc])

                lemmas.append(lemmatized_doc)

        else: 

            start_chars.append(0)

            end_chars.append(len(doc.text))

            entities.append("Non-English")

            sentences.append(doc.text)

            vectors.append(np.zeros(200))

            _ids.append(df.iloc[i,0])

            languages.append(doc._.language["language"])

            if dataType == "tables":

                columns.append(df.iloc[i]["figure"])

            elif dataType == "text":

                columns.append(column)

            lemmas.append("Non-English")

            

        rows = pd.DataFrame(data={"_id": _ids, "language": languages, "section": columns, "sentence": sentences, 

            "startChar": start_chars, "endChar": end_chars, "entities": entities, "lemma": lemmas, "w2vVector":vectors})

        rows.to_csv(filename, mode='a', header=False, index=False)

        del rows
files = [f for f in os.listdir("./unnabreviated_parts/") if f.startswith("unna") and not f.endswith("csv")]

for f in tqdm(files):

    f = "./unnabreviated_parts/" + f

    df = pd.read_csv(f)

    pipeline(df=df, column="text", dataType="text", filename="tables_unnabrev_lemma")

    os.remove(f)
df_list = []

df = pd.concat([i for i in [pd.read_csv(f) for f in files]])

timestamp = time.strftime("%Y%m%d")

df.to_csv(f"covid_TitleAbstract_processed-{timestamp}.csv", index=False)