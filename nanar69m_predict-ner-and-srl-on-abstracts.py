!pip install spacy scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz  #scispacy model
import os

import json

from spacy.tokens import DocBin

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

from pprint import pprint

import spacy

from scispacy.abbreviation import AbbreviationDetector

from scispacy.umls_linking import UmlsEntityLinker

from allennlp.predictors.predictor import Predictor
metadata = (

    pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

    .assign(publish_time=lambda df_: pd.to_datetime(df_.publish_time, errors='coerce'))

)
(metadata.publish_time

 .groupby(metadata.publish_time.dt.year)

 .count()

 .plot(kind="bar", figsize=(20,5), title="count per year")

)
biorxiv = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")

commuse = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")

noncommuse = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")

pmc = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")



biorxiv["source"] = "biorxiv"

commuse["source"] = "commuse"

noncommuse["source"] = "noncommuse"

pmc["source"] = "pmc"



all_data = pd.concat([biorxiv, commuse, noncommuse, pmc])
new_metadata = metadata[metadata.publish_time.dt.year >= 2019]

len(new_metadata)
nlp = spacy.load("en_core_sci_sm")

abbreviation_pipe = AbbreviationDetector(nlp)

nlp.add_pipe(abbreviation_pipe)

linker = UmlsEntityLinker(resolve_abbreviations=True)

nlp.add_pipe(linker)
new_metadata['abstract_spacy'] = [nlp(abstract) if pd.notnull(abstract) else None for abstract in tqdm(new_metadata.abstract)]
new_metadata.abstract_spacy.tolist()
new_metadata.abstract_spacy.iloc[0]
def replace_abbr_with_json(spacy_doc): 

    new_abbrs = []

    for short in spacy_doc._.abbreviations:

        if type(short) == dict:

            return

        short_text = short.text 

        short_start = short.start 

        short_end = short.end 

        long = short._.long_form 

        long_text = long.text 

        long_start = long.start 

        long_end = long.end 

        serializable_abbr = {"short_text": short_text, "short_start": short_start, "short_end": short_end, "long_text": long_text, "long_start": long_start, "long_end": long_end} 

        short._.long_form = None 

        new_abbrs.append(serializable_abbr) 

    spacy_doc._.abbreviations = new_abbrs

    

# cast otherwise pickling spacy docs won't work with Pandas, so save in different object

# https://github.com/allenai/scispacy/issues/205

#for doc in new_metadata.abstract_spacy:

#    if doc is not None:

#        replace_abbr_with_json(doc)
# pickling spacy docs won't work with Pandas, so save in different object

new_metadata.drop('abstract_spacy', axis=1).to_pickle('metadata_2019_2020.pkl')
from spacy.tokens import Doc

doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"], store_user_data=True)

for doc in tqdm(new_metadata.abstract_spacy):

    doc_bin.add(doc or Doc(nlp.vocab))

bytes_data = doc_bin.to_bytes()

with open('abstract_spacy_2019_2020.pkl', 'wb') as ofs:

    pickle.dump(bytes_data, ofs)