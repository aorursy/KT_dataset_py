print("Loading NER model ... ")

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz -q

print("loaded.")
"""

Return a list of all chemicals present in the papers abstract of CORD-19

"""



import spacy

from pathlib import Path

import pandas as pd



"""

LOAD DATA

"""



CLEAN_DATA_PATH = Path("../input/cord-19-eda-parse-json-and-generate-clean-csv/")



biorxiv = pd.read_csv(CLEAN_DATA_PATH / "biorxiv_clean.csv")

pmc = pd.read_csv(CLEAN_DATA_PATH / "clean_pmc.csv")

comm_use = pd.read_csv(CLEAN_DATA_PATH / "clean_comm_use.csv")

noncomm_use = pd.read_csv(CLEAN_DATA_PATH / "clean_noncomm_use.csv")





"""

HELPERS

"""



# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

def chunkIt(seq, num):

    avg = len(seq) / float(num)

    out = []

    last = 0.0



    while last < len(seq):

        out.append(seq[int(last):int(last + avg)])

        last += avg



    return out







"""

MAIN

"""



papers = pd.concat(

    [biorxiv,pmc, comm_use, noncomm_use], axis=0

).reset_index(drop=True)

abstracts = list(papers['abstract'].dropna())



# Split abstracts into chunks since SpaCy has a 1 million characters limit.

abstracts_chunks = chunkIt(abstracts, 50)



nlp_bc = spacy.load('en_ner_bc5cdr_md', disable=["tagger", "parser"]) # leave only ner in the pipeline

chemical_set= set()



for chunk in abstracts_chunks:

    doc_bc = nlp_bc(' '.join(chunk))

    for ent in doc_bc.ents:

        if ent.label_ == 'CHEMICAL':

            chemical_set.add(str(ent))



# Store into pandas Series and save into disk

chemical_list_df = pd.Series(list(chemical_set), name="chemicals")



chemical_list_df.to_csv("abstract_chemical_list.csv", header=True, index=False)

chemical_list_df.head(50)