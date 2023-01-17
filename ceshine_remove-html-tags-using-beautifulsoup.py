import os

import re

import html as ihtml



import pandas as pd

from bs4 import BeautifulSoup
input_dir = '../input'



answers = pd.read_csv(os.path.join(input_dir, 'answers.csv'))
def clean_text(text):

    text = BeautifulSoup(ihtml.unescape(text)).text

    text = re.sub(r"http[s]?://\S+", "", text)

    text = re.sub(r"\s+", " ", text)    

    return text
sample_text = answers.loc[31425, "answers_body"]

sample_text
sample_text = ihtml.unescape(sample_text)

sample_text
sample_text = BeautifulSoup(sample_text, "lxml").text

sample_text
sample_text = re.sub(r"\s+", " ", sample_text)

sample_text
def clean_text(text):

    text = BeautifulSoup(ihtml.unescape(text), "lxml").text

    text = re.sub(r"\s+", " ", text)

    return text
sample_text = answers.loc[13137, "answers_body"]

sample_text
sample_text = clean_text(sample_text)

sample_text
re.sub(r"http[s]?://\S+", "", sample_text)
def clean_text(text):

    text = BeautifulSoup(ihtml.unescape(text), "lxml").text

    text = re.sub(r"http[s]?://\S+", "", text)

    text = re.sub(r"\s+", " ", text)

    return text

clean_text(answers.loc[13137, "answers_body"])
import spacy

nlp = spacy.load('en_core_web_sm')

nlp.remove_pipe('parser')

nlp.remove_pipe('ner')

nlp.remove_pipe('tagger')



" ".join(

    [token.lower_ for token in 

     list(nlp.pipe([clean_text(answers.loc[13137, "answers_body"])], n_threads=1))[0]]

)
%timeit answers.loc[~answers["answers_body"].isnull(), "answers_body"].apply(clean_text)
results = answers.loc[~answers["answers_body"].isnull(), "answers_body"].apply(clean_text)
answers["answers_body"][0]
results[0]