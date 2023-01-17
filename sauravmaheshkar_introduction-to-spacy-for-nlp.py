import numpy as np # Linear Algebra

import pandas as pd # Data Processing, CSV file I/O (e.g. pd.read_csv)

import spacy # Fastest NLP framework for Python

from bs4 import BeautifulSoup # HTML and XML Parsing

import requests # HTTP Library for Python

import re # Regular Expression library
nlp_spacy = spacy.load("en_core_web_sm")
def url_to_string(url):

    res = requests.get(url)

    html = res.text

    soup = BeautifulSoup(html, 'html5lib')

    for script in soup(["script", "style", 'aside']):

        script.extract()

    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

custom_sentence = url_to_string('https://www.nytimes.com/2020/09/15/us/hurricane-sally.html?utm_campaign=Carbon%20Brief%20Daily%20Briefing&utm_medium=email&utm_source=Revue%20newsletter')
doc = nlp_spacy(custom_sentence) 

  

for ent in doc.ents: 

    print(ent.text, ent.start_char, ent.end_char, ent.label_) 