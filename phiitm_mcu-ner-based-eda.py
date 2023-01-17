import numpy as np

import pandas as pd

import re

import os

import spacy

from collections import Counter,OrderedDict

nlp = spacy.load('en_core_web_lg', parse=True, tag=True, entity=True)
filename_list = os.listdir('../input/')

movie_list = [movie.replace('.txt','').replace('.',' ').replace('-',' ') for movie in filename_list]

dialogue_list = [open(f'../input/{files}','r',errors='ignore').readlines() for files in filename_list]

data = {'Filename':filename_list,'Movie Name':movie_list,'Dialogues':dialogue_list}

df = pd.DataFrame(data=data)

df.head(23)
patrn = re.compile('\([A-Z ]*\)')

def filter_dialogue(dia_list):

    dia_list = [lines.replace('\n','') for lines in dia_list]

    mod_list = [patrn.sub('',lines) for lines in dia_list]

    return list(filter(None, mod_list))



df['no of dialogues'] = df['Dialogues'].apply(lambda x: len(x))

df['edited_dialogues'] = df['Dialogues'].apply(lambda x: filter_dialogue(x))

df['no of dialogues_edited'] = df['edited_dialogues'].apply(lambda x: len(x))

df.head(23)
def find_names(dia_list):

    name_list = []

    for lines in dia_list:

        doc = nlp(lines.lower())

        for token in list(doc.ents):

            if token.label_ == 'PERSON':

                name_list.append(token.text)

    return pd.Series([dict(OrderedDict(sorted(dict(Counter(name_list)).items(),key=lambda item: item[1],reverse=True))),name_list])





df[['Character Names Counter','Character Names List']] = df.apply(lambda x: find_names(x['edited_dialogues']),axis=1)

df.head(23)     
master_list = [item for my_list in df['Character Names List'].tolist() for item in my_list]

dict(OrderedDict(sorted(dict(Counter(master_list)).items(),key=lambda item: item[1],reverse=True)))
def find_names(dia_list):

    propn_list = []

    for lines in dia_list:

        lines = lines.replace('.','').replace(',','').replace('-','')

        doc = nlp(lines.lower())

        for token in doc:

            if token.pos_ == 'PROPN':

                propn_list.append(token.text)

    return pd.Series([dict(OrderedDict(sorted(dict(Counter(propn_list)).items(),key=lambda item: item[1],reverse=True))),propn_list])





df[['Proper Noun Counter','Proper Noun List']] = df.apply(lambda x: find_names(x['edited_dialogues']),axis=1)

df.head(23)
master_prop_list = [item for my_list in df['Proper Noun List'].tolist() for item in my_list]

dict(OrderedDict(sorted(dict(Counter(master_prop_list)).items(),key=lambda item: item[1],reverse=True)))