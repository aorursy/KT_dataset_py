import numpy as np

import pandas as pd

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
filename_list = os.listdir('../input/marvel-cinematic-universe-dialogue-dataset/')

movie_list = [movie.replace('.txt','').replace('.',' ').replace('-',' ') for movie in filename_list]

dialogue_list = [open(f'../input/marvel-cinematic-universe-dialogue-dataset/{files}','r',errors='ignore').readlines() for files in filename_list]

data = {'Filename':filename_list,'Movie Name':movie_list,'Dialogues':dialogue_list}

df = pd.DataFrame(data=data)

df.head(23)
patrn = re.compile('\([A-Z ]*\)')

def filter_dialogue(dia_list):

    dia_list = [lines.replace('\n','') for lines in dia_list]

    mod_list = [patrn.sub('',lines) for lines in dia_list]

    return list(filter(None, mod_list))
df['no of dialogues'] = df['Dialogues'].apply(lambda x: len(x))

df.head(23)
df['edited_dialogues'] = df['Dialogues'].apply(lambda x: filter_dialogue(x))

df['no of dialogues_edited'] = df['edited_dialogues'].apply(lambda x: len(x))

df.head(23)