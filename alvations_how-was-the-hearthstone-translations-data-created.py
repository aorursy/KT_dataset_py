import sys

import requests

import json

import pickle



import pandas as pd

from tqdm import tqdm



headers={

    "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com",

    "X-RapidAPI-Key": "f17416b4acmshd2335508b0e4a22p1c79e8jsnbf4792d96cfa"

  }





langs = 'enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH'.split(', ')



lang2response = {}

for lang in tqdm(langs):

    url = f'https://omgvamp-hearthstone-v1.p.rapidapi.com/cards?locale={lang}'

    response = requests.get(url, headers=headers)

    lang2response[lang] = response.json()





with open('hearthstone.pkl', 'wb') as fout:

    pickle.dump(lang2response, fout)

cards = []



for lang in lang2response:

    for deck in lang2response[lang].keys():

        for card in lang2response[lang][deck]:

            cards.append(card)



lang_header = 'deDE\tenGB\tenUS\tesES\tesMX\tfrFR\titIT\tjaJP\tkoKR\tplPL\tptBR\truRU\tthTH\tzhCN\tzhTW'

with open('hearthstone.translated.tsv', 'w') as fout:

    print('idx\tfield\t'+lang_header, end='\n', file=fout)

    for idx, group in pd.DataFrame.from_dict(cards).groupby('dbfId'):

        lang, name = zip(*sorted(pd.Series(group['name'].values,index=group['locale']).to_dict().items()))

        lang, text = zip(*sorted(pd.Series(group['text'].values,index=group['locale']).to_dict().items()))

        idx = str(idx)

        

        assert len(name) == 15

        print(f'{idx}\tname\t'+'\t'.join(name), end='\n', file=fout)

        try:

            print(f'{idx}\ttext\t'+'\t'.join(text), end='\n', file=fout)

            assert len(name) == 15

        except TypeError:

            continue

import pandas as pd
pd.read_csv('hearthstone.translated.tsv', sep='\t').to_csv('hearthstone.translated.csv', sep=',')