import pandas as pd

import numpy as np
dataPraproses = pd.read_csv("../input/dataSeetelahPraproses5.csv",encoding = "ISO-8859-1")
kamus = pd.read_csv("../input/Kamus Kata Baku.csv",encoding = "ISO-8859-1",sep=';')
map_dict = {}

for item in kamus.values:

    map_dict[item[0]] = item[1]
hasil = dataPraproses.copy()

for key,val in enumerate(hasil['text'].values):

    lis = val.split(" ")

    for idx,value in enumerate(lis):

        if value in map_dict:

            lis[idx] = map_dict[value]

    lis = ' '.join(lis)

    hasil['text'].iloc[key] = lis

hasil