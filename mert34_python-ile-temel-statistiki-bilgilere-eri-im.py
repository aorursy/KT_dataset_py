import pandas as pd

import numpy as np

import matplotlib as plt
data = pd.read_csv('../input/data.csv',sep=";")
data=data.apply(lambda x:x.str.replace(",",".")) #dosyayı okurken virgülleri nokta yapmamız gerekiyor.String algılamaması için
data.info()
#Enflasyon ve işsizlik verisi string olarak gözükmekte.Tiplerini float yapmam gerekiyor.

data['enflasyon'] = data['enflasyon'].astype('float')

data['issizlik'] = data['issizlik'].astype('float')
data.describe()
data['issizlik'].mode()
data['issizlik'].median()
data['enflasyon'].mean()
data.mean(axis = 0, skipna = True)
data['enflasyon'].std()
data['issizlik'].std()
data.cov()
data.corr()