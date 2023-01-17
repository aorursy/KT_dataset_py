#okunan metinde n-gram bulma

import nltk

import pandas as pd

import re

from nltk.util import ngrams

s = open('../input/habermetni/haber.txt','r').read() #text okuma

s = s.lower() #texti küçük harfe çevirme

s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s) #texti noktalama işaretlerinden kurtarma

tokens = [token for token in s.split(" ") if token != ""] #texti tokenlarına ayırma

output = list(ngrams(tokens, 3)) #output değişkenine, istenilen n-gramı tutan listeyi atama

for i in output: # n-gram yazdırma

    print(i)