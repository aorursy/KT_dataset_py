import pandas as pd
import json
import gzip
import re
import nltk
import numpy as np
from stemming.porter2 import stem
from nltk.corpus import stopwords
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster.bicluster import SpectralCoclustering

nltk.download('punkt')
nltk.download('stopwords')


dat = pd.DataFrame(pd.read_csv('../input/jarchive_cleaned.csv'))
print(dat.head())
qlist = []

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

for row in dat.iterrows():
  txt = row[1]['text'].lower()
  txt = cleanhtml(txt)
  txt = re.sub(r'[^a-z ]',"",txt)
  txt = re.sub(r'  ',' ',txt)
#   txt = ' '.join([stem(w) for w in txt.split(" ")])
  qlist.append([txt,row[1]['answer'],row[1]['category']])

print(qlist[:10])
swords = set(stopwords.words('english'))
tv = TfidfVectorizer(stop_words = swords , strip_accents='ascii')

queslst = [q for (q,a,c) in qlist]
qlen = len(set([c for (q,a,c) in qlist]))

mtx = tv.fit_transform(queslst)

cocluster = SpectralCoclustering(n_clusters=qlen, svd_method='arpack', random_state=0) #

t = time()
cocluster.fit(mtx)