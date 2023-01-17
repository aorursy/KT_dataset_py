import pandas as pd
sentences = pd.read_csv('../input/sentences_detailed.csv.gz', sep='\t', header= None, names =  ['id', 'lang', 'sentence', 'user', 'date1', 'date2'])

links = pd.read_csv('../input/links.csv.gz', sep='\t', names = ['source', 'target'])
sentences.head()
links.head()
%matplotlib inline

from matplotlib import pyplot as plt

plt.figure(figsize=(12, 8))

sentences.groupby('lang')['lang'].count().sort_values(ascending=False)[:30].plot(kind= 'barh');