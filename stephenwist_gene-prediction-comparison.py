import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sb

import Bio



print(Bio.__version__)
# Loading  data



agsts = pd.read_csv('../input/genes-augustus.csv')

gnscn = pd.read_csv('../input/genes-genscan.csv')

rfsq = pd.read_csv('../input/genes-refseq.csv')

xrfsq = pd.read_csv('../input/genes-xeno-refseq.csv')



### NOTE: cannot find gene names when searched on ensembl.org!!!

ensmbl = pd.read_csv('../input/genes-ensembl.csv')

predictions = {'AUGUSTUS':agsts, 'GENSCAN':gnscn, 'REFSEQ':rfsq, 'XENOREFSEQ':xrfsq, 'ENSEMBL':ensmbl}

agsts.head()
gnscn.head()
# It would be interesting to sort by transcription start to see the first genes of some chromosomes

agsts.sort_values(by='txStart')
for method, dataset in predictions.items():

    print(method + " # of genes: " + str(dataset.shape[0]) + '\n')
refseq = pd.read_csv('../input/genes-refseq.csv')

refseq = refseq.groupby('chrom')['chrom'].count()

refseq = refseq.loc[refseq.values > 10]



ensmbl = pd.read_csv('../input/genes-ensembl.csv')

ensmbl = ensmbl.groupby('chrom')['chrom'].count()

ensmbl = ensmbl.loc[ensmbl.values > 10]



data = {'rfsq': refseq, 'ensmbl': ensmbl}

counts = pd.DataFrame.from_dict(data=data)

counts
