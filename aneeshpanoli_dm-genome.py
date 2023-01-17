# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# AUGUSTUS is a piece of software that predicts genes ab initio using Hidden Markov Models.

file_path_augustus = os.path.join('..', 'input', 'genes-augustus.csv')

augustus_df = pd.read_csv(file_path_augustus)



augustus_df.head()
augustus_df.info()
UTR_5_df = abs(augustus_df['txStart'] - augustus_df['cdsStart'])

UTR_5_df.head()
sns.set_style('whitegrid')

sns.set_context('talk')

sns.set_palette('husl')

g = sns.boxenplot(data=UTR_5_df)
g = sns.boxenplot(data=UTR_5_df)

g.set_yscale('log')
sns.set_palette('bright')

g = sns.boxenplot(data=augustus_df['exonCount'])
mRNA_df = abs(augustus_df['txEnd'] - augustus_df['txStart'])

normalization_factor = mRNA_df.max()/augustus_df['exonCount'].max()

mRNA_df = mRNA_df / mRNA_df.max()

exon_num_df = augustus_df['exonCount']*normalization_factor

exon_num_df = exon_num_df/exon_num_df.max()


corr = mRNA_df.corr(exon_num_df)

print(corr)

exon_mRNA_df = pd.concat([exon_num_df, mRNA_df], axis=1)

exon_mRNA_df.columns = ['exonCount', 'mRNA_len']

exon_mRNA_df.head()
import matplotlib.pyplot as plt

sns.set_palette('bright')

ax = sns.scatterplot(x='mRNA_len', y='exonCount', data=exon_mRNA_df)

augustus_df['exonCount'].describe()