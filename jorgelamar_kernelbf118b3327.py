# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
genoma = pd.read_csv('../input/genome_file_description.csv')

genoma1 = pd.read_csv('../input/genome_zeeshan_usmani.csv')
genoma
genoma1
genoma1.describe()
genoma1.head(5)
genoma1.columns
genoma1.count()
genoma1.head(10)
genoma1.info()
genoma1.sort_values(by='genotype')
genoma1.tail()
genoma1.groupby('genotype').mean()
genoma2 = genoma1['genotype'].value_counts()

genoma2
#Trabalhando com gráficos

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.distplot(genoma2)
sns.distplot(genoma1['position'])
sns.distplot(genoma1['position'], kde = False, color ='blue', bins = 25)
genoma1["position"].plot.hist(bins=30, edgecolor='black')
genoma1['genotype'].value_counts().plot.bar()