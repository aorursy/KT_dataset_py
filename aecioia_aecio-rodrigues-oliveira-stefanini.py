# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.dtypes
df.diagnosis.value_counts()
df.describe()
df1 = df.groupby('diagnosis')
df1.describe()
grouped_corr = df1.corr()

grouped_corr
import matplotlib.pyplot as plt



fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(25, 25))



# Para virar um array 1D

axes = axes.reshape(-1)





features = df.columns[1:32]

num_features = len(features)



# Matrix transposta. Para acesso ficar mais fácil

grouped_corr_t = grouped_corr.T



# Retorna todos os grupos

groups = grouped_corr_t.keys().levels[0]



# Itera sobre eixos e grupos

for ax, group in zip(axes, groups):

  

  # Carrega matriz de correlação correspondente

  corr = grouped_corr_t[group]

  

  cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1, extent=[0, num_features, 0, num_features])

  ax.set_title(group, y=-0.05)

  ax.set_xticks(np.arange(.5, num_features))

  ax.set_yticks(np.arange(.5, num_features)) 

  ax.set_xticklabels(features, rotation=90)

  ax.set_yticklabels(features[::-1])





  

# Exibe imagem

plt.subplots_adjust(wspace = .25)