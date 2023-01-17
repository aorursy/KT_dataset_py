import os

import warnings

import re



import numpy as np

import pandas as pd



import matplotlib.cm as cm

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm, tqdm_notebook



%matplotlib inline

%config InlineBackend.figure_format = 'svg'



colors = cm.gist_rainbow(np.linspace(0, 1, 10))



warnings.filterwarnings("ignore")



plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')



pd.options.display.max_rows = 16

pd.options.display.max_columns = 32
text = pd.read_csv('../input/training.csv')

ciph = pd.read_csv('../input/test.csv')
text.head()
ciph.head()
ciph.difficulty.value_counts()
text['word_list'] = text.text.str.split()

text['word_num'] = text['word_list'].map(len)

text.head()
ciph_4 = ciph[ciph.difficulty == 4]

ciph_4.set_index('ciphertext_id',inplace=True)

ciph_4.drop(columns = 'difficulty')
ciph_4['ciphertextlist'] = ciph_4.ciphertext.str.split()
ciph_4.head(10)
text['word_num'].value_counts()
ciph_4['length'] = ciph_4.ciphertextlist.map(len)

ciph_4.length.value_counts()
for i in range(10):

    ciph_char = str(i)

    ciph_4[ciph_char] = ciph_4.ciphertext.map(lambda x: len(re.findall(ciph_char,x)))

    ciph_4[ciph_char] = ciph_4[ciph_char] / ciph_4.length

ciph_4.head()    
colors[0].reshape(-1,4)
for i in range(10):

    ciph_4[str(i)].plot(color = colors[i].reshape(-1,4),legend=True)
for i in range(10):

    sns.kdeplot(ciph_4[str(i)],color = colors[i])
ciph_4[[str(x) for x in range(10)]].std()/ciph_4[[str(x) for x in range(10)]].mean()