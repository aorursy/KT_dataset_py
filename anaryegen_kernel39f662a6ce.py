# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize import WordPunctTokenizer

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/nyc-jobs.csv")
df.head()
df.describe()
df.info()
df['Post Until'].value_counts()
df.drop(['Post Until', 'Recruitment Contact'], axis=1, inplace=True)
df[df.isna().values]
df.drop(['Posting Date', 'Posting Updated', 'Process Date'], inplace=True, axis=1)
df.head()
df.columns
df['Average Salary'] = (df['Salary Range To'] + df['Salary Range From']) / 2
df['Salary Range From'].head()
df['Salary Range To'].head()
df['Average Salary'].head()
df['Log Avg Salary'] = np.log(df['Average Salary'])
df['Log Avg Salary']
df.columns
plt.hist(df['Average Salary'])

# plt.hist(df['Log1p Avg Salary'])
df['Average Salary'].head()
df['Log10 Avg Salary'] = np.log10(df['Average Salary'])
plt.hist(df['Log10 Avg Salary'])
text_cols = df[['Business Title', 'Minimum Qual Requirements', 'Preferred Skills', 'Preferred Skills']]

# df['Minimum Qual Requirements'] = df['Minimum Qual Requirements'].astype('str')
tokenize = WordPunctTokenizer()



for col in text_cols:

    df[col] = list(map(' '.join, map(tokenizer.tokenize, map(str.lower, df[col].astype('str')))))

# tokens = [tokenize.tokenize(word) for word in df['Minimum Qual Requirements']]

# for word in df['Minimum Qual Requirements']:

#     tokens.append(tokenize.tokenize(word))
pred_df = df[['Business Title', 'Minimum Qual Requirements', 'Preferred Skills', 'Preferred Skills', 'Log Avg Salary']]
from collections import Counter



token_counts = Counter()

# for col in df['Business Title']:

#     token_counts.update(col.split())

# for col in df['Minimum Qual Requirements']:

#     token_counts.update(col.split())

for cols in text_cols:

    for col in df[cols]:

        token_counts.update(col.split())
token_counts.most_common()[:100:10]
min_accurence = 10



tokens = {token for token, count in token_counts.items()

             if count >= min_accurence}
vocabulary_size = len(tokens)

print('Vocabulary size: ', vocabulary_size)
UNK, PAD = 'UNK', 'PAD'

tokens = [UNK, PAD] + sorted(tokens)
# from collections import defaultdict

# D = defaultdict()

# for i, s in enumerate(tokens):

#      D[s].append(i)

dict(((string, dict(i for i,w in enumerate(tokens) if w == string)) for string in tokens))
UNK_IX, PAD_IX = str(map(D.get, [UNK, PAD]))



def as_matrix(sequences, max_len=None):

    """ Convert a list of tokens into a matrix with padding """

    if isinstance(sequences[0], str):

        sequences = list(map(str.split, sequences))

        

    max_len = min(max(map(len, sequences)), max_len or float('inf'))

    

    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))

    for i,seq in enumerate(sequences):

        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]

        matrix[i, :len(row_ix)] = row_ix

    

    return matrix
print("Lines:")

print('\n'.join(df["Business Title"][::1000].values), end='\n\n')

print("Matrix:")

print(as_matrix(df["Business Title"][::1000]))
print(as_matrix(df["Business Title"][::1000]))
PAD_IX