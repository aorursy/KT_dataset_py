import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
BASE_DIR = '../input/hello-world-in-programming-languages/'
df = pd.read_csv(os.path.join(BASE_DIR, 'index.csv'), index_col=0)

print(f'Shape of the dataset: {df.shape}')
print('Example of some languages:')

df.sample(10)
print('Number of NULL values:')

df.isna().sum()
df[df['language_name'] == 'Python3']
program = df[df['language_name'] == 'Python3']['program'].values[0]

print(program)
df[df['language_name'] == 'Emojicode']
program = df[df['language_name'] == 'Emojicode']['program'].values[0]

print(program)
df[df['language_name'] == 'C++']
program = df[df['language_name'] == 'C++']['program'].values[0]

print(program)
max_len = df['program'].str.len().max()

max_ind = df['program'].str.len().argmax()

min_len = df['program'].str.len().min()

min_ind = df['program'].str.len().argmin()
print('One of the languages with the smallest program length:')

df.iloc[min_ind]
print('One of the languages with the longest program length:')

df.iloc[max_ind]