import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

from scipy.stats import norm
def like(x, pattern):

    r = re.compile(pattern)

    vlike = np.vectorize(lambda val: bool(r.fullmatch(val)))

    return vlike(x)
df = pd.read_csv('../input/medium-articles-dataset/medium_data.csv')
df
df['title'].isna().sum()
df['claps'].isna().sum()
regex = '.*[0-9]+.*'
df_numbers = df.loc[like(df['title'], regex), ['title', 'claps']]
df_not_numbers = df.loc[~like(df['title'], regex), ['title', 'claps']]
df_numbers
df_not_numbers
n = len(df_numbers.index)

n
m = len(df_not_numbers.index)

m
x_bar = df_numbers['claps'].values.mean()

x_bar
y_bar = df_not_numbers['claps'].values.mean()

y_bar
var1 = df_numbers['claps'].values.var()

var1
var2 = df_not_numbers['claps'].values.var()

var2
z = (x_bar - y_bar)/np.sqrt(var1/n + var2/m)

z
p = 1 - norm.cdf(z)

p