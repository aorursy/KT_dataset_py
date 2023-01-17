#importing modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/tarantino.csv') #importing the csv file

df.head(10) #checking the top 10 entries in the dataframe
len(df)
pd.value_counts(df['type'])
Count = pd.value_counts(df['movie'])

Count.plot.bar(color='red')
death=df[df.type=='death']

death_count=pd.value_counts(death['movie'])

death_count.plot.bar(color='red')
word=df[df.type=='word']

word_count=pd.value_counts(word['movie'])

word_count.plot.bar(color='black')
pd.value_counts(df['word'])