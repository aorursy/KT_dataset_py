# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

sns.set(font_scale=1)
train = pd.read_json("../input/train.json")
y = train['interest_level']

X = train.drop('interest_level', axis=1)
def capital_letters(string):

    return sum(1 for c in string if c.isupper())
X['capital_letters'] = X['description'].apply(capital_letters)
X['num_words'] = X['description'].apply(str.split).apply(len)

X['Capital_Letter_Rate'] = X['capital_letters'] / ( X['num_words'] + 1 )
order = ['low', 'medium', 'high']

sns.stripplot(y,X["Capital_Letter_Rate"],jitter=True,order=order)

plt.ylim(0,10)

plt.title("Capital Letter Rate Vs Interest_level");
order = ['low', 'medium', 'high']

sns.stripplot(y,X["capital_letters"],jitter=True,order=order)

plt.ylim(0,10)

plt.title("Capital Letter Rate Vs Interest_level");