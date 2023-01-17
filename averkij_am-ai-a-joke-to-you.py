import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
jokes_data = pd.read_json('/kaggle/input/reddit-jokes-dataset/reddit_jokes.json')
jokes_data.shape
most_funny = jokes_data.sort_values('score', ascending=False).reset_index()
print('— ', list(most_funny.loc[[0]].title))

print('— ', list(most_funny.loc[[0]].body))
print('— ', list(most_funny.loc[[6]].title))

print('— ', list(most_funny.loc[[6]].body))
jokes_data['long'] = jokes_data['body'].str.len() > 200

jokes_data['body_length'] = jokes_data['body'].str.len()
sns.distplot(jokes_data['body_length'], hist=True, kde=False, 

             bins=1000, color = 'blue',

             hist_kws={'edgecolor':'black'})



plt.xlim(0, 2000)

plt.title('Jokes length')

plt.xlabel('Length')

plt.ylabel('Amount of jokes')
