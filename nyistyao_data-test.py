# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")



test_df = pd.read_csv("../input/test.csv")

is_dup = train_df['is_duplicate'].value_counts()







plt.figure(figsize=(8,4))



sns.barplot(is_dup.index, is_dup.values, alpha=0.8, color=color[1])



plt.ylabel('Number of Occurrences', fontsize=12)



plt.xlabel('Is Duplicate', fontsize=12)



plt.show()
print('Total number of question pairs for training: {}'.format(len(train_df)))

print('Duplicate pairs: {}%'.format(round(train_df['is_duplicate'].mean()*100, 2)))

qids = pd.Series(train_df['qid1'].tolist() + train_df['qid2'].tolist())

print('Total number of questions in the training data: {}'.format(len(

    np.unique(qids))))

print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))



plt.figure(figsize=(12, 5))

plt.hist(qids.value_counts(), bins=50)

plt.yscale('log', nonposy='clip')

plt.title('Log-Histogram of question appearance counts')

plt.xlabel('Number of occurences of question')

plt.ylabel('Number of questions')

print()
from sklearn.metrics import log_loss



p = train_df['is_duplicate'].mean() # Our predicted probability

print('Predicted score:', log_loss(train_df['is_duplicate'], np.zeros_like(train_df['is_duplicate']) + p))



df_test = pd.read_csv('../input/test.csv')

sub = pd.DataFrame({'test_id': test_df['test_id'], 'is_duplicate': p})

sub.to_csv('naive_submission.csv', index=False)

sub.head()