import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

% matplotlib inline

plt.style.use("ggplot")

import seaborn as sns

pal = sns.color_palette()
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.info()
# How duplicate in questions

duplicates = train.qid1.append(train.qid2).value_counts()



# visualization

plt.figure(figsize=(12, 5))

plt.hist(duplicates, bins=100)

plt.title('Log-Histogram of question appearance counts')

plt.xlabel('Number of occurences of question')

plt.ylabel('Number of questions')

plt.yscale('log', nonposy='clip')

plt.show()
from sklearn.metrics import log_loss



p = train['is_duplicate'].mean() # Our predicted probability

print('Predicted score:', log_loss(train['is_duplicate'], np.zeros_like(train['is_duplicate']) + p))



sub = pd.DataFrame({'test_id': test['test_id'], 'is_duplicate': p})

sub.to_csv('naive_submission.csv', index=False)

sub.head()