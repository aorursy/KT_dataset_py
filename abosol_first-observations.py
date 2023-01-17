import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/train.csv')
df.shape
df.head()
df['target'].value_counts() #normalize=True)
toxic = df.loc[df['target'] == 1, 'question_text']
toxic.head()
words = df['question_text'].str.split()
words.head()
num_words = words.apply(len)
num_words.head()
sns.distplot(num_words[df['target'] == 0], label='normal')
sns.distplot(num_words[df['target'] == 1], label='toxic')
plt.xlabel('num. words')
plt.legend()
plt.show()
pd.crosstab(num_words, df['target'])
first_word = list(map(lambda l: l[0], words))
first_word[:5]
pd.Series(first_word).value_counts().head()
pd.Series(first_word).groupby(df['target']).value_counts()
first_character = list(map(lambda w: w[0], first_word))
pd.Series(first_character).groupby(df['target']).value_counts()
print(classification_report(df['target'], num_words == 1))
print(classification_report(df['target'], num_words >= 30))
test_df = pd.read_csv('../input/test.csv')
test_df.head()
words_test = test_df['question_text'].str.split()
num_words_test = words_test.apply(len)
out = pd.DataFrame({'qid': test_df['qid'], 'prediction': (num_words_test >= 30).apply(int)})
predicting_toxic = test_df.loc[out['prediction'] == 1, 'question_text']
predicting_toxic.head()
out.to_csv('submission.csv', index=False)
