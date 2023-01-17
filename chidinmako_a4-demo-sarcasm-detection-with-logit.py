!ls ../input/sarcasm/
# some necessary imports

import os

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns

sns.set()

from matplotlib import pyplot as plt



%config InlineBackend.figure_format = 'svg'
train_df = pd.read_csv('../input/sarcasm/train-balanced-sarcasm.csv')
train_df.head()
train_df.info()
train_df.dropna(subset=['comment'], inplace=True)
train_df['label'].value_counts()
# plt.figure(figsize=(10,4))

train_df.loc[train_df['label'] == 1, 'comment'].str.len().apply(np.log1p).hist(alpha=.5, label='sarcastic')

train_df.loc[train_df['label'] == 0, 'comment'].str.len().apply(np.log1p).hist(alpha=.5, label='normal')

plt.legend()
train_df.head()
sub_label = train_df.groupby("subreddit")['label'].agg([np.mean, np.sum, np.size])

sub_label.sort_values(by='sum', ascending=False).head(10)
sub_label[sub_label['size'] > 1000].sort_values(by='mean', ascending=False).head(10)
author_label = train_df.groupby("author")['label'].agg([np.mean, np.sum, np.size])

author_label.sort_values(by='sum', ascending=False).head(10)
author_label[author_label['size'] > 400].sort_values(by='mean', ascending=False).head(10)
score_label = train_df[train_df['score'] >= 0].groupby('score')['label'].agg([np.mean, np.sum, np.size])

score_label[score_label['size'] > 400].sort_values(by='mean', ascending=False).head(10)
score_label2 = train_df[train_df['score'] < 0].groupby('score')['label'].agg([np.mean, np.sum, np.size])

# score_label2.head(10)

score_label2[score_label2['size'] > 400].sort_values(by='mean', ascending=False).head(10)
print('Maximum score: ', train_df['score'].max(), '\n')

print('Minimum score: ', train_df['score'].min(), '\n')

print('Mean score: ', train_df['score'].mean(), '\n')

print('Standard Deviation score: ', train_df['score'].std(), '\n')

print('Median score: ', train_df['score'].median())
max_score = train_df['score'].max()

min_score = train_df['score'].min()



parent_comment_max_score = train_df.loc[train_df['score'] == max_score, 'parent_comment'].iloc[0]

parent_comment_min_score = train_df.loc[train_df['score'] == min_score, 'parent_comment'].iloc[0]



comment_max_score = train_df.loc[train_df['score'] == max_score, 'comment'].iloc[0]

comment_min_score = train_df.loc[train_df['score'] == min_score, 'comment'].iloc[0]



sarcasm_max_score = train_df.loc[train_df['score'] == max_score, 'label'].iloc[0]

sarcasm_max_score = (sarcasm_max_score == 1)



sarcasm_min_score = train_df.loc[train_df['score'] == min_score, 'label'].iloc[0]

sarcasm_min_score = (sarcasm_min_score == 1)



print('The comment "{}", scored the highest at {}, had a parent comment of "{}" and it is labelled as sarcastic: {}'

      .format(comment_max_score, max_score, parent_comment_max_score, sarcasm_max_score), '\n')



print('The comment "{}", scored the lowest at {}, had a parent comment of "{}" and it is labelled as sarcastic: {}'

      .format(comment_min_score, min_score, parent_comment_min_score, sarcasm_min_score))
train_df['date'] = pd.to_datetime(train_df['date'], yearfirst=True)

train_df['year'] = train_df['date'].apply(lambda d: d.year)

train_df.head()
year_comments = train_df.groupby('year')['label'].agg([np.mean, np.size, np.sum])

year_comments.sort_values(by='sum', ascending=False).head(10)
# plt.figure(figsize=(10,6))

year_comments['mean'].plot(kind='line')

plt.title('Rate of Sarcastic Comments by Year')

plt.ylabel('Mean Sarcastic Comments by Year')
X = train_df['comment']

y = train_df['label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17) 
tf_idf = TfidfVectorizer(ngram_range=(1,2), max_features=60000, min_df=2)

logist = LogisticRegression(n_jobs=4, solver='lbfgs', random_state=17, verbose=1)

tf_idf_logist_pipeline = Pipeline([('tf_idf', tf_idf),

                                  ('logist', logist)])
# fit

tf_idf_logist_pipeline.fit(X_train, y_train)
# predict

pred = tf_idf_logist_pipeline.predict(X_test)
# accuracy

accuracy_score(y_test, pred)
print('Accuracy score is: {:.2%}'.format(accuracy_score(y_test, pred)))
from sklearn.metrics import classification_report
classification_report(y_test, pred)
confusion_matrix(y_test, pred)
# plot confusion matrix

plt.figure(figsize=(10, 6))



conmat = pd.DataFrame(confusion_matrix(y_test, pred), index=['Not Sarcastic', 'Sarcastic'], 

                      columns=['Not Sarcastic', 'Sarcastic'])



ax = sns.heatmap(conmat, annot=True, cbar=False, cmap='viridis', linewidths=0.5, fmt='.0f')

ax.set_title('Confusion Matrix for Sarcasm Detection', fontsize=18, y=1.05)

ax.set_ylabel('Real', fontsize=12)

ax.set_xlabel('Predicted', fontsize=12)

ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')

# ax.xaxis.set_label_position('bottom')

ax.tick_params(labelsize=10)
import eli5
eli5.show_weights(estimator=tf_idf_logist_pipeline.named_steps['logist'], 

                  vec=tf_idf_logist_pipeline.named_steps['tf_idf'])
# using grid cv

from sklearn.model_selection import GridSearchCV
model = Pipeline([('tfidf',TfidfVectorizer(min_df=2)),

                    ('logit',LogisticRegression(solver='lbfgs', max_iter=3000))])

params = {'tfidf__ngram_range':[(1,1),(1,2)],'tfidf__use_idf':(True,False)}

grid = GridSearchCV(estimator=model, param_grid=params, verbose=1, n_jobs=-1, cv=3)
grid.fit(X_train, y_train)
grid.best_params_
better_model = Pipeline([('tfidf',TfidfVectorizer(min_df=2, ngram_range=(1,2), use_idf=True)),

                    ('logit',LogisticRegression(solver='lbfgs', max_iter=3000))])

better_model.fit(X_train, y_train)
better_pred = better_model.predict(X_test)
accuracy_score(y_test, better_pred)
print('Accuracy score is: {:.2%}'.format(accuracy_score(y_test, better_pred)))
confusion_matrix(y_test, better_pred)
# plot confusion matrix again

plt.figure(figsize=(10, 6))



conmat = pd.DataFrame(confusion_matrix(y_test, better_pred), index=['Not Sarcastic', 'Sarcastic'], 

                      columns=['Not Sarcastic', 'Sarcastic'])



ax = sns.heatmap(conmat, annot=True, cbar=False, cmap='viridis', linewidths=0.5, fmt='.0f')

ax.set_title('Confusion Matrix for Sarcasm Detection', fontsize=18, y=1.05)

ax.set_ylabel('Real', fontsize=12)

ax.set_xlabel('Predicted', fontsize=12)

ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')

ax.tick_params(labelsize=10)