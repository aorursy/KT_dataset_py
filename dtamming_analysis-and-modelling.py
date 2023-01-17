!pip install Afinn
import numpy as np

import pandas as pd

from tqdm import tqdm

import time

import nltk

from afinn import Afinn

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns
stop_words = set(nltk.corpus.stopwords.words('english'))
data = pd.read_csv('/kaggle/input/national-hockey-league-interviews/interview_data.csv').set_index('RowId')

data['date'] = pd.to_datetime(data.date)
data.isna().sum()
sns.countplot(data.job, order=data.job.value_counts().index);
data = data[data.job != 'other']
plt.figure(figsize=(10, 5))

sns.countplot(data.date.map(lambda x: x.year)).set_title('Number of Interviews Each Year');
data['num_words'] = data.text.map(lambda x: len(x.split()))
sns.distplot(data.num_words);
sns.distplot(np.log(data.num_words)).set(xlabel='Natural Logarithm of num_words', title='Skew: {}'.format(round(np.log(data.num_words).skew(),3)));
kwargs = {'cumulative': True}

sns.distplot(data.num_words, kde=False, norm_hist=True, hist_kws=kwargs).set(ylabel='% of Entries');
afinn = Afinn()

# this takes 50 seconds

data['sentiment'] = data.text.map(lambda x: afinn.score(x)/len(x.split()))
sns.distplot(data.sentiment);
low_sent = data.loc[data.sentiment.idxmin()]

print(low_sent.drop(columns=['text']))

print('\n'+low_sent.text)
high_sent = data.loc[data.sentiment.idxmax()]

print(high_sent.drop(columns=['text']))

print('\n'+high_sent.text)
sns.scatterplot(data.num_words, data.sentiment);
sns.regplot(data.num_words, data.sentiment*data.num_words);

print(data.num_words.corr(data.sentiment*data.num_words))
sns.violinplot(x='job', y='sentiment', data=data);
player_sentiment = data[data.job == 'player']['sentiment']

coach_sentiment = data[data.job == 'coach']['sentiment']

_, p, _ = sm.stats.ttest_ind(player_sentiment, coach_sentiment, alternative='larger', usevar='unequal')

print(player_sentiment.mean())

print(coach_sentiment.mean())

print(p)
selfish_lexicon = {

    'i':1,'my':1,'i\'m':1,'i\'ve':1,'i\'ll':1,'myself':1,

    'we':-1, 'our':-1,'us':-1,'we\'re':-1,'we\'ve':-1,

    'we\'ll':-1,'ourselves':-1

}

start_time = time.time()

data['selfishness'] = data.text.apply(lambda x: sum([selfish_lexicon.get(w, 0) for w in x.split()])/len(x.split()))

print(time.time() - start_time)
sns.violinplot(x='job', y='selfishness', data=data);
player_selfishness = data[data.job == 'player']['selfishness']

coach_selfishness = data[data.job == 'coach']['selfishness']

_, p, _ = sm.stats.ttest_ind(player_selfishness, coach_selfishness, alternative='two-sided', usevar='unequal')

print(player_selfishness.mean())

print(coach_selfishness.mean())

print(p)
np.random.seed(314)

mask = np.random.random(len(data)) < 0.8

y = (data.job == 'player').astype(int).values

X = data.text.values

y_train = y[mask]

y_test = y[~mask]

X_train = X[mask]

X_test = X[~mask]

print((y_train==1).sum()/len(y_train))
params = {

    'C': [10**i for i in range(-4, 8)]

}

kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=314)

lr_pipe_grid = make_pipeline(

    CountVectorizer(),

    TfidfTransformer(),

    GridSearchCV(LogisticRegression(max_iter=10**4), params, cv=kfold)

)

lr_pipe_grid.fit(X_train, y_train)

grid = lr_pipe_grid[2]

print(grid.best_params_)

if len(params) == 1:

    param_name = next(iter(params))

    param_list = next(iter(params.values()))

    accs = np.array(grid.cv_results_['mean_test_score'])

    sns.lineplot(param_list, accs).set(xscale='log', xlabel=param_name, ylabel='Mean CV Accuracy');
# this cell takes approximately 3 minutes to run

start_time = time.time()

params = {

    'C': [10**i for i in range(2, 8)]

}

kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=30, random_state=314)

lr_pipe_grid = make_pipeline(

    CountVectorizer(),

    TfidfTransformer(),

    GridSearchCV(LogisticRegression(max_iter=10**4), params, cv=kfold)

)

lr_pipe_grid.fit(X_train, y_train)

grid = lr_pipe_grid[2]

print(grid.best_params_)

if len(params) == 1:

    param_name = next(iter(params))

    param_list = next(iter(params.values()))

    accs = np.array(grid.cv_results_['mean_test_score'])

    sns.lineplot(param_list, accs).set(xscale='log', xlabel=param_name, ylabel='Mean CV Accuracy');

print(time.time() - start_time)
lr_pipe = make_pipeline(

    CountVectorizer(), 

    TfidfTransformer(), 

    LogisticRegression(C=10**6, class_weight='balanced')

)
lr_pipe.fit(X_train, y_train);
y_pred = lr_pipe.predict(X_test)

(y_pred == y_test).sum()/len(y_pred)
fig, ax = plt.subplots(figsize = (10,7))

conf_mat = confusion_matrix(y_test, y_pred)/len(y_pred)

sns.heatmap(conf_mat, annot=True, ax=ax).set(xlabel='Predicted', ylabel='True');
prob_preds = lr_pipe.predict_proba(X_test)

prob_preds = prob_preds[:,1]
fpr, tpr, threshold = roc_curve(y_test, prob_preds)

roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

'AUC: {}'.format(round(roc_auc, 3))
lr_prec, lr_rec, _ = precision_recall_curve(y_test, prob_preds)

baseline = (y_test==1).sum()/len(y_test)

sns.lineplot(lr_rec, lr_prec, label='Logistic Regression').set(xlabel='Recall', ylabel='Precision');

sns.lineplot([0, 1], [baseline, baseline], label='Baseline');

lr_f1 = f1_score(y_test, y_pred)

lr_auc = auc(lr_rec, lr_prec)

'F1: {}, AUC: {}'.format(round(lr_f1, 3), round(lr_auc, 3))