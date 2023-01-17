import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic visualisation library

import seaborn as sns # advanced and nice visualisations

import warnings # library to manage wornings

from wordcloud import WordCloud, STOPWORDS # library to create a wordcloud



warnings.simplefilter(action='ignore', category=FutureWarning) # silencing FutureWarnings out



print("numpy version: {}".format(np.__version__))

print("pandas version: {}".format(pd.__version__))

print("seaborn version: {}".format(sns.__version__))
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

#test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv') # I will use train data only in this notebook



print('There are {} rows and {} columns in a training dataset.'.format(train_data.shape[0],train_data.shape[1]))
train_data.head()
target_counts = train_data['target'].value_counts().div(len(train_data)).mul(100) # calculating percentages of target values



ax = sns.barplot(target_counts.index, target_counts.values)

ax.set_xlabel('Target variable')

ax.set_ylabel('Percentage [%]')

ax.set_xticklabels(['False','True'])

plt.show()
missing_cols = ['keyword', 'location']



fig, ax = plt.subplots(ncols=1, figsize=(5, 5))

train_nans = train_data[missing_cols].isnull().sum()/len(train_data)*100  # calculating percent of missing data in each column



sns.barplot(x=train_nans.index, y=train_nans.values, ax=ax)  # creating a barplot



ax.set_ylabel('Missing Values [%]', size=15, labelpad=20)

ax.set_yticks(np.arange(0,40,5))

ax.set_ylim((0,35))



ax.set_title('Training dataset', fontsize=13)

plt.show()
fig, ax = plt.subplots(figsize=(12,6))

true_ratios = train_data.groupby('keyword')['target'].mean().sort_values(ascending=False)

sns.barplot(x=true_ratios.index[:30], y=true_ratios.values[:30], ax=ax)

plt.xticks(rotation=90)

plt.title("TOP 30 most 'true' keywords")

plt.ylabel("True ratio")

plt.show()
fig, ax = plt.subplots(figsize=(12,6))

true_ratios = train_data.groupby('keyword')['target'].mean().sort_values(ascending=False)

sns.barplot(x=true_ratios.index[-30:], y=true_ratios.values[-30:], ax=ax)

plt.xticks(rotation=90)

plt.title("TOP 30 most 'fake' keywords")

plt.ylabel("True ratio")

plt.show()
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize # funtions for standard tokenisation

from nltk.tokenize import TweetTokenizer # function for tweets tokenization
tknzr = TweetTokenizer() # initialization of Tweet Tokenizer



def mean_words_length(text):

    words = word_tokenize(text)

    word_lengths = [len(w) for w in words]

    return round(np.mean(word_lengths),1)



# words count

train_data['words_count'] = train_data['text'].apply(lambda x: len(tknzr.tokenize(x))) # number of words in tweet



# numbers count

numbers_regex = r"(\d+\.?,?\s?\d+)"

train_data['numbers_count'] = train_data['text'].apply(lambda x: len(regexp_tokenize(x, numbers_regex))) # count of extracted mentions



# hashtags count

hashtags_regex = r"#\w+"

train_data['hashtags_count'] = train_data['text'].apply(lambda x: len(regexp_tokenize(x, hashtags_regex))) # count of extracted hashtags



# mentions count

mentions_regex = r"@\w+"

train_data['mentions_count'] = train_data['text'].apply(lambda x: len(regexp_tokenize(x, mentions_regex))) # count of extracted mentions



# mean words length

train_data['mean_words_length'] = train_data['text'].apply(mean_words_length) # count of extracted mentions



# mean words length

train_data['characters_count'] = train_data['text'].apply(lambda x: len(x)) # count of extracted mentions
train_data.head()
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))

sns.distplot(train_data[train_data['target']==1]['words_count'], label='True', ax=ax1)

sns.distplot(train_data[train_data['target']==0]['words_count'], label='Fake', ax=ax1)

ax1.legend()



sns.distplot(train_data[train_data['target']==1]['mean_words_length'], label='True', ax=ax2)

sns.distplot(train_data[train_data['target']==0]['mean_words_length'], label='Fake', ax=ax2)

ax2.legend()



sns.distplot(train_data[train_data['target']==1]['characters_count'], label='True', ax=ax3)

sns.distplot(train_data[train_data['target']==0]['characters_count'], label='Fake', ax=ax3)

ax3.legend()

plt.show()
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))



sns.countplot(x='numbers_count', hue='target', data=train_data, ax=ax1)

ax1.legend(labels=['Fake','True'], loc=1)



sns.countplot(x='hashtags_count', hue='target', data=train_data, ax=ax2)

ax2.legend(labels=['Fake','True'], loc=1)



sns.countplot(x='mentions_count', hue='target', data=train_data, ax=ax3)

ax3.legend(labels=['Fake','True'], loc=1)



plt.show()
from nltk.corpus import stopwords

import string



# lowercase tokens

train_data['lowercase_bag_o_w'] = train_data['text'].apply(lambda x: [w for w in tknzr.tokenize(x.lower())])



# stopwords

train_data['stopwords'] = train_data['lowercase_bag_o_w'].apply(lambda x: [t for t in x if t in stopwords.words('english')])



# stopwords count

train_data['stopwords_count'] = train_data['stopwords'].apply(lambda x: len(x))



# alpha words only (excludes mentions and hashtags)

train_data['alpha_only'] = train_data['lowercase_bag_o_w'].apply(lambda x: [t for t in x if t.isalpha()])



# counts of alpha words only

train_data['alpha_count'] = train_data['alpha_only'].apply(lambda x: len(x))



# counts of punctuation marks only

punctuation_regex = r"[^\w\s]"

train_data['punctuation_count'] = train_data['text'].apply(lambda x: len(regexp_tokenize(x, punctuation_regex)))

train_data.head()
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))

sns.distplot(train_data[train_data['target']==1]['stopwords_count'], label='True', ax=ax1)

sns.distplot(train_data[train_data['target']==0]['stopwords_count'], label='Fake', ax=ax1)

ax1.legend()



sns.distplot(train_data[train_data['target']==1]['alpha_count'], label='True', ax=ax2)

sns.distplot(train_data[train_data['target']==0]['alpha_count'], label='Fake', ax=ax2)

ax2.legend()



sns.distplot(train_data[train_data['target']==1]['punctuation_count'], label='True', ax=ax3)

sns.distplot(train_data[train_data['target']==0]['punctuation_count'], label='Fake', ax=ax3)

ax3.legend()

plt.show()
from PIL import Image



# defining mask

fire_mask = np.array(Image.open('/kaggle/input/images/fire-cartoon_sil.png'))

alpha_only_cloud = " ".join(train_data['alpha_only'].explode())



# defining cloud of words

cloud_words_raw = " ".join(review for review in train_data.text)

stopwords = set(STOPWORDS)

stopwords.update(["http", "https","co","com","amp"])



# creating cloud of words

fig, ax1 = plt.subplots(figsize=(8,6))

wordcloud = WordCloud(stopwords=stopwords, background_color="white",height=300, mask = fire_mask, contour_color='firebrick', contour_width=3).generate(cloud_words_raw)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# creating cloud of words

fig, ax1 = plt.subplots(figsize=(8,6))

wordcloud = WordCloud(stopwords=stopwords, background_color="white",height=300, mask = fire_mask, contour_color='firebrick', contour_width=3).generate(alpha_only_cloud)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
from nltk.util import bigrams



train_data['bigrams'] = train_data['lowercase_bag_o_w'].apply(lambda x: list(bigrams(x)))

bigrams = train_data['bigrams'].tolist()

bigrams = list([a for b in bigrams for a in b])
import collections

counter_bigrams = collections.Counter(bigrams)

top30_bigrams = counter_bigrams.most_common(30)
labels = [str(r[0]) for r in top30_bigrams]

values = [r[1] for r in top30_bigrams]



fig, ax1 = plt.subplots(figsize=(8,6))

sns.barplot(x=labels, y=values, ax=ax1)

plt.xticks(rotation=90)

plt.show()
X = train_data.drop(['id','location','bigrams','target','text','lowercase_bag_o_w','stopwords','alpha_only'], axis=1)

y = train_data['target']

X.head()
X = pd.get_dummies(X, prefix=['key'], columns=['keyword'])

X
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold



xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)
# A parameter grid for XGBoost

params_1 = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [5, 6, 7]

        }
folds = 3  # number of folds to be used

param_comb = 5 # number of parameters combinations



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1)  # define a stratified K-Fold to preserve percentage of each target class



random_search = RandomizedSearchCV(xgb, param_distributions=params_1, n_iter=param_comb, scoring=['roc_auc','accuracy','recall','precision'],

                                   n_jobs=-1, cv=skf.split(X,y), verbose=2, random_state=1001, refit='roc_auc')



# Alternatively you can perform a full grid search but it will take much longer time

#grid = GridSearchCV(estimator=xgb, param_grid=params, scoring=['roc_auc','accuracy','recall','precision'], n_jobs=4, cv=skf.split(X,y), verbose=3, refit='roc_auc')



random_search.fit(X, y) # fitting
random_search.best_params_
roc_auc_results = random_search.cv_results_['mean_test_roc_auc']

loc = np.where(roc_auc_results == np.amax(roc_auc_results))[0][0]



print("ROC_AUC = {:.3f}".format(random_search.cv_results_['mean_test_roc_auc'][loc]))

print("Precision = {:.3f}".format(random_search.cv_results_['mean_test_precision'][loc]))

print("Recall = {:.3f}".format(random_search.cv_results_['mean_test_recall'][loc]))

print("Accuracy = {:.3f}".format(random_search.cv_results_['mean_test_accuracy'][loc]))
xgb_best = random_search.best_estimator_
from sklearn.ensemble import AdaBoostClassifier



ada = AdaBoostClassifier()



params_2 = {

        'n_estimators': np.arange(10,1000,10),

        'learning_rate': np.arange(0.1,1.1,0.1),

        }



random_search_1 = RandomizedSearchCV(ada, param_distributions=params_2, n_iter=200, scoring=['roc_auc','accuracy','recall','precision'],

                                   n_jobs=-1, cv=skf.split(X,y), verbose=3, random_state=1001, refit='roc_auc')

random_search_1.fit(X, y) # fitting



print("ROC_AUC = {:.3f}".format(random_search_1.cv_results_['mean_test_roc_auc'][loc]))

print("Precision = {:.3f}".format(random_search_1.cv_results_['mean_test_precision'][loc]))

print("Recall = {:.3f}".format(random_search_1.cv_results_['mean_test_recall'][loc]))

print("Accuracy = {:.3f}".format(random_search_1.cv_results_['mean_test_accuracy'][loc]))



ada_best = random_search_1.best_estimator_
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_validate



ensemble = VotingClassifier(estimators=[("XGB",xgb_best),("ADA",ada_best)], voting="soft")



cv_results = cross_validate(ensemble, X, y, cv=3, scoring=("roc_auc","precision","recall","accuracy"), return_train_score=True)
print("ROC_AUC = {:.3f}".format(cv_results["test_roc_auc"].mean()))

print("Precision = {:.3f}".format(cv_results["test_precision"].mean()))

print("Recall = {:.3f}".format(cv_results["test_recall"].mean()))

print("Accuracy = {:.3f}".format(cv_results["test_accuracy"].mean()))#