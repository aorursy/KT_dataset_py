import os

print(os.listdir("../input"))
import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd

%matplotlib inline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = 10, 5
train = pd.read_csv('../input/train_data.csv', encoding='utf', engine='python', index_col=0)

test = pd.read_csv('../input/test_data.csv', encoding='utf', engine='python', index_col=0)
train.shape, test.shape
targ_distr = pd.DataFrame([train.score.value_counts().rename("Abs"), 

                           (train.score.value_counts()/train.shape[0]).rename("Rel")])
display(targ_distr.T)

_ = train.score.value_counts().plot.bar()

plt.title('Score distribution')

plt.xticks(rotation=0)

plt.show()
train['text_len'] = train.text.str.len()

test['text_len'] = test.text.str.len()



train['title_len'] = train.title.str.len()

test['title_len'] = test.title.str.len()
prsntls = [.25, .5, .75, 0.9, 0.95, 0.99]

text_length = pd.DataFrame([train['text_len'].describe(percentiles=prsntls).rename("train"), 

              test['text_len'].describe(percentiles=prsntls).rename("test")])

display(text_length)
text_length = pd.DataFrame([train['title_len'].describe(percentiles=prsntls).rename("train"), 

              test['title_len'].describe(percentiles=prsntls).rename("test")])

display(text_length)
fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].set_title("Lens text")

train['text_len'].plot.hist(alpha=0.7, ax=ax[0])

test['text_len'].plot.hist(alpha=0.7, ax=ax[0])

ax[0].legend(["Train", "Test"]);



ax[1].set_title("Lens titles")

train['title_len'].plot.hist(alpha=0.7, ax=ax[1])

test['title_len'].plot.hist(alpha=0.7, ax=ax[1])

ax[1].legend(["Train", "Test"]);

plt.tight_layout()
display(train.groupby('score')['text_len'].describe(percentiles=prsntls))

display(train.groupby('score')['title_len'].describe(percentiles=prsntls))

fig, ax = plt.subplots(nrows=1, ncols=2)

_ = sns.catplot(x="score", y="text_len", data=train, kind="bar", ax=ax[0])

plt.close(_.fig)

_ = sns.catplot(x="score", y="title_len", data=train, kind="bar", ax=ax[1])

plt.close(_.fig)

plt.show()
# https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, 

                   #max_words=200, 

                   max_font_size=100, 

                   title = None, image_color=False, ax = None):

    stopwords = set(STOPWORDS)

    # more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    # stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    #max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    ax.imshow(wordcloud);

    ax.set_title(title)

    ax.axis('off');

    plt.tight_layout()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))

plot_wordcloud(train["text"], title="Word Cloud Train Text", ax=ax[0][0])

plot_wordcloud(test["text"], title="Word Cloud Test Text", ax=ax[0][1])

plot_wordcloud(train["title"], title="Word Cloud Train Title", ax=ax[1][0])

plot_wordcloud(test["title"], title="Word Cloud Test Title", ax=ax[1][1])

plt.tight_layout()
mask_pos = train['score'] == 'Позитивный'

mask_neg = train['score'] == 'Негативный'
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 16))

plot_wordcloud(train['text'], title="Word Cloud Train Text", ax=ax[0][0])

plot_wordcloud(train['text'][mask_pos], title="Word Cloud Score Pos Text", ax=ax[0][1])

plot_wordcloud(train['text'][mask_neg], title="Word Cloud Score Neg Text", ax=ax[0][2])



plot_wordcloud(train['title'], title="Word Cloud Train Text", ax=ax[1][0])

plot_wordcloud(train['title'][mask_pos], title="Word Cloud Score Pos Title", ax=ax[1][1])

plot_wordcloud(train['title'][mask_neg], title="Word Cloud Score Neg Title", ax=ax[1][2])

plt.tight_layout()
train[train.text.str.lower().str.contains('благодарность')].groupby('score').count()
# https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram

import re 



def count_regexp_occ(regexp="", text=None):

    """ Simple way to get the number of occurence of a regex"""

    return len(re.findall(regexp, text))
def get_indicators_and_clean_comments(df):

    """

    Check all sorts of content as it may help find toxic comment

    Though I'm not sure all of them improve scores

    """

    # Get length in words and characters

    df["raw_word_len"] = df["text"].apply(lambda x: len(x.split()))

    # Check number of upper case, if you're angry you may write in upper case

    df["nb_upper"] = df["text"].apply(lambda x: count_regexp_occ(r"[А-Я]", x))

    # Check for time stamp

    df["has_timestamp"] = df["text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
%%time

train_prepr = train.copy()

test_prepr = test.copy()



get_indicators_and_clean_comments(train_prepr)

get_indicators_and_clean_comments(test_prepr)
tf_idf_word = TfidfVectorizer(ngram_range=(1, 2),

                         stop_words=stopwords.words('russian'), 

                         token_pattern=('\w{1,}'),

                         #preprocessor=None,

                         analyzer='word',

                         #max_df=0.8, 

                         #min_df=10,

                         max_features=20000,

                         sublinear_tf=True,

                        )
%%time

tf_idf_model = tf_idf_word.fit(np.concatenate([train['text'], test['text']]))
%%time

train_tf_idf_vec = tf_idf_model.transform(train['text'])

test_tf_idf_vec = tf_idf_model.transform(test['text'])
from scipy.sparse import hstack
train_tf_idf_vec = hstack([train_tf_idf_vec, 

                           train_prepr[train_prepr.columns.difference(['title', 'text', 'score'])]])

test_tf_idf_vec = hstack([test_tf_idf_vec, 

                           test_prepr[test_prepr.columns.difference(['title', 'text', 'score'])]])
train_tf_idf_vec, test_tf_idf_vec
lm = LogisticRegression(#solver='newton-cg', 

                        #n_jobs=-1,

                        #solver='lbfgs',

                        # penalty='l2',

                        #tol=0.000000001,

                        random_state=42,

                        C=10, 

                        max_iter=100000)
lm_params = {'penalty':['l1', 'l2'],

             'C':[0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100],

             #'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

             #'tol' : [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.0001]

    

    

}

lm_search = GridSearchCV(estimator=lm, 

                         param_grid=lm_params, 

                         scoring ='roc_auc', 

                         cv=StratifiedKFold(10), 

                         n_jobs=-1,

                         verbose=1)
%%time

lm_search_fitted = lm_search.fit(X=train_tf_idf_vec, y=pd.factorize(train.score)[0])
lm_search_fitted.best_estimator_
pred_scores = cross_val_score(estimator=lm_search_fitted.best_estimator_, X=train_tf_idf_vec, y=pd.factorize(train.score)[0],

                scoring='roc_auc',  

                cv=10, #stratified by default

                n_jobs=-1)

display(np.mean(pred_scores))
predicts = lm_search_fitted.best_estimator_.predict_proba(test_tf_idf_vec)[:, 0]
sub = pd.DataFrame({'index': range(0, len(predicts)),

                    'score':predicts})

sub.to_csv('LRandTFIDF_sample_submission.csv', index=False)
pd.read_csv('LRandTFIDF_sample_submission.csv').head()