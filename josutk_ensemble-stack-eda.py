# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip3 install mglearn
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
import re
def count_twitters_user(df):
    twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
    count = 0
    list_ = []
    for text in df['text']:
        users_in_twitter = re.findall(twitter_username_re, text)
        for user in users_in_twitter:
            list_.append(user)
        count += len(users_in_twitter)
    return len(set(list_)), set(list_)


def count_twitters_hashtags(df):
    twitter_hashtag_re = re.compile(r'#([A-Za-z0-9_]+)')
    count = 0
    list_ = []
    for text in df['text']:
        hashtags = re.findall(twitter_hashtag_re, text)
        for tags in hashtags:
            list_.append(tags)
        count += len(hashtags)
    return len(set(list_)), set(list_)

real = df_train[df_train['target']==1]
fake = df_train[df_train['target']==0]

count_real, users_real = count_twitters_user(real)
count_fake, users_fake = count_twitters_user(fake)


count_tags_real, tags_real = count_twitters_hashtags(real)
count_tags_fake, tags_fake = count_twitters_hashtags(fake)
import plotly.express as px
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Pie(
    values=list(df_train['target'].value_counts()),
    labels=['Real', 'Fake'],
   # marker_color='lightsalmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Distribution',
})

fig.show()
fig = go.Figure()
fig.add_trace(go.Bar(
    y=[count_tags_real, count_tags_fake],
    x=['Real', 'Fake'],
    marker_color='lightsalmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'unique hashtags mentions in twitters',
})
fig.show()
fig = go.Figure()
fig.add_trace(go.Bar(
    y=[count_real, count_fake],
    x=['Real', 'Fake'],
    marker_color='lightsalmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'unique "@users" mentions in twitters',
})
fig.show()
users_only_fake = list(users_fake - users_real)
print('there are', len(users_only_fake), 'mentionate only fake twitter news')
users_only_fake = list(tags_fake - tags_real)
print('there are', len(users_only_fake), 'mentionate only fake twitter news')
users_only_fake = list(users_real - users_fake)
print('there are', len(users_only_fake), 'mentionate only fake twitter real')
users_only_fake = list(tags_real - tags_fake)
print('there are', len(users_only_fake), 'mentionate only fake twitter real')
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None, stop=None):
    vec = CountVectorizer(stop_words=stop).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{"colspan": 2}, None]],
    subplot_titles=("Real twitters","Fake twitters", "All twitters"))

data = get_top_n_words(df_train['text'], 25)
new_list_words = [ seq[0] for seq in data ]
new_list_values = [ seq[1] for seq in data ]

data_real = get_top_n_words(real['text'], 25)
new_list_words_real = [ seq[0] for seq in data_real ]
new_list_values_real = [ seq[1] for seq in data_real ]

data_fake = get_top_n_words(fake['text'], 25)
new_list_words_fake = [ seq[0] for seq in data_fake ]
new_list_values_fake = [ seq[1] for seq in data_fake ]


fig.add_trace(go.Bar(x=new_list_words_real, y=new_list_values_real),
                 row=1, col=1)

fig.add_trace(go.Bar(x=new_list_words_fake, y=new_list_values_fake),
                 row=1, col=2)
fig.add_trace(go.Bar(x=new_list_words, y=new_list_values),
                 row=2, col=1)

fig.update_layout(showlegend=False, title_text="Specs with Subplot Title")
fig.show()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import mglearn
vec = CountVectorizer()

X = vec.fit_transform(df_train['text'])

components = 6

lda=LDA(n_components=components, n_jobs=-1, random_state=42)
lda_dtf=lda.fit_transform(X)
sorting=np.argsort(lda.components_)[:,::-1]
features=np.array(vec.get_feature_names())
mglearn.tools.print_topics(topics=range(components), feature_names=features,sorting=sorting, topics_per_chunk=components, n_words=15)

topic = []
for n in range(lda_dtf.shape[0]):
    topic_most_pr = lda_dtf[n].argmax()
    topic.append(topic_most_pr)
df_train['topic'] = topic
import nltk
stopwords = nltk.corpus.stopwords.words('english')
stopwords = stopwords + ["http", "https", "co"]
real = df_train[df_train['target']==1]
fake = df_train[df_train['target']==0]
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{"colspan": 2}, None]],
    subplot_titles=("Real twitters","Fake twitters", "All twitters"))

data = get_top_n_words(df_train['text'], 25, stopwords)
new_list_words = [ seq[0] for seq in data ]
new_list_values = [ seq[1] for seq in data ]

data_real = get_top_n_words(real['text'], 25, stopwords)
new_list_words_real = [ seq[0] for seq in data_real ]
new_list_values_real = [ seq[1] for seq in data_real ]

data_fake = get_top_n_words(fake['text'], 25, stopwords)
new_list_words_fake = [ seq[0] for seq in data_fake ]
new_list_values_fake = [ seq[1] for seq in data_fake ]


fig.add_trace(go.Bar(x=new_list_words_real, y=new_list_values_real),
                 row=1, col=1)

fig.add_trace(go.Bar(x=new_list_words_fake, y=new_list_values_fake),
                 row=1, col=2)
fig.add_trace(go.Bar(x=new_list_words, y=new_list_values),
                 row=2, col=1)

fig.update_layout(showlegend=False, title_text="Specs with Subplot Title")
fig.show()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

vectorizer = CountVectorizer(stop_words = stopwords)

X = vectorizer.fit_transform(df_train['text'])

chi2score = chi2(X,df_train['target'])[0]
wscores = dict(zip(vectorizer.get_feature_names(), chi2score))
dict_ = {k: v for k, v in sorted(wscores.items(), key=lambda item: item[1], reverse=True)}
keys = list(dict_.keys())
values = list(dict_.values())
fig = px.bar(x=list(keys[0:50]), y=list(values[0:50]))
fig.show()
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

vec = TfidfVectorizer(stop_words=stopwords)
X= vec.fit_transform(df_train['text'])
y = df_train['target']
xgboost_params = {'n_estimators' :[25,50,100, None],
                   'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                  'gamma':[0.5, 0.1, 1, 10],
                  'max_depth':[5, 10, 15, None]}

xgb = xgboost.XGBClassifier(random_state=42)
clf_xgb = GridSearchCV(xgb, xgboost_params, cv=5,n_jobs= 4, verbose = 1)
clf_xgb.fit(X, y)
print(clf_xgb.best_estimator_)
print(clf_xgb.best_score_)


import lightgbm as lgb
lightgbm_params ={'learning_rate':[0.0001, 0.001, 0.003, 0.01, 0.1],
                  'n_estimators':[10,20, 50, 100, None],
                 'max_depth':[4, 6, 10, 15, 20, 50, None]}
gbm = lgb.LGBMClassifier(random_state = 42)
clf_gbm = GridSearchCV(gbm, lightgbm_params, cv=5,n_jobs= 4, verbose = 1)
clf_gbm.fit(X, y)
print(clf_gbm.best_estimator_)
print(clf_gbm.best_score_)
from sklearn.svm import LinearSVC

svr_params = {'C':[0.0001, 0.001,0.01, 0.1, 1 , 10, 100]}
svr = LinearSVC(random_state=42)
clf_svr = GridSearchCV(svr, svr_params, cv=5, n_jobs=4, verbose=1)
clf_svr.fit(X, y)
print(clf_svr.best_estimator_)
print(clf_svr.best_score_)

from sklearn.ensemble import RandomForestClassifier
lightran_params ={'n_estimators':[10,20, 50, 100, None],
                 'max_depth':[4, 6, 10, 15, 20, 50, None]}
random = RandomForestClassifier(random_state = 42)
clf_random = GridSearchCV(random, lightran_params, cv=5,n_jobs= 4, verbose = 1)
clf_random.fit(X, y)
print(clf_random.best_estimator_)
print(clf_random.best_score_)
from sklearn.linear_model import LogisticRegression
lf_params ={'C':[0.0001, 0.001,0.01, 0.1, 1 , 10, 100],
           'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

lf = LogisticRegression(random_state=42)
clf_lf = GridSearchCV(lf, lf_params, cv=5,n_jobs= 4, verbose = 1)
clf_lf.fit(X, y)

print(clf_lf.best_estimator_)
print(clf_lf.best_score_)
from sklearn.ensemble import ExtraTreesClassifier
lightree_params ={'n_estimators':[10,20, 50, 100, None],
                  'max_depth':[4, 6, 10, 15, 20, 50, None]}

tree = ExtraTreesClassifier(random_state=42, n_jobs=4)
clf_tree = GridSearchCV(tree, lightree_params, cv=5, n_jobs= 4, verbose = 1)
clf_tree.fit(X, y)
print(clf_tree.best_estimator_)
print(clf_tree.best_score_)
import xgboost
from sklearn.model_selection import GridSearchCV

vec = TfidfVectorizer()
X= vec.fit_transform(df_train['text'])
y = df_train['target']
xgboost_params = {'n_estimators' :[25,50,100, None],
                   'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                  'gamma':[0.5, 0.1, 1, 10],
                  'max_depth':[5, 10, 15, None]}

xgb = xgboost.XGBClassifier(random_state=42, n_jobs=4)
clf_xgb = GridSearchCV(xgb, xgboost_params, cv=5,n_jobs= 4, verbose = 1)
clf_xgb.fit(X, y)
print(clf_xgb.best_estimator_)
print(clf_xgb.best_score_)


import lightgbm as lgb
lightgbm_params ={'learning_rate':[0.0001, 0.001, 0.003, 0.01, 0.1],
                  'n_estimators':[10,20, 50, 100, None],
                 'max_depth':[4, 6, 10, 15, 20, 50, None]}
gbm = lgb.LGBMClassifier(random_state = 42, n_jobs=4)
clf_gbm = GridSearchCV(gbm, lightgbm_params, cv=5,n_jobs= 4, verbose = 1)
clf_gbm.fit(X, y)
print(clf_gbm.best_estimator_)
print(clf_gbm.best_score_)
from sklearn.svm import LinearSVC

svr_params = {'C':[0.0001, 0.001,0.01, 0.1, 1 , 10, 100]}
svr = LinearSVC(random_state=42)
clf_svr = GridSearchCV(svr, svr_params, cv=5, n_jobs=4, verbose=1)
clf_svr.fit(X, y)
print(clf_svr.best_estimator_)
print(clf_svr.best_score_)

from sklearn.ensemble import RandomForestClassifier
lightran_params ={'n_estimators':[10,20, 50, 100, None],
                 'max_depth':[4, 6, 10, 15, 20, 50, None]}
random = RandomForestClassifier(random_state = 42, n_jobs=42)
clf_random = GridSearchCV(random, lightran_params, cv=5,n_jobs= 4, verbose = 1)
clf_random.fit(X, y)
print(clf_random.best_estimator_)
print(clf_random.best_score_)
from sklearn.ensemble import ExtraTreesClassifier
lightree_params ={'n_estimators':[10,20, 50, 100, None],
                  'max_depth':[4, 6, 10, 15, 20, 50, None]}

tree = ExtraTreesClassifier(random_state=42, n_jobs=42)
clf_tree = GridSearchCV(tree, lightree_params, cv=5, n_jobs= 4, verbose = 1)
clf_tree.fit(X, y)
print(clf_tree.best_estimator_)
print(clf_tree.best_score_)
from sklearn.linear_model import LogisticRegression
lf_params ={'C':[0.0001, 0.001,0.01, 0.1, 1 , 10, 100],
           'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

lf = LogisticRegression(random_state=42)
clf_lf = GridSearchCV(lf, lf_params, cv=5,n_jobs= 4, verbose = 1)
clf_lf.fit(X, y)

print(clf_lf.best_estimator_)
print(clf_lf.best_score_)
from sklearn.ensemble import StackingClassifier
estimators = [
    ('svc', LinearSVC(C=0.1, random_state=42)),
    ('extra', ExtraTreesClassifier(random_state=42, n_jobs=4)),
    ('random',RandomForestClassifier(random_state=42, n_jobs=4)),
    ('lgb', lgb.LGBMClassifier(max_depth=50, n_estimators=50, random_state=42)),
    ('xgb', xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=10, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1,
              n_estimators=50, n_jobs=0, num_parallel_tree=1, random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None))
]
clf_stack = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(C=1, random_state=42, solver='newton-cg')
)
from sklearn.ensemble import StackingClassifier
estimators = [
    ('lr',LogisticRegression(C=1, random_state=42, solver='newton-cg')),
    ('extra', ExtraTreesClassifier(random_state=42, n_jobs=4)),
    ('random',RandomForestClassifier(random_state=42, n_jobs=4)),
    ('lgb', lgb.LGBMClassifier(max_depth=50, n_estimators=50, random_state=42, n_jobs=4)),
]
clf_stack1 = StackingClassifier(
    estimators=estimators, final_estimator=LinearSVC(C=0.1, random_state=42)
)

from sklearn.ensemble import StackingClassifier
estimators = [
    ('lr',LogisticRegression(C=1, random_state=42, solver='newton-cg')),
    ('extra', ExtraTreesClassifier(random_state=42, n_jobs=4)),
    ('random',RandomForestClassifier(random_state=42, n_jobs=4)),
    ('lgb', lgb.LGBMClassifier(max_depth=50, n_estimators=50, random_state=42, n_jobs=4)),
    ('xgb', xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=10, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1,
              n_estimators=50, n_jobs=4, num_parallel_tree=1, random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None))
]
clf_stack2 = StackingClassifier(
    estimators=estimators, final_estimator=LinearSVC(C=0.1, random_state=42)
)

from sklearn.ensemble import StackingClassifier
estimators = [
    ('lr',LogisticRegression(C=1, random_state=42, solver='newton-cg')),
    ('extra', ExtraTreesClassifier(random_state=42, n_jobs=4)),
    ('lgb', lgb.LGBMClassifier(max_depth=50, n_estimators=50, random_state=42, n_jobs=4)),
    ('xgb', xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=10, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1,
              n_estimators=50, n_jobs=4, num_parallel_tree=1, random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)),
    ('svc',LinearSVC(C=0.1, random_state=42))
]
clf_stack3 = StackingClassifier(
    estimators=estimators, final_estimator=RandomForestClassifier(random_state=42, n_jobs=4)
)
from tqdm import tqdm
scores = {}

lg = LogisticRegression(C=1, random_state=42, solver='newton-cg')
list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(lg, X, df_train['target'], cv=a)
    list_.append(score.mean())
scores['Logistic Regression'] = list_

linear = LinearSVC(C=0.1, random_state=42)
list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(linear, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['Linear SVC'] = list_

random = RandomForestClassifier(random_state=42, n_jobs=4)
list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(random, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['Random Forest'] = list_

extr = ExtraTreesClassifier(random_state=42,  n_jobs=4)
list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(extr, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['Extra Tree'] = list_
    
lgbm = lgb.LGBMClassifier(max_depth=50, n_estimators=50, random_state=42,  n_jobs=4)
list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(lgbm, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['LGBM'] = list_
xgb = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=10, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1,
              n_estimators=50, n_jobs=4, num_parallel_tree=1, random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(xgb, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['XGBoost'] = list_
list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(clf_stack, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['Stack1'] = list_

list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(clf_stack1, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['Stack2'] = list_
    
list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(clf_stack2, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['Stack3'] = list_

list_ = []
for a in tqdm(range(2,9, 2)):
    score = cross_val_score(clf_stack3, X, df_train['target'], cv=a)
    list_.append(score.mean())

scores['Stack4'] = list_
import plotly.graph_objects as go
def plot_scores(scores):
    fig = go.Figure()
    for key, values in zip(scores.keys(), scores.values()):
        fig.add_trace(go.Scatter(y=values, x=[2,4,6,8],
                        mode='lines',
                        name='scores '+str(key)))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Results CV',
    })
    fig.show()
plot_scores(scores)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

stds = []
for value in scores.values():
    std = np.std(np.array(value), axis=0)
    stds.append(std)
df_to_scatter = pd.DataFrame([])
df_to_scatter['scores'] = stds
df_to_scatter['models'] = scores.keys()

fig = px.scatter(df_to_scatter, x="models", y="scores", color="models",
                 size='scores')
fig.show()
X_test = vec.transform(df_test['text'])
sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
model = clf_stack1.fit(X, df_train['target'])
predict = clf_stack1.predict(X_test)
sub['target'] = predict
sub.to_csv('submission.csv', index=False)
