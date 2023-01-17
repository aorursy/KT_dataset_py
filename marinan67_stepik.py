# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#%matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
events_data = pd.read_csv('../input/event-data-train/event_data_train.csv')
submission_data = pd.read_csv('../input/submission-data-train/submissions_data_train.csv')
events_data.head(10)
submission_data.head(10)
events_data['action'].unique()
events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
submission_data['date'] = pd.to_datetime(submission_data.timestamp, unit='s')
events_data.head(10)
submission_data.head(10)
wrong = submission_data[submission_data.submission_status == 'wrong']
wrong_user = wrong.groupby('step_id', as_index=False).agg({'user_id':'sum'})
df_sorted = wrong_user.sort_values("user_id", ascending=False)
df_sorted
events_data.dtypes
events_data['day'] = events_data.date.dt.date
submission_data['day'] = submission_data.date.dt.date
events_data.head(10)
submission_data.head(10)
events_data.groupby('day').user_id.nunique().plot(figsize=(15,8))
df = events_data.groupby('day', as_index=False).aggregate({'user_id': 'nunique'}).rename(columns = {'user_id': 'user'})
df
plt.close ()
df.plot(figsize=(15,8)) 
print(df['user'].dtypes)
print(df['day'].dtypes)
events_data.action.hist()
events_data[events_data.action == 'passed'].groupby('user_id', as_index=False).agg({'step_id':'count'}).rename(columns={'step_id':'passed_steps'}).passed_steps.hist()
users_events_data = events_data.pivot_table(index='user_id', columns='action', values='step_id', aggfunc='count', fill_value=0).reset_index()
users_events_data
users_scores = submission_data.pivot_table(index='user_id', columns='submission_status', values='step_id', aggfunc='count', fill_value=0).reset_index()
users_scores.head(10)
res1 = users_scores.loc[[(users_scores['correct']).idxmax()]]
res1
subset_1 = users_scores[(users_scores.correct > 400)]
subset_1
users_events_data.hist(figsize=(10, 6))
events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']).groupby('user_id')['timestamp'].apply(list).head()
events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']).groupby('user_id')['timestamp'].apply(list).apply(np.diff).head()
gap_data = events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']).groupby('user_id')['timestamp'].apply(list).apply(np.diff).values
gap_data = pd.Series(np.concatenate(gap_data, axis=0))
gap_data = gap_data / (24 * 60 * 60)
gap_data
gap_data[gap_data < 200].hist()
gap_data.quantile(0.95)
users_data = events_data.groupby('user_id', as_index=False).agg({'timestamp':'max'}).rename(columns={'timestamp':'last_timestamp'})
now = events_data.timestamp.max()
drop_out_threshold = 30 * 24 * 60 * 60
print(now, drop_out_threshold)
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold
users_data.head()
users_data = users_data.merge(users_scores, how='outer')
users_data
users_data = users_data.fillna(0)
users_data
users_data = users_data.merge(users_events_data, how='outer')
users_data
users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index()
users_days
users_data = users_data.merge(users_days, how='outer')
users_data
print(users_data.user_id.nunique(), events_data.user_id.nunique(), sep=(' '))
users_data['passed_course'] = users_data.passed > 170
users_data
users_data.groupby('passed_course').count()
100 * 1425 / 17809
users_data[(users_data.passed_course ==True) &  (users_data.is_gone_user ==False)]
from sklearn import tree
import math
E_sh_sob=(1/1)*math.log2((1/1)) - 0
E_sh_kot=-(4/9)*math.log2((4/9)) - (5/9)*math.log2((5/9))
E_gav_sob=0 - (5/5)*math.log2((5/5))
E_gav_kot=-(4/5)*math.log2((4/5)) - (1/5)*math.log2((1/5))
E_laz_sob=0 - (6/6)*math.log2((6/6))
E_laz_kot=-(4/4)*math.log2((4/4)) - 0
print('E_sh_sob=', E_sh_sob, 'E_sh_kot=', E_sh_kot, 'E_gav_sob=', E_gav_sob, 'E_gav_kot=', E_gav_kot, 'E_laz_sob=', E_laz_sob, 'E_laz_kot=', E_laz_kot, sep=' ')
E = -(4/10)*math.log2((4/10)) - (6/10)*math.log2((6/10))

IG_sh = round((E - (1/10)*E_sh_sob - (9/10)*E_sh_kot), 2)
IG_gav = round((E - (5/10)*E_gav_sob - (5/10)*E_gav_kot),2)
IG_laz = round((E - (6/10)*E_laz_sob - (6/10)*E_laz_kot),2)
print(IG_sh, IG_gav, IG_laz, sep=(' '))
titanic_data = pd.read_csv('../input/titanic/train.csv')
titanic_data.head()
titanic_data.isnull().sum()
Xt = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
yt = titanic_data['Survived']
Xt.head()
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
Xt = pd.get_dummies(Xt)
Xt.head()
Xt.shape
Xt = Xt.fillna({'Age': Xt.Age.median()})
from sklearn.model_selection import train_test_split
X_traint, X_testt, y_traint, y_testt = train_test_split(Xt, yt, test_size=0.33, random_state=42)
clf.fit(Xt, yt)
clf.fit(X_traint, y_traint)
clf.score(Xt, yt)
clf.score(X_traint, y_traint)
clf.score(X_testt, y_testt)
scores_data = pd.DataFrame()
from sklearn.model_selection import cross_val_score
for max_depth in range(1, 100):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_traint, y_traint)
    train_score = clf.score(X_traint, y_traint)
    test_score = clf.score(X_testt, y_testt)
    mean_cross_val_score = cross_val_score(clf, X_traint, y_traint, cv=5).mean()
    temp_score_data = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score], 'cross_val_score':[mean_cross_val_score]})
    scores_data = scores_data.append(temp_score_data)
scores_data.head()
scores_data_long = pd.melt(scores_data, id_vars =['max_depth'], value_vars =['train_score', 'test_score', 'cross_val_score'], var_name='set_type', value_name='score')
scores_data_long.head()
sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
cross_val_score(clf, X_traint, y_traint, cv=5)
cross_val_score(clf, X_traint, y_traint, cv=5).mean()
train_iris = pd.read_csv('../input/train-iris/train_iris.csv')
test_iris = pd.read_csv('../input/test-iris/test_iris.csv')
train_iris
test_iris.head()
X_iris = train_iris.drop(['Unnamed: 0', 'species'], axis=1)
y_iris = train_iris['species']
X_test_iris = test_iris.drop(['Unnamed: 0', 'species'], axis=1)
y_test_iris = test_iris['species']
from sklearn.metrics import accuracy_score
scores_data_iris = pd.DataFrame()
rc =np.random.seed(0)
for max_depth in range(1, 100):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=rc)
    clf.fit(X_iris, y_iris)
    train_score = clf.score(X_iris, y_iris)
    test_score = clf.score(X_test_iris, y_test_iris)
    mean_cross_val_score = cross_val_score(clf, X_iris, y_iris, cv=5).mean()
    temp_score_data = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score], 'cross_val_score':[mean_cross_val_score]})
    scores_data_iris = scores_data_iris.append(temp_score_data)
scores_data_iris
scores_data_long_iris = pd.melt(scores_data_iris, id_vars =['max_depth'], value_vars =['train_score', 'test_score', 'cross_val_score'], var_name='set_type', value_name='score')
scores_data_long_iris
sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long_iris)
dogs_cats =pd.read_csv('../input/dogs-cats/dogs_n_cats.csv')
dogs_cats
dogs_cats.Шерстист.unique()
X_dogs_cats = dogs_cats.drop(['Вид'], axis=1)
y_dogs_cats = dogs_cats['Вид']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dogs_cats, y_dogs_cats, test_size=0.33, random_state=42)
scores_data_dogs_cats = pd.DataFrame()
rc =np.random.seed(0)
for max_depth in range(1, 5):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=rc)
    clf.fit(X_train_d, y_train_d)
    train_score = clf.score(X_train_d, y_train_d)
    test_score = clf.score(X_test_d, y_test_d)
    mean_cross_val_score = cross_val_score(clf, X_train_d, y_train_d, cv=5).mean()
    temp_score_data = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score], 'cross_val_score':[mean_cross_val_score]})
    scores_data_dogs_cats = scores_data_dogs_cats.append(temp_score_data)
scores_data_dogs_cats
scores_data_long_dogs_cats = pd.melt(scores_data_dogs_cats, id_vars =['max_depth'], value_vars =['train_score', 'test_score', 'cross_val_score'], var_name='set_type', value_name='score')
sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long_dogs_cats)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=rc)
clf.fit(X_dogs_cats, y_dogs_cats)
data = pd.read_json('/kaggle/input/new-data/dataset_209691_15.txt')
data
rez = clf.predict(data)
rez = list(rez)
result = {i: rez.count(i) for i in rez}
result

data = pd.read_json('../input/12345678/dataset_209691_15.txt')
rez = clf.predict(data)
rez = list(rez)
result = {i: rez.count(i) for i in rez}
result
song= pd.read_csv('../input/songs-data/songs.csv')
song
X_song = song.drop(['artist', 'song', 'year'], axis=1)
y_song = song['artist']
X_song
import re
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub(r"\d+", "", text, flags=re.UNICODE)
    return text.strip()
X_song['lyrics'] = [preprocess_text(t) for t in X_song['lyrics']]
X_song['genre'] = pd.get_dummies(X_song['genre'])
from nltk.stem.snowball import SnowballStemmer
englishStemmer=SnowballStemmer("english")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import sent_tokenize, word_tokenize
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(englishStemmer.stem(word))
        stem_sentence.append(" ")
    stem_sentence = [x for x in stem_sentence if x not in stop_words and len(x)>=3]
    return " ".join(stem_sentence)
X_song['lyrics'] = [stemSentence(t) for t in X_song['lyrics']]
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer()
x = v.fit_transform(X_song['lyrics'])
df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
res = pd.concat([X_song, df1], axis=1)
X_song_data = res.drop(['lyrics'], axis=1)
X_song_data
x_trains, x_tests,y_trains, y_tests = train_test_split(X_song_data, y_song,train_size=0.75,random_state=1)
print(x_trains.shape[0],'x_train samples')
print(len(y_trains))
print(x_trains.shape)
print(x_tests.shape[0],'x_test samples')
print(len(y_tests))
print(x_tests.shape)
scores_data_song = pd.DataFrame()
rc =np.random.seed(0)
for max_depth in range(2, 10):
    clf = tree.DecisionTreeClassifier(criterion='entropy', \
                max_depth=max_depth, random_state=rc, min_samples_leaf=3)
    clf.fit(x_trains, y_trains)
    train_score = clf.score(x_trains, y_trains)
    test_score = clf.score(x_tests, y_tests)
    mean_cross_val_score = cross_val_score(clf, x_trains, y_trains, cv=5).mean()
    temp_score_data = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score], 'cross_val_score':[mean_cross_val_score]})
    scores_data_song = scores_data_song.append(temp_score_data)
scores_data_long_song = pd.melt(scores_data_song, id_vars =['max_depth'], value_vars =['train_score', 'test_score', 'cross_val_score'], var_name='set_type', value_name='score')
sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long_song)
mod = tree.DecisionTreeClassifier(criterion='entropy', \
                max_depth=6, random_state=rc, min_samples_leaf=3)
mod.fit(x_trains, y_trains)
predictions = mod.predict(x_tests)
from sklearn.metrics import precision_score, recall_score
precision_score(y_tests, predictions, average='micro')
'''mod = DecisionTreeClassifier(criterion='entropy', \
                max_depth=10, random_state=np.random.seed(0), min_samples_leaf=3)
mod.fit(X_train, y_train)
predictions = mod.predict(X_test)
precision = precision_score(y_test, predictions, average='micro')'''
from sklearn.model_selection import GridSearchCV
clf = tree.DecisionTreeClassifier()
clf
parametrs = {'criterion':['entropy', 'gini'], 'max_depth':range(1,30)}
grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5)
grid_search_cv_clf.get_params()
grid_search_cv_clf.fit(X_traint, y_traint)
grid_search_cv_clf.best_params_
best_clf = grid_search_cv_clf.best_estimator_
best_clf
y_pred = best_clf.predict(X_testt)
best_clf.score(X_testt, y_testt)
print(precision_score(y_testt, y_pred, average='macro'))
print(recall_score(y_testt, y_pred, average='macro'))
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
y_predicted_prob =  best_clf.predict_proba(X_testt)
fpr, tpr, thresholds = roc_curve(y_testt, y_predicted_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label = 'ROC curve (area=\0.2f)'% (roc_auc))
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
train_data_tree = pd.read_csv('../input/train-data-tree/train_data_tree.csv')
train_data_tree
X_tree = train_data_tree.drop(['num'], axis=1)
y_tree = train_data_tree['num']
mod = tree.DecisionTreeClassifier(criterion='entropy')
param = {'min_samples_split': range(2,5), 'max_depth':range(2,30)}
grid_search_cv_tree = GridSearchCV(mod, param, cv=5)
grid_search_cv_tree.get_params()
grid_search_cv_tree.fit(X_tree, y_tree)
#grid_search_cv_tree.best_params_
best_tree = grid_search_cv_tree.best_estimator_
best_tree
mod1=tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
mod1.fit(X_tree, y_tree)
mod1.score(X_tree, y_tree)
tree.plot_tree(mod1, filled=True)
mod1.tree_.n_node_samples
mod1.tree_.impurity
IG = 0.996 - (157*0.903 + 81*0.826)/238
IG
users_data
events_data.head()
user_min_time = events_data.groupby('user_id', as_index=False).agg({'timestamp':'min'}).rename({'timestamp':'min_timestamp'}, axis=1)
user_min_time
users_data = users_data.merge(user_min_time, how='outer')
users_data
events_data['user_time'] = events_data.user_id.map(str) + '_' + events_data.timestamp.map(str)
events_data.head()
learning_time_htreshold = 3*24*60*60
learning_time_htreshold
user_learning_time_htreshold = user_min_time.user_id.map(str) + '_' + (user_min_time.min_timestamp + learning_time_htreshold).map(str)
user_learning_time_htreshold.head()
user_min_time['user_learning_time_htreshold'] = user_learning_time_htreshold
events_data = events_data.merge(user_min_time[['user_id', 'user_learning_time_htreshold']], how='outer')
events_data 
event_data_train = events_data[events_data.user_time <= events_data.user_learning_time_htreshold]
event_data_train
event_data_train.groupby('user_id').day.nunique().max()
submission_data['user_time'] = submission_data.user_id.map(str) + '_' + submission_data.timestamp.map(str)
submission_data = submission_data.merge(user_min_time[['user_id', 'user_learning_time_htreshold']], how='outer')
submission_data_train = submission_data[submission_data.user_time <= submission_data.user_learning_time_htreshold]
submission_data_train.groupby('user_id').day.nunique().max()
X = submission_data_train.groupby('user_id').day.nunique().to_frame().rename(columns={'day':'days'}).reset_index()
X
steps_tried = submission_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index().rename(columns={'step_id':'steps_tried'})
steps_tried
X = X.merge(steps_tried, on='user_id', how='outer')
X
submission_data_train.pivot_table(index='user_id', columns='submission_status', values='step_id', aggfunc='count', fill_value=0).reset_index().head()
X = X.merge(submission_data_train.pivot_table(index='user_id', columns='submission_status', values='step_id', aggfunc='count', fill_value=0).reset_index())
X['correct_ratio'] = X.correct / (X.correct + X.wrong)
X
X = X.merge(event_data_train.pivot_table(index='user_id', columns='action', values='step_id', aggfunc='count', fill_value=0).reset_index()[['user_id', 'viewed']], how='outer')
X = X.fillna(0)
X
X = X.merge(users_data[['user_id', 'passed_course', 'is_gone_user']], how='outer')
X
X = X[~((X.is_gone_user == False) & (X.passed_course == False))]
X
y = X.passed_course.astype(int)
y
X.groupby(['passed_course', 'is_gone_user']).user_id.count()
X = X.drop(['passed_course', 'is_gone_user'],axis=1)
X = X.set_index(X.user_id)
X = X.drop('user_id', axis=1)
X
X.hist(figsize=(10,6))
mod2 = tree.DecisionTreeClassifier(criterion='entropy')
params = {'min_samples_split': range(2,5), 'max_depth':range(2,50)}
grid_search_cv_user = GridSearchCV(mod2, params, cv=5)
grid_search_cv_user.fit(X, y)
best_user = grid_search_cv_user.best_estimator_
best_user
best_user.fit(X, y)
best_user.score(X, y)
y.hist()
heart = pd.read_csv('../input/heart-disease-uci/heart.csv')
heart
from sklearn.ensemble import RandomForestClassifier
X_heart = heart.drop(['target'], axis=1)
y_heart = heart['target']

rf = RandomForestClassifier(10, max_depth=5, random_state =np.random.seed(0))
rf.fit(X_heart, y_heart)
imp = pd.DataFrame(rf.feature_importances_, index=X_heart.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
mush = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
mush
'''Encoding categorical data¶
Label encoding

from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder() 
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)'''
train_mush = pd.read_csv('../input/train-mush/training_mush.csv.crdownload')
train_mush
train_mush = train_mush.dropna()
X_mush = train_mush.drop('class', axis=1)
y_mush = train_mush['class']
y_mush

X_mush
X_mush.shape
pd.isnull(X_mush)
r = RandomForestClassifier(random_state=0)
params_r = {'n_estimators': range(10,50,10),'min_samples_split': range(2,9,2), 'max_depth':range(1,12,2), 'min_samples_leaf': range(1,7)}
grid_search_cv_r = GridSearchCV(r, params_r, cv=3, n_jobs=-1)
grid_search_cv_r.fit(X_mush, y_mush)
par = grid_search_cv_r.best_params_
best_r = grid_search_cv_r.best_estimator_
best_r
grid_search_cv_r.best_params_
imp_r = pd.DataFrame(best_r.feature_importances_, index=X_mush.columns, columns=['importance'])
imp_r.sort_values(by='importance', ascending=False)
test_mush = pd.read_csv('../input/test-mush/testing_mush.csv.crdownload')
test_mush
#testing = pd.read_csv('../input/testing/testing_y_mush.csv.zip')
test_mush.isnull().sum().sum()
test_mush = test_mush.fillna(0)
best_r.fit(X_mush, y_mush)
preds_mush = best_r.predict(test_mush)
rez_mush = pd.Series(preds_mush).value_counts()
rez_mush
invasion = pd.read_csv('../input/invasion/invasion.csv')
invasion
invasion.describe()
invasion.isnull().sum().sum()
X_invasion = invasion.drop('class', axis=1)
y_invasion = invasion['class']
inv = RandomForestClassifier(random_state=0)
params_inv = {'n_estimators': range(10,150,10),'min_samples_split': range(2,9,2), 'max_depth':range(1,20,2), 'min_samples_leaf': range(1,7)}
grid_search_cv_inv = GridSearchCV(inv, params_inv, cv=5, n_jobs=-1)
grid_search_cv_inv.fit(X_invasion, y_invasion)
grid_search_cv_inv.best_params_
best_inv = grid_search_cv_inv.best_estimator_
best_inv
grid_search_cv_inv.best_params_
best_inv = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_leaf=1, min_samples_split= 2, random_state=0)
imp_inv = pd.DataFrame(best_inv.feature_importances_, index=X_invasion.columns, columns=['importance'])
imp_inv.sort_values(by='importance', ascending=False)
best_inv.fit(X_invasion, y_invasion)
%%timeit
df = pd.DataFrame(range(10000000))
%time df.apply(np.mean)

%time df.apply('mean')

%time df.describe().loc['mean']

%time df.mean(axis=0)




