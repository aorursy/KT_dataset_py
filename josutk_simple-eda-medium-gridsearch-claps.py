# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/medium-articles-dataset/medium_data.csv')
df.head()
import seaborn as sns
sns.distplot(df['claps'])
df.isna().sum()
import plotly.express as px
fig = px.bar(x=list(df['publication'].value_counts().index), y=list(df['publication'].value_counts()))
fig.show()
import plotly.express as px
fig = px.bar(x=list(df.isna().sum().index), y=list(df.isna().sum()), title='Nan values')
fig.show()
import plotly.express as px
fig = px.box(df,x='publication', y='claps', title='Box plot claps')
fig.show()
import plotly.express as px
fig = px.scatter(df,y='claps', x='reading_time', title='Read time vs Claps', color='publication')
fig.show()
fig = px.scatter(df,y='claps', x='responses', title='Claps vs Response', color='publication')
fig.show()
fig = px.scatter(df,y='responses', x='reading_time', title='Response vs Read time', color='publication')
fig.show()
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_scatter(df):
    fig = make_subplots(rows=len(df['publication'].unique()), cols=3, specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                                                                            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                                                                            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                                                                            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                                                                            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                                                                            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                                                                            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]])
    i=1
    for public in df['publication'].unique():
        aux = df[df['publication'] == public]
        fig.append_trace(go.Scatter(x=aux['claps'], y=aux['responses'], name=str(public)+" claps vs " + str(public)+ " responses",  mode='markers'), row=i, col=1)
        fig.append_trace(go.Scatter(x=aux['claps'], y=aux['reading_time'], name=str(public)+" claps vs " + str(public)+ " reading_time",  mode='markers'), row=i, col=2)
        fig.append_trace(go.Scatter(x=aux['reading_time'], y=aux['responses'], name=str(public)+" reading_time" + " vs " + str(public)+ " responses",  mode='markers'), row=i, col=3)
        i+=1
    fig.show()
plot_scatter(df)
from sklearn.feature_extraction.text import CountVectorizer

import nltk
import re
from tqdm import tqdm 
tqdm.pandas()
def get_top_n_words(corpus, n=None, vocabulary=None):
    vec = CountVectorizer(vocabulary=vocabulary).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def preprocess(df):
    stopwords = nltk.corpus.stopwords.words('english')
    df['title_process'] = df['title'].astype(str)
    df['title_process'] = df['title_process'].progress_apply(lambda x : x.lower())
    df['title_process'] = df['title_process'].progress_apply(lambda x : nltk.word_tokenize(x))
    df['title_process'] = df['title_process'].progress_apply(lambda x : [item for item in x if item not in stopwords])
    df['title_process'] = df['title_process'].progress_apply(lambda x : " ".join(x))
    df['title_process'] = df['title_process'].str.replace('@[^\s]+', "")
    df['title_process'] = df['title_process'].str.replace('https?:\/\/.*[\r\n]*', '')
    df['title_process'] = df['title_process'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df['title_process'] = df['title_process'].str.replace('\d+', '')
    df['title_process'] = df['title_process'].str.replace('[^\w\s]', '')
    return df

df = preprocess(df)
list_ = get_top_n_words(df['title_process'], n=20)
new_list_words = [ seq[0] for seq in list_ ]
new_list_values = [ seq[1] for seq in list_ ]
fig = px.bar(y=new_list_words, x=new_list_values, title='Real news Frequency words',  orientation='h')
fig.show()
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['title_process'])
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

random = RandomForestRegressor()
random.fit(X, df['claps'])

import plotly.express as px
dict_ = dict(zip(vectorizer.get_feature_names(), random.feature_importances_))
dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
dict_
fig = px.bar(x=list(dict_.values())[0:50], y=list(dict_.keys())[0:50], orientation='h')
fig.show()
from tqdm import tqdm
import plotly.graph_objects as go

def bar_plot(df):
        fig = make_subplots(rows=4, cols=2, specs=[[{"type": "bar"}, {"type": "bar"}],
                                                    [{"type": "bar"}, {"type": "bar"}],
                                                    [{"type": "bar"}, {"type": "bar"}],
                                                    [{"type": "bar"}, {"type": "bar"}]])
        i =1
        j =1
        for class_ in tqdm(df['publication'].unique()):
            aux = df[df['publication']==class_]
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(aux['title_process'])
            random = RandomForestRegressor()
            random.fit(X, aux['claps'])
            dict_ = dict(zip(vectorizer.get_feature_names(), random.feature_importances_))
            dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
            fig.append_trace(go.Bar(x=list(dict_.values())[0:50], y=list(dict_.keys())[0:50], orientation='h', name=class_), row=i, col=j)
            if j == 2:
                i+=1
                j=0
            j+=1
        fig.show()
            
bar_plot(df)
def parallel(df):
    for class_ in tqdm(df['publication'].unique()):
        aux = df[df['publication'] == class_]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(aux['title_process'])
        random = RandomForestRegressor()
        random.fit(X, aux['claps'])
        dict_ = dict(zip(vectorizer.get_feature_names(), random.feature_importances_))
        dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
        frame = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

        frame = frame[list(dict_.keys())[0:10]]
        fig = px.parallel_categories(frame, title=class_+" 10 important features correlated")
        fig.show()
parallel(df)
def contains_word(word, text):
    if str(word) in text:
        return True
    return False

def get_stds(df):
    mask = []
    stds = {}
    df['claps'] = df['claps'].astype(float)
    for class_ in tqdm(df['publication'].unique()):
        aux = df[df['publication'] == class_]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(aux['title_process'])
        random = RandomForestRegressor()
        random.fit(X, aux['claps'])
        dict_ = dict(zip(vectorizer.get_feature_names(), random.feature_importances_))
        dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
        for word in list(dict_.keys())[0:10]:
            for text in aux['title_process']:
                bool_ = contains_word(word, text)
                mask.append(bool_)
            select = aux[mask]
            if select.shape[0] == 1:
                a= list(select['claps'])
                stds[class_+' '+word]=(a[0], 1, a[0])
            else:
                stds[class_+' '+word]=(select['claps'].std(),select.shape[0], select['claps'].mean()) 
            mask = []
    return stds
stds = get_stds(df)
stds
list_1 = []
list_2 = []
list_3 = []
for a, b, c in list(stds.values()):
    list_1.append(a)
    list_2.append(b)
    list_3.append(c)
fig = px.scatter(x=list(stds.keys()), y=list_3, size=list_1)
fig.show()
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

lightgbm_params ={'learning_rate':[0.0001, 0.001, 0.003, 0.01, 0.1],
                  'n_estimators':[3,5,10,20, 50, 100],
                 'max_depth':[4, 6, 10, 15, 20, 50]}
gbm = lgb.LGBMRegressor(random_state = 42)
clf_gbm = GridSearchCV(gbm, lightgbm_params, cv=4, n_jobs= 4, verbose = 1)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['title_process'])
clf_gbm.fit(X, df['claps'])
print(clf_gbm.best_estimator_)
print(clf_gbm.best_score_)
from sklearn.ensemble import AdaBoostRegressor
adam_boosting_params = {'learning_rate':[0.0001, 0.001, 0.003, 0.01, 0.1,1],
                        'n_estimators':[10,20, 50, 100]}
ada = AdaBoostRegressor(random_state=42)
clf_ada = GridSearchCV(ada, adam_boosting_params, cv=4,n_jobs= 4, verbose = 1)
clf_ada.fit(X, df['claps'])
print(clf_ada.best_estimator_)
print(clf_ada.best_score_)
from sklearn.ensemble import RandomForestRegressor
lightgbm_params ={'n_estimators':[3,5,10,20, 50, 100],
                 'max_depth':[4, 6, 10, 15, 20, 50]}

random = RandomForestRegressor(random_state = 42)
clf_random = GridSearchCV(random, lightgbm_params, cv=4, n_jobs= 4, verbose = 1)
clf_random.fit(X, df['claps'])
print(clf_random.best_estimator_)
print(clf_random.best_score_)
from sklearn.linear_model import Ridge
ridge_params = {'alpha': [0.0001,0.001, 0.01, 1, 0.1, 10, 100, 1000, 10000, 100000, 1000000]}

rid = Ridge(random_state=42)
clf_ada = GridSearchCV(rid, ridge_params, cv=4,  scoring='neg_mean_squared_error',n_jobs= 4, verbose = 1)
clf_ada.fit(X, df['claps'])
print(clf_ada.best_estimator_)
print(clf_ada.best_score_)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(
    X,  df['claps'], random_state=42
)
scores ={}
clf = lgb.LGBMRegressor(learning_rate=0.01, max_depth=50, n_estimators=20, random_state=42)
model = clf.fit(X_train, y_train)
pred = model.predict(X_test)
score = np.sqrt(mean_squared_error(y_test, pred))
scores['lgb'] = score
import plotly.graph_objects as go

def plot_predict(pred, true):
    indexs = []
    for i in range(len(pred)):
        indexs.append(i)
        

    fig = go.Figure()

    fig.add_trace(go.Line(
        x=indexs,
        y=pred,
        name="Predict"
    ))

    fig.add_trace(go.Line(
        x=indexs,
        y=true,
        name="Test"
    ))

    fig.show()
plot_predict(pred, y_test)
clf = AdaBoostRegressor(learning_rate=0.01, n_estimators=10, random_state=42)
model = clf.fit(X_train, y_train)
pred = model.predict(X_test)
score = np.sqrt(mean_squared_error(y_test, pred))
scores['ada'] = score
plot_predict(pred, y_test)
clf = Ridge(alpha=10, random_state=42)
model = clf.fit(X_train, y_train)
pred = model.predict(X_test)
score = np.sqrt(mean_squared_error(y_test, pred))
scores['ridge'] = score
plot_predict(pred, y_test)
clf = RandomForestRegressor(max_depth=4, random_state=42)
model = clf.fit(X_train, y_train)
pred = model.predict(X_test)
score = np.sqrt(mean_squared_error(y_test, pred))
scores['rf'] = score
plot_predict(pred, y_test)
result = pd.DataFrame([])
result['model'] = list(scores.keys())
result['score'] = list(scores.values())
result = result.sort_values(['score'])
result.head(10)
