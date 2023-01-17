# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



# Standard plotly imports

#import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

import plotly.figure_factory as ff



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

#cufflinks.go_offline(connected=True)



# Preprocessing, modelling and evaluating

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial



import os

import gc



import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

plt.style.use('fivethirtyeight')



plt.figure(figsize=(25,25))



import pandas_profiling as pp



%reload_ext autoreload

%autoreload 2

%matplotlib inline



!pip install fastai==1.0.57

import fastai



from fastai import *

from fastai.tabular import *



# from torchvision.models import *

# import pretrainedmodels



# from utils import *

import sys



from fastai.callbacks.hooks import *



from fastai.callbacks.tracker import EarlyStoppingCallback

from fastai.callbacks.tracker import SaveModelCallback



import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from sklearn.manifold import TSNE



from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingClassifier



import gc

from datetime import datetime 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



!pip3 install catboost

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier

from sklearn import svm

import lightgbm as lgb

from lightgbm import LGBMClassifier

import xgboost as xgb

from xgboost.sklearn import XGBClassifier  

from xgboost.sklearn import XGBRegressor



from scipy.special import erfinv

import matplotlib.pyplot as plt

import torch

from torch.utils.data import *

from torch.optim import *

from fastai.tabular import *

import torch.utils.data as Data

from fastai.basics import *

from fastai.callbacks.hooks import *

from tqdm import tqdm_notebook as tqdm



from hyperopt import hp, tpe

from hyperopt.fmin import fmin

from hyperopt import STATUS_OK



from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import make_scorer



import warnings

warnings.filterwarnings('ignore')



sns.set_style('whitegrid')
df1=pd.read_csv('../input/the-movies-dataset/credits.csv')

df2=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')

df3=pd.read_csv('../input/the-movies-dataset/keywords.csv')
df1.head(3)
df2.head(3)
df3.head(3)
df2 = df2[df2.id!='1997-08-20']

df2 = df2[df2.id!='2012-09-29']

df2 = df2[df2.id!='2014-01-01']
df2['id'] = df2['id'].astype(int)
df2 = df2[df2['original_language']=='en']

df2.shape
df2=df2.merge(df1, on='id')

df2=df2.merge(df3, on='id')
df2.shape
df2 = df2.dropna(subset=['budget','revenue', 'poster_path', 'genres'], axis=0)

df2.shape
df2 = df2[(df2[['budget','revenue', 'poster_path', 'genres']] != 0).all(axis=1)]

df2.shape
df2.head(3)
C= df2['vote_average'].mean()

C
m= df2['vote_count'].quantile(0.9)

m
q_movies = df2.copy().loc[df2['vote_count'] >= m]

q_movies.shape
# IMDB formula

def weighted_rating(x, m=m, C=C):

    v = x['vote_count']

    R = x['vote_average']

    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above

q_movies = q_movies.sort_values('score', ascending=False)



#Print the top 15 movies

q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g = sns.barplot(x='title', y = 'score', data=q_movies.head(50))

g.set_title("Movies and Scores", fontsize=22)

g.set_xlabel("Movies", fontsize=18)

g.set_ylabel('Score', fontsize=18)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y='score',  data=q_movies.head(50))

plt.legend(title='Movies and Score', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y='runtime', data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("runtime", fontsize=17)

g1.set_title("Movie Score Runtime Wise", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("Score", fontsize=17)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y='score',  data=q_movies.head(50))

plt.legend(title='Movies and Score', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y=q_movies['budget'].astype(float), data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("Budget", fontsize=17)

g1.set_title("Movie Score Budget Wise", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("Score", fontsize=17)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y='score',  data=q_movies.head(50))

plt.legend(title='Movies and Score', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y=q_movies['revenue'].astype(float), data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("Revenue", fontsize=17)

g1.set_title("Movie Score Revenue Wise", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("Score", fontsize=17)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y=q_movies['revenue'].astype(float),  data=q_movies.head(50))

plt.legend(title='Movies Revenues vs Budget', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y=q_movies['budget'].astype(float), data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("budget", fontsize=17)

g1.set_title("Movie Revenue vs Budget", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("revenue", fontsize=17)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y=q_movies['score'].astype(float),  data=q_movies.head(50))

plt.legend(title='Movies Score vs Popularity', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y=q_movies['popularity'].astype(float), data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("popularity", fontsize=17)

g1.set_title("Movie Score vs Popularity", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("Score", fontsize=17)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y=q_movies['score'].astype(float),  data=q_movies.head(50))

plt.legend(title='Movies Score vs Votes', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y=q_movies['vote_count'].astype(float), data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("votes", fontsize=17)

g1.set_title("Movie Score vs Votes", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("Score", fontsize=17)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y=q_movies['score'].astype(float),  data=q_movies.head(50))

plt.legend(title='Movies Score vs VoteAvg', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y=q_movies['vote_average'].astype(float), data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("votes", fontsize=17)

g1.set_title("Movie Score vs VoteAvg", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("Score", fontsize=17)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y=q_movies['vote_count'].astype(float),  data=q_movies.head(50))

plt.legend(title='Movies Vote count vs Vote avg', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y=q_movies['vote_average'].astype(float), data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("vote-avg", fontsize=17)

g1.set_title("Movie Votes / Count", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("vote-count", fontsize=17)

plt.show()
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

g1 = sns.barplot(x='title', y=q_movies['runtime'].astype(float),  data=q_movies.head(50))

plt.legend(title='Movies Runtime vs Budget', loc='best')

gt = g1.twinx()

gt = sns.pointplot(x='title', y=q_movies['budget'].astype(float), data=q_movies.head(50), color='black', legend=False, ci=70, scale=0.5)

gt.set_ylabel("budget", fontsize=17)

g1.set_title("Movie Runtime vs Budget", fontsize=19)

g1.set_xlabel("Movie", fontsize=17)

g1.set_ylabel("Runtime", fontsize=17)

plt.show()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

text = df2.overview.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
df_rating = pd.read_csv("../input/the-movies-dataset/ratings_small.csv", parse_dates=['timestamp'])

df_rating.head()
df_movie = df2[['id', 'title', 'genres']]

df_movie.rename(columns = {'id':"movieId"}, inplace=True)



print("df shape before removing NaN:".ljust(15), df_movie.shape)

df_movie.dropna(inplace=True)

print("df shape after removing NaN:".ljust(15), df_movie.shape)
df_movie.head()
R_df = df_rating.pivot(index = 'userId', columns='movieId', values='rating').fillna(0)

R_df.head()
R = R_df.as_matrix()

user_ratings_mean = np.mean(R, axis = 1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds

U, sigma, Vt = svds(R_demeaned, k = 50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
preds_df.head()
def recommend_movies(predictions_df, userId, movies_df, original_ratings_df, num_recommendations=5):

    

    # Get and sort the user's predictions

    user_row_number = userId - 1 # UserID starts at 1, not 0

    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    

    # Get the user's data and merge in the movie information.

    user_data = original_ratings_df[original_ratings_df.userId == (userId)]

    user_full = (user_data.merge(df_movie, how = 'left', left_on = 'movieId', right_on = 'movieId').

                     sort_values(['rating'], ascending=False)

                 )



    print ('User {0} has already rated {1} movies.'.format(userId, user_full.shape[0]))

    print ('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    

    # Recommend the highest predicted rating movies that the user hasn't seen yet.

    recommendations = (df_movie[~df_movie['movieId'].isin(user_full['movieId'])].

         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',

               left_on = 'movieId',

               right_on = 'movieId').

         rename(columns = {user_row_number: 'Predictions'}).

         sort_values('Predictions', ascending = False).

                       iloc[:num_recommendations, :-1]

                      )



    return user_full, recommendations



already_rated, predictions = recommend_movies(preds_df, 8, df_movie, df_rating, 5)
already_rated.dropna().head(5)
predictions
df2.overview[9]
print('dataset shape before removing NaN:', df2.shape)

df2.dropna(subset = ['overview', 'release_date', 'production_companies'], inplace=True)

print('dataset shape after removing NaN:', df2.shape)
df2.head(2)
from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres', 'production_companies']

for feature in features:

    df2[feature] = df2[feature].apply(literal_eval)
def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan
def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > 3:

            names = names[:3]

        return names



    #Return empty list in case of missing/malformed data

    return []
df2['director'] = df2['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres', 'production_companies']

for feature in features:

    df2[feature] = df2[feature].apply(get_list)
df2[['title', 'cast', 'director', 'keywords', 'genres', 'production_companies']].head(3)
def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i) for i in x]

    else:

        #Check if director exists. If not, return empty string

        if isinstance(x, str):

            return str.lower(x)

        else:

            return ''
# Apply clean_data function to your features.

features = ['cast', 'keywords', 'director', 'genres', 'production_companies']



for feature in features:

    df2[feature] = df2[feature].apply(clean_data)
def create_soup1(x):

    return ' '.join(x['keywords'])  + ' ' + ' '.join(x['genres'])



df2['soup1'] = df2.apply(create_soup1, axis=1)
def create_soup2(x):

    return  ' '.join(x['cast']) + ' ' + x['director']  + ' ' + ' '.join(x['production_companies'])



df2['soup2'] = df2.apply(create_soup2, axis=1)
df2.head(3)
df2[['title', 'overview', 'soup1', 'soup2']].head(4)
df2['score'] = df2.apply(weighted_rating, axis=1)

df2['score'].min(), df2['score'].max()
df2['score_rank'] = df2.score.apply(lambda x: "bad_movie" if x <=7 else "good_movie")

df2['vote_avg_rank'] = df2.vote_average.apply(lambda x: "low_average" if x <=7 else "high_average")
df2.score_rank.value_counts(normalize=True)
df2.vote_avg_rank.value_counts(normalize=True)
df2.loc[df2['score'].idxmax()]['title']
df2.loc[df2['score'].idxmin()]['title']
df2.sort_values(['score'],ascending=False)['title'][:50]
df2.sort_values(['vote_average'],ascending=False)['title'][:50]
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)



sns.distplot(df2['score'], hist=True, kde=True, 

             bins=int(180/5), color = 'blue',

             hist_kws={'edgecolor':'black'})
df2['budget'] = df2['budget'].astype(float)

df2['revenue'] = df2['revenue'].astype(float)

df2['runtime'] = df2['runtime'].astype(float)

df2['popularity'] = df2['popularity'].astype(float)

df2['score'] = df2['score'].astype(float)

df2['vote_average'] = df2['vote_average'].astype(float)

df2['vote_count'] = df2['vote_count'].astype(float)
df2['release_date'] = pd.to_datetime(df2['release_date'])
df2['hit_flop'] = np.where((df2['revenue']/df2['budget']) >1.0 , 'hit', 'flop')

                           

df2.hit_flop.value_counts()
a = pd.Series([item for sublist in df2.genres for item in sublist])

df_genres = a.groupby(a).size().rename_axis('genres').reset_index(name='f')
df_genres
df2.status.value_counts()
df2.head(2)
df_reduce = df2[['title', 'overview', 'soup1', 'soup2', 

                 'budget', 'revenue', 'popularity', 'runtime',

                 'adult', 'release_date', 'score', 'director', 'vote_count', 'status', 'original_language', 'score_rank', 'hit_flop']]
df_reduce.head(2)
%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.callbacks.tracker import *

from fastai.text import *

from fastai.tabular import *
data_lm = (TextList.from_df(df_reduce[['overview']])

                    .split_by_rand_pct(0.1)

                   .label_for_lm()

                   .databunch(bs=64)

          )
data_lm.show_batch()
data_lm.save('data_lm.pkl')
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.6).to_fp16()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, 5e-3, moms=(0.8,0.7))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(skip_end=5)
learn.fit_one_cycle(10, 1e-4, moms=(0.8,0.7))

learn.recorder.plot_losses()
learn.save_encoder('fine_tuned_enc')
TEXT = "A story of "

N_WORDS = 20

N_SENTENCES = 2



for temp in [0.1,0.5,1,1.5,2,2.5]: 

    print(learn.predict(TEXT, N_WORDS, temperature=temp))

    print('-'*10)
from fastai.callbacks import *

auroc = AUROC()
def get_val_idxs(train,n_splits=20):

    np.random.seed(42)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_idxs, valid_idxs = next(cv.split(train))

    return train_idxs,valid_idxs
ymin = 1

ymax = 9.5



class scaledSigmoid(nn.Module):

    def forward(self, input):

        return torch.sigmoid(input) * (ymax - ymin) + ymin
txt_cols=['overview', 'soup1', 'soup2', 'title', 'director']



train_idxs,val_idxs = get_val_idxs(df_reduce[['overview', 'soup1', 'soup2', 'title', 'director', 'hit_flop']], n_splits=20)

train_idxs,val_idxs

train_idxs.shape,val_idxs.shape



dep_var = 'hit_flop'



data_txt = (TextList.from_df(df_reduce[['overview', 'soup1', 'soup2', 'title', 'director', 'hit_flop']], 

                             cols = txt_cols, vocab=data_lm.vocab)

                            .split_by_idx(val_idxs)

                            .label_from_df(cols=dep_var) #,label_cls=FloatList)

                            .databunch(bs=64))
data_txt.show_batch()
learn = text_classifier_learner(data_txt,AWD_LSTM,metrics=[accuracy, auroc], 

                                    loss_func=LabelSmoothingCrossEntropy())

    

learn.load_encoder('fine_tuned_enc')

learn.model
# learn.model[1].add_module("sSig", module= scaledSigmoid())
learn.freeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.freeze()

learn.fit_one_cycle(2, 5e-2, moms=(0.8, 0.7))
learn.save('tmp1')

_=learn.load('tmp1')

learn.freeze_to(-2)



learn.fit_one_cycle(3, slice(1e-03/(2.6**4),1e-03), moms=(0.8, 0.7))
learn.save('tmp2')

_=learn.load('tmp2')

learn.freeze_to(-3)



learn.fit_one_cycle(3, slice(1e-04/(2.6**4),1e-04), moms=(0.8, 0.7))
learn.unfreeze()

learn.fit_one_cycle(4, slice(1e-05/(2.6**4),1e-05), moms=(0.8, 0.7))
val_df = df_reduce.loc[val_idxs].copy()

val_df.head(10)
one_item = val_df.loc[132]

one_item.overview
learn.predict(one_item)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
def get_val_idxs(train,n_splits=20):

    np.random.seed(42)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_idxs, valid_idxs = next(cv.split(train))

    return train_idxs,valid_idxs
df_reduce['year'] = df_reduce['release_date'].dt.year

df_reduce['month'] = df_reduce['release_date'].dt.month

df_reduce['day'] = df_reduce['release_date'].dt.day

df_reduce['weekday'] = df_reduce['release_date'].dt.weekday

df_reduce['half_year'] = df_reduce['month'].apply(lambda x: 'first_half' if x <=6 else 'second_half')
df_reduce.head(3)
df_g_budget = df_reduce.groupby(['year'], as_index=False)['budget'].mean()

df_g_budget = pd.DataFrame(df_g_budget)

df_g_budget.reset_index(drop=True, inplace=True)

df_g_budget.rename(columns={"budget": "budget_avg"}, inplace=True)



df_g_revenue = df_reduce.groupby(['year'], as_index=False)['revenue'].mean()

df_g_revenue = pd.DataFrame(df_g_revenue)

df_g_revenue.reset_index(drop=True, inplace=True)

df_g_revenue.rename(columns={"revenue": "revenue_avg"}, inplace=True)



df_g_revenue.head(3)
gc.collect()

df_reduce['budget_rev'] = df_reduce.year.map(df_g_budget.set_index('year').budget_avg)

df_reduce['revenue_rev'] = df_reduce.year.map(df_g_revenue.set_index('year').revenue_avg)
df_reduce['budget_2'] = np.where(df_reduce['budget']==0., df_reduce.budget_rev, df_reduce.budget)

df_reduce['revenue_2'] = np.where(df_reduce['revenue']==0., df_reduce.revenue_rev, df_reduce.revenue)
df_reduce.head(5)
df_reduce['budget'] = df_reduce['budget'].astype(float)

df_reduce['revenue'] = df_reduce['revenue'].astype(float)

df_reduce['budget_2'] = df_reduce['budget_2'].astype(float)

df_reduce['revenue_2'] = df_reduce['revenue_2'].astype(float)

df_reduce['budget_rev'] = df_reduce['budget_rev'].astype(float)

df_reduce['revenue_rev'] = df_reduce['revenue_rev'].astype(float)

df_reduce['runtime'] = df_reduce['runtime'].astype(float)

df_reduce['popularity'] = df_reduce['popularity'].astype(float)

df_reduce['score'] = df_reduce['score'].astype(float)

df_reduce['vote_count'] = df_reduce['vote_count'].astype(float)
df_reduce.shape
cat_names=['adult', 'director',  'status', 'original_language', 'year', 'month', 'day', 'weekday', 'half_year']

cont_names= [ 'budget_2', 'revenue_2', 'runtime' ]



print(f'# of continuous feas: {len(cont_names)}')

print(f'# of categorical feas: {len(cat_names)}')



dep_var = 'hit_flop'



procs = [FillMissing,Categorify, Normalize]



txt_cols=['overview', 'soup1', 'soup2', 'title']

print(txt_cols[0])



len(cat_names) + len(cont_names) + 4 + 1 + 1 + 4 + 4 == df_reduce.shape[1]
df_reduce.shape
train_idxs,val_idxs = get_val_idxs(df_reduce,n_splits=20)

train_idxs,val_idxs

train_idxs.shape,val_idxs.shape
def get_tabular_databunch(df_reduce,bs=100,val_idxs=val_idxs):

    return (TabularList.from_df(df_reduce, cat_names, cont_names, procs=procs)

                            .split_by_idx(val_idxs)

                            .label_from_df(cols=dep_var)#,label_cls=FloatList)

                            .databunch(bs=bs))
def get_text_databunch(df_reduce,bs=100,val_idxs=val_idxs):

    return (TextList.from_df(df_reduce, cols = txt_cols, vocab=data_lm.vocab)

                            .split_by_idx(val_idxs)

                            .label_from_df(cols=dep_var)#,label_cls=FloatList)

                            .databunch(bs=bs))
def get_tabular_learner(data,params,seed=42):

    

    return tabular_learner(data,metrics=[accuracy, auroc],loss_func=LabelSmoothingCrossEntropy(), **params)



def get_text_learner(data,params,seed=42):



    learn = text_classifier_learner(data,AWD_LSTM,metrics=[accuracy, auroc], 

                                    loss_func=LabelSmoothingCrossEntropy(),**params)

    

    learn.load_encoder('fine_tuned_enc') 

    return learn
from fastai.text import *

from fastai.tabular import *





class ConcatDataset(Dataset):

    def __init__(self, x1, x2, y): self.x1,self.x2,self.y = x1,x2,y

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return (self.x1[i], self.x2[i]), self.y[i]

    

def tabtext_collate(batch):

    x,y = list(zip(*batch))

    x1,x2 = list(zip(*x)) # x1 is (cat,cont), x2 is numericalized ids for text

    x1 = to_data(x1)

    x1 = list(zip(*x1))

    x1 = torch.stack(x1[0]), torch.stack(x1[1])

    x2, y = pad_collate(list(zip(x2, y)), pad_idx=1, pad_first=True)

    return (x1, x2), y



class ConcatModel(nn.Module):

    def __init__(self, mod_tab, mod_nlp, layers, drops): 

        super().__init__()

        self.mod_tab = mod_tab

        self.mod_nlp = mod_nlp

        lst_layers = []

        activs = [nn.ReLU(inplace=True),] * (len(layers)-2) + [None]

        for n_in,n_out,p,actn in zip(layers[:-1], layers[1:], drops, activs):

            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)

        self.layers = nn.Sequential(*lst_layers)



    def forward(self, *x):

        x_tab = self.mod_tab(*x[0])

        x_nlp = self.mod_nlp(x[1])[0]

        x = torch.cat([x_tab, x_nlp], dim=1)

        return self.layers(x)    







def get_tabtext_learner(data,tab_learner,text_learner,lin_layers,ps):

    tab_learner.model.layers = tab_learner.model.layers[:-2] # get rid of related output layers



    text_learner.model[-1].layers =text_learner.model[-1].layers[:-3] # get rid of related output layers

    

    lin_layers = lin_layers+ [tab_learner.data.train_ds.c]

    model = ConcatModel(tab_learner.model, text_learner.model, lin_layers, ps)

    

    loss_func = tab_learner.loss_func



    # assign layer groups for gradual training (unfreezing group)

    layer_groups = [nn.Sequential(*flatten_model(text_learner.layer_groups[0])),

                    nn.Sequential(*flatten_model(text_learner.layer_groups[1])),

                    nn.Sequential(*flatten_model(text_learner.layer_groups[2])),

                    nn.Sequential(*flatten_model(text_learner.layer_groups[3])),

                    nn.Sequential(*(flatten_model(text_learner.layer_groups[4]) + 

                                    flatten_model(model.mod_tab) +

                                    flatten_model(model.layers)))] 

    learner = Learner(data, model, loss_func=loss_func, layer_groups=layer_groups,metrics = tab_learner.metrics)

    return learner



def predict_one_item(learner,item,tab_db,text_db, **kwargs):

    '''

    learner: tabular text learner

    item: pandas series

    Return raw prediction from model and modified prediction (based on y.analyze_pred)

    '''

    tab_oneitem = tab_db.one_item(item,detach=True,cpu=True)

    text_oneitem= text_db.one_item(item,detach=True,cpu=True)

    _batch = [( ([tab_oneitem[0][0][0],tab_oneitem[0][1][0]],text_oneitem[0][0]), tab_oneitem[1][0] )]

    tabtext_onebatch = tabtext_collate(_batch)



    # send to gpu

    tabtext_onebatch = to_device(tabtext_onebatch,None)



    # taken from fastai.basic_train Learner.predict function

    res = learner.pred_batch(batch=tabtext_onebatch)

    raw_pred,x = grab_idx(res,0,batch_first=True),tabtext_onebatch[0]



    ds = learner.data.single_ds

    pred = ds.y.analyze_pred(raw_pred, **kwargs)

    return pred, raw_pred
def get_databunches(bs=64):

    # get tabtext databunch, tabular databunch (for tabular model) and text databunch (for text model)

    tab_db = get_tabular_databunch(df_reduce[cat_names + cont_names+ [dep_var]])

    text_db = get_text_databunch(df_reduce[txt_cols +[dep_var]])

    

    train_ds = ConcatDataset(tab_db.train_ds.x, text_db.train_ds.x, tab_db.train_ds.y)

    valid_ds = ConcatDataset(tab_db.valid_ds.x, text_db.valid_ds.x, tab_db.valid_ds.y)

    

    train_sampler = SortishSampler(text_db.train_ds.x, key=lambda t: len(text_db.train_ds[t][0].data), bs=bs//2)

    valid_sampler = SortSampler(text_db.valid_ds.x, key=lambda t: len(text_db.valid_ds[t][0].data))



#     train_dl = DataLoader(train_ds, bs//2, sampler=train_sampler)

    train_dl = DataLoader(train_ds, bs//2, sampler=train_sampler,shuffle=False)

    valid_dl = DataLoader(valid_ds, bs, sampler=valid_sampler)

    data = DataBunch(train_dl, valid_dl, device=defaults.device, collate_fn=tabtext_collate)

    return data,tab_db,text_db
data,tab_db,text_db = get_databunches(bs=64)
text_db.show_batch()
tab_db.show_batch()
tab_params={

    'layers':[500],

    'emb_drop': 0.3,

    'y_range': [1,9.5],

    #'use_bn': True,    

    }



text_params={

    #     'lin_ftrs':[1000],

    #     'ps': [0.001,0,0],

        'bptt':70,

        'max_len':20*70,

        'drop_mult': 1., 

         #'use_bn': True,    

    }
tab_learner = get_tabular_learner(tab_db,tab_params)

text_learner = get_text_learner(text_db,text_params)
text_learner.model
lin_layers=[500]

ps=[0.3]
# 50 is the default lin_ftrs in AWD_LSTM

lin_layers[-1]+= 50 if 'lin_ftrs' not in text_params else text_params['lin_ftrs']
lin_layers
# first layer = tabular data layer + 50 (from LSTM)

# second layer = as per your choice



lin_layers=[500+50]

ps=[0.3]

lin_layers
# be careful here. If no lin_ftrs is specified, the default lin_ftrs is 50

learner = get_tabtext_learner(data,tab_learner,text_learner,lin_layers ,ps)
learner.model
len(learner.layer_groups)

learner.layer_groups
learner.freeze()

learner.lr_find()
learner.recorder.plot(skip_end=1)
learner.fit_one_cycle(3, 5e-2, moms=(0.8, 0.7))
learner.save('tmp1')
_=learner.load('tmp1')

learner.freeze_to(-2)



learner.fit_one_cycle(3, slice(1e-03/(2.6**4),1e-03), moms=(0.8, 0.7))
learner.save('tmp2')
_=learner.load('tmp2')

learner.freeze_to(-3)



learner.fit_one_cycle(5, slice(1e-04/(2.6**4),1e-04), moms=(0.8, 0.7))
learner.save('tmp3')

_=learner.load('tmp3')

learner.unfreeze()



learner.fit_one_cycle(5, slice(1e-05/(2.6**4),1e-05), moms=(0.8, 0.7))
learner.save('final')

_ = learner.load('final')
val_df = df_reduce.loc[val_idxs].copy()
val_df.head(10)
one_item = val_df.loc[80]

one_item
pred,raw_pred = predict_one_item(learner,one_item,tab_db,text_db)
pred,raw_pred
from fastai.collab import *

ratings_1 = pd.read_csv("../input/the-movies-dataset/ratings_small.csv", parse_dates=True)

ratings_2 = pd.read_csv("../input/the-movies-dataset/ratings.csv", parse_dates=True)

ratings_1.shape, ratings_2.shape
ratings_1.head()
ratings_2.head()
df2.head()
df_ratings_movies = pd.merge(df2, ratings_1, left_on='id', right_on='movieId', how='inner')

df_ratings_movies.shape
df_ratings_movies.head(3)
df_ratings_movies = df_ratings_movies[['userId', 'movieId', 'timestamp', 'title', 'rating']]

df_ratings_movies.head(3)
data = CollabDataBunch.from_df(df_ratings_movies, seed = 42, valid_pct=0.2, user_name='userId', 

                               item_name='title', rating_name='rating')



data.show_batch()
ratings_1.rating.min(), ratings_1.rating.max()
learner = collab_learner(data, n_factors=50, y_range=(0., 5.5), wd=1e-1)
learner.lr_find()
learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(5, 1e-2)
learner = collab_learner(data, use_nn=True, emb_szs = {'userId': 50, 'title': 50},

                         layers = [256, 128], y_range=(0., 5.5))
learner.lr_find()

learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(5, 1e-3, wd=1e-1)
learner.predict(df_ratings_movies.iloc[0])
df_ratings_movies.iloc[0]
learner.get_preds(ds_type=DatasetType.Valid)
learner.model
import re

import pandas as pd

import bs4

import requests

import spacy

from spacy import displacy

nlp = spacy.load('en_core_web_sm')



from spacy.matcher import Matcher 

from spacy.tokens import Span 



import networkx as nx



import matplotlib.pyplot as plt

from tqdm import tqdm



pd.set_option('display.max_colwidth', 200)

%matplotlib inline
sample_txt = df2['overview'][0]

sample_txt
doc = nlp(sample_txt)



for tok in doc:

  print(tok.text, "...", tok.dep_)
def get_entities(sent):

  ## chunk 1

  ent1 = ""

  ent2 = ""



  prv_tok_dep = ""    # dependency tag of previous token in the sentence

  prv_tok_text = ""   # previous token in the sentence



  prefix = ""

  modifier = ""



  #############################################################

  

  for tok in nlp(sent):

    ## chunk 2

    # if token is a punctuation mark then move on to the next token

    if tok.dep_ != "punct":

      # check: token is a compound word or not

      if tok.dep_ == "compound":

        prefix = tok.text

        # if the previous word was also a 'compound' then add the current word to it

        if prv_tok_dep == "compound":

          prefix = prv_tok_text + " "+ tok.text

      

      # check: token is a modifier or not

      if tok.dep_.endswith("mod") == True:

        modifier = tok.text

        # if the previous word was also a 'compound' then add the current word to it

        if prv_tok_dep == "compound":

          modifier = prv_tok_text + " "+ tok.text

      

      ## chunk 3

      if tok.dep_.find("subj") == True:

        ent1 = modifier +" "+ prefix + " "+ tok.text

        prefix = ""

        modifier = ""

        prv_tok_dep = ""

        prv_tok_text = ""      



      ## chunk 4

      if tok.dep_.find("obj") == True:

        ent2 = modifier +" "+ prefix +" "+ tok.text

        

      ## chunk 5  

      # update variables

      prv_tok_dep = tok.dep_

      prv_tok_text = tok.text

  #############################################################



  return [ent1.strip(), ent2.strip()]
get_entities(sample_txt)
from tqdm import tnrange, tqdm_notebook

entity_pairs = []



for i in tqdm_notebook(df2['overview']):

  entity_pairs.append(get_entities(i))
entity_pairs[10:20]
def get_relation(sent):



  doc = nlp(sent)



  # Matcher class object 

  matcher = Matcher(nlp.vocab)



  #define the pattern 

  pattern = [{'DEP':'ROOT'}, 

            {'DEP':'prep','OP':"?"},

            {'DEP':'agent','OP':"?"},  

            {'POS':'ADJ','OP':"?"}] 



  matcher.add("matching_1", None, pattern) 



  matches = matcher(doc)

  k = len(matches) - 1



  span = doc[matches[k][1]:matches[k][2]] 



  return(span.text)
import gc

gc.collect()

relations = [get_relation(i) for i in tqdm_notebook(df2['overview'])]
pd.Series(relations).value_counts()[:50]
# extract subject

source = [i[0] for i in entity_pairs]



# extract object

target = [i[1] for i in entity_pairs]



kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
# create a directed-graph from a dataframe

G=nx.from_pandas_edgelist(kg_df, "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())
gc.collect()
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="takes"], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()