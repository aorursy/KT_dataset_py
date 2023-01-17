# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

#df = df.dropna()

df.shape

test = pd.read_csv('../input/test.csv')
import ast

df.columns

print(df.shape)

ids = df.belongs_to_collection

ids = df.belongs_to_collection.apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))

ids.str.strip("[]")

ids.isnull().sum()

ids.shape
runtime_val = df.runtime

count = len(runtime_val)

runtime_val = runtime_val.sort_values()

runtime_val = runtime_val.dropna()

plt.xlabel('Runtime')

plt.ylabel('Count')

sn.set(style='darkgrid')

sn.distplot(runtime_val,kde=True,rug=True,color='green')

plt.show()
df.status.value_counts()

df.status.factorize()[0]
released = len(df.status=='Released')

rumor = len(df.status=='Rumored')



label = ['Released','Rumored']

li = [released,rumor]



index = np.arange(len(label))



sn.barplot(label,index)

plt.show()
tag = df.tagline

tag = tag.dropna()

print(tag.shape)

plot = df.overview

plot = plot.dropna()

print(plot.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords



tfidf = TfidfVectorizer(sublinear_tf=True,min_df=6,encoding='latin-1',ngram_range=(1,2),stop_words='english')

tfidf1 = TfidfVectorizer(sublinear_tf=True,min_df=10,encoding='latin-1',ngram_range=(1,3),stop_words='english')



features = tfidf.fit_transform(tag)

features_overview = tfidf1.fit_transform(plot)

points = tfidf.vocabulary_

most_freq = tfidf1.vocabulary_

print('TAGLINE: ->')

print(points)

print()
print('OVERVIEW: ->')

print(most_freq)
df.revenue.shape                # Our TF-IDF values for our list
features.shape
box_office = df.revenue >999999

box_office.value_counts()



## THESE MOVIES ARE AT A STANDARD BOX-OFFICE EARNINGS
genre = df.genres
df.corr()
sn.heatmap(df.corr(),cmap = 'RdYlBu',annot=True)
sn.scatterplot(df.budget,df.revenue)

plt.show()

sn.scatterplot(df.runtime,df.popularity)

plt.show()
import ast

import itertools

from collections import Counter

from wordcloud import WordCloud,STOPWORDS

from nltk.tokenize import word_tokenize

stopwords=set(STOPWORDS)

size = plot.shape

print(size)

word = []

respo = pd.DataFrame(df.genres)

respo = df.genres.apply(lambda x: list(map(lambda d: list(d.values())[1], ast.literal_eval(x)) if isinstance(x, str) else []))

respo_count = Counter(itertools.chain.from_iterable(respo))

print()

print("Total number of genres:",len(respo_count))

print("Genre frequency:\n"+'\n'.join(['{} : {}'.format(g, respo_count[g]) for g in respo_count]))

## CAST IN ORDERS OF APPEARANCE



casting = df.cast.apply(lambda y:list(map(lambda e:list(e.values())[5],ast.literal_eval(y)) if isinstance(y,str) else []))

cast_count = Counter(itertools.chain.from_iterable(casting))



#FIND THE MOST APPEARED ACTOR/ACTRESS

#RUN THE LINE BELOW*#



#print(cast_count.most_common())

print()

print(len(cast_count))
df.columns
from sklearn.model_selection import train_test_split



y_val = df.revenue

Y= y_val.apply(np.log10)

feat_movie  = ['budget','popularity','runtime']

feat = df[feat_movie]

feat.runtime = pd.get_dummies(feat.runtime)

print(feat.isnull().sum())

X_train, X_test, Y_train, Y_test = train_test_split(feat,Y, test_size=0.3,random_state=4)
params = {'objective':'regression',

          'num_leaves' : 40,

          'min_data_in_leaf' : 20,

          'max_depth' : 6,

          'learning_rate': 0.001,

          "metric": 'rmse',

          "random_state" : 42,

          "lambda_l2" : 0.005,

          "verbosity": -1}
test = pd.read_csv('../input/test.csv')

X_t = test[feat_movie]

X_t.shape
import lightgbm as lgbm



lgbm_train = lgbm.Dataset(X_train,Y_train)

lgbm_eval = lgbm.Dataset(X_test,Y_test,reference = lgbm_train)

print(X_train.shape)

print(X_t.shape)
print("TRAINING: ")



gbm = lgbm.train(params,lgbm_train,num_boost_round=3000,valid_sets = lgbm_eval,early_stopping_rounds=15)


y_predictions = gbm.predict(X_t)

y_predictions = (10 ** y_predictions)

print("\n",y_predictions)

print(y_predictions.shape)
#FOR LGBM

submissions= pd.DataFrame({'id':test.id,'revenue':y_predictions},

                         columns = ['id','revenue'])



submissions.to_csv('submission.csv',index=False)
