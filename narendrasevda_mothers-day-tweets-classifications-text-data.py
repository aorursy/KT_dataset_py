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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import warnings

import xgboost as xgb

import lightgbm as lgb

from scipy.stats import skew

from scipy import stats

from scipy.stats.stats import pearsonr

from scipy.stats import norm

from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve,train_test_split

from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler, Normalizer, RobustScaler,LabelEncoder

from collections import Counter

warnings.filterwarnings('ignore')

sns.set(style='white', context='notebook', palette='deep')
#importing data

train = pd.read_csv("/kaggle/input/new_train.csv")

test = pd.read_csv("/kaggle/input/new_test.csv")

test.head()
train.head()
# creating new variable id as we needed in final output file

id = test["id"].values

y_train_1 = train["sentiment_class"]

#train.drop(["sentiment_class"], axis=1,inplace=True)#droping target data from file

train.head()
#now combining train and test

dataset = pd.concat([train,test],sort=False,ignore_index=True)

temp_df = pd.concat([train,test],sort=False,ignore_index=True)

dataset.drop("id",axis=1,inplace=True)#dropping id from final data

dataset.info()
print(dataset["lang"].isnull().value_counts())
print(y_train_1)
from wordcloud import WordCloud,STOPWORDS

df=train[train['sentiment_class']==0]# do analy for 1 and -1 too

words = ' '.join(df['original_text'])

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=3000,

                      height=2500

                     ).generate(cleaned_word)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#we are here cleaning the tweets like removing stopwords,and other stuffs so we get only letters

import re

import nltk

from nltk.corpus import stopwords

def tweet_to_words(raw_tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 

def clean_tweet_length(raw_tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return(len(meaningful_words)) 

dataset.drop("sentiment_class",axis=1,inplace=True)

dataset['clean_tweet']=dataset['original_text'].apply(lambda x: tweet_to_words(x))

dataset['Tweet_length']=dataset['original_text'].apply(lambda x: clean_tweet_length(x))

dataset.drop("original_text",axis=1,inplace=True)

dataset["lang"] = dataset["lang"].fillna("en")#filling some values

dataset.head()
#for transforming clean_tweet column as it contain sentances or words or string

from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer(analyzer = "word")

train_features= v.fit_transform(dataset['clean_tweet'])

dataset.drop("clean_tweet",axis=1,inplace=True)

features = train_features.toarray()

features = pd.DataFrame(features)

dataset = dataset.join(features)
#label encoding on lang and author columns

le = LabelEncoder()

dataset['lang']=le.fit_transform(dataset["lang"].values)

dataset['original_author']=le.fit_transform(dataset["original_author"].values)

dataset.head()
# to encode categorical data eg. lang and author columns

enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(dataset[['lang']]).toarray())

enc_df_1 = pd.DataFrame(enc.fit_transform(dataset[['original_author']]).toarray())
#now concating both encoded data to main data

dataset = pd.concat([dataset,enc_df],axis=1,sort=False,ignore_index=True)

dataset = pd.concat([dataset,enc_df_1],axis=1,sort=False,ignore_index=True)
from sklearn.model_selection import cross_val_score,GridSearchCV

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier,StackingClassifier,RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import f1_score
print(dataset.shape)

dataset.head()
# we have so many columns(22272) so reduce them in 10 columns

pca  = PCA(n_components=10, random_state=1)

df = pca.fit_transform(dataset)

df = pd.DataFrame(data=df, columns=["compenent_1","compenent_1","compenent_1","compenent_1",

                                    "compenent_1","compenent_1","compenent_1","compenent_1",

                                     "compenent_1","compenent_1"])
#splitting train and test data

X_train = dataset[:len(train)]

test = dataset[len(train):]
#apply train test split to test model performence only donot do in final prediction

T_train,T_test,y_train,y_test = train_test_split(X_train,y_train_1, random_state=1)
lr = LogisticRegression(C=1)

xgb = XGBClassifier( n_estimators = 120 )              

gboost = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 4, n_estimators = 100)

bayes = GaussianNB()

rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=12, max_features='auto',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=100,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)
#u can use grid search to find parameters i already used that



xgb.fit(T_train,y_train)

y_pred = xgb.predict(T_test)

print(xgb.score(T_train,y_train))

print(100*f1_score(y_test, y_pred,average='weighted'))
output = pd.DataFrame({"id":id,"sentiment_class":y_pred})

output.to_csv("submission_1.csv",index=False)