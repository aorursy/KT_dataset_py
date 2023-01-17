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
import warnings

import gc,time

#nlp

import string

import re    #for regex

import nltk

from nltk.corpus import stopwords



import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style("dark")

eng_stopwords = set(stopwords.words("english"))

warnings.filterwarnings("ignore")
train=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

test=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

submission=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

test_labels=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
train.head()


x=train.iloc[:,2:].sum()

x#COLUMN wise sum



fig = px.bar( x=x.index, y=x.values,

             height=400)

fig.show()
clean_df=train.loc[(train.toxic==0) &  (train.severe_toxic==0) &(train.obscene==0) &

                   (train.threat==0)  &(train.insult==0) &(train.identity_hate==0)] # clean  comments



toxic_df=train.loc[(train.toxic==1)]# toxic comments



#creating test set

clean_test=clean_df.iloc[:28669]# 20percent of  total clean comments which are  approximately 144000

toxic_test=toxic_df.iloc[:3059]# 20percent of Toxic  comments which are approximately 15000



test_set=clean_test.append(toxic_test,ignore_index=True).sample(frac=1)# appending 2 dataframes and shuffling them

test_set.drop(['id','severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)

print(test_set.shape)



#creating train set

clean_train=clean_df.iloc[28669:]

toxic_train=toxic_df.iloc[3059:]

df=clean_train.append(toxic_train,ignore_index=True).sample(frac=1)



# df=clean_df.append(toxic_df,ignore_index=1).sample(frac=1)# appending 2 dataframes and shuffling them

df.drop(['id','severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)

df.shape

# Applying a first round of text cleaning techniques



def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text

# Applying the cleaning function to both test and training datasets

df['comment_text'] = df['comment_text'].apply(lambda x: clean_text(x))

test_set['comment_text']=test_set['comment_text'].apply(lambda x:clean_text(x))
dictionary_clean={0:'clean',1:'toxic'}

df['target_name']=df['toxic'].map(dictionary_clean)

test_set['target_name']=test_set['toxic'].map(dictionary_clean)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegressionCV

from sklearn.pipeline import make_pipeline

from imblearn.over_sampling import  SMOTE





smt = SMOTE(random_state=777, k_neighbors=1)



vec = TfidfVectorizer(min_df=3,max_features=10000,strip_accents='unicode',

                     analyzer='word',ngram_range=(1,2),token_pattern=r'\w{1,}',use_idf=1,smooth_idf=1,sublinear_tf=1,

                     stop_words='english')



vec_fit=vec.fit_transform(df.comment_text)



clf = LogisticRegressionCV()





# Over Sampling

X_SMOTE, y_SMOTE = smt.fit_sample(vec_fit, df.toxic)

from collections import Counter

#we over sampled it 

print(Counter(y_SMOTE))

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=0.1, solver='sag')

scores = cross_val_score(clf, X_SMOTE,y_SMOTE, cv=5,scoring='f1_weighted')
scores
clf.fit(X_SMOTE,y_SMOTE)
from sklearn import metrics



def print_report1(df):

    y_test =  df.toxic

    test_features=vec.transform(df.comment_text)

    y_pred = clf.predict(test_features)

    report = metrics.classification_report(y_test, y_pred,

        target_names=list(df.target_name.unique()))

    print(report)

    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))



print_report1(test_set)
import eli5

# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

eli5.show_weights(clf, vec=vec, top=15,

                  target_names=['clean','toxic'])



#  if we got the BIAS term that occurs

#because we are using Linear model for classification and the Intercept added to the equation is termed BIAS here



print(test_set.comment_text[0])

print('\n')

print(test_set.toxic[0])
import eli5

eli5.show_prediction(clf, test_set.comment_text[0], vec=vec,

                     target_names=list(df.target_name.unique()),top=15)

# it shows probability of each of  the 2 classes and then shows which features contributed the most and which

# contributed the least in each class

# top argument shows the  top n features that contibuted to the prediction of each class