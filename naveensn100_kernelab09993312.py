# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re



from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import precision_score, recall_score, f1_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ds_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

ds_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

ds_sample_sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



ds_train.head()

print (ds_train.shape, ds_test.shape, ds_sample_sub.shape)





ds_train['target'].value_counts(normalize=True)*100
print(ds_train.duplicated(subset=['text']).sum())

print(ds_train.duplicated(subset=['text','target']).sum())

print(ds_train.duplicated(subset=['keyword','text']).sum())

print(ds_train.duplicated(subset=['location','text']).sum())

print(ds_train.duplicated(subset=['keyword','location','text']).sum())

print(ds_train.duplicated(subset=['keyword','location','text','target']).sum())

print(ds_train.duplicated(subset=['keyword','text','target']).sum())

print(ds_train.duplicated(subset=['location','text','target']).sum())
ds_train.isnull().sum()
ds_test.isnull().sum()
print(f'Number of unique values in keyword = {ds_train["keyword"].nunique()} (Training) - {ds_test["keyword"].nunique()} (Test)')

print(f'Number of unique values in location = {ds_train["location"].nunique()} (Training) - {ds_test["location"].nunique()} (Test)')
ds_train['target_mean'] = ds_train.groupby('keyword')['target'].transform('mean')



fig = plt.figure(figsize=(8, 72), dpi=100)



sns.countplot(y=ds_train.sort_values(by='target_mean', ascending=False)['keyword'],

              hue=ds_train.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()



ds_train.drop(columns=['target_mean'], inplace=True)
raw_loc = ds_train.location.value_counts()

top_loc = list(raw_loc[raw_loc>=10].index)

top_only = ds_train[ds_train.location.isin(top_loc)]



top_l = top_only.groupby('location').mean()['target'].sort_values(ascending=False)

plt.figure(figsize=(14,6))

sns.barplot(x=top_l.index, y=top_l)

plt.axhline(np.mean(ds_train.target))

plt.xticks(rotation=80)

plt.show()
import re



def clean(tweet): 

            

    # Special characters

    tweet = re.sub(r"\x89Û_", "", tweet)

    tweet = re.sub(r"\x89ÛÒ", "", tweet)

    tweet = re.sub(r"\x89ÛÓ", "", tweet)

    tweet = re.sub(r"\x89ÛÏ", "", tweet)

    tweet = re.sub(r"\x89Û÷", "", tweet)

    tweet = re.sub(r"\x89Ûª", "'", tweet)

    tweet = re.sub(r"\x89Û\x9d", "", tweet)

    tweet = re.sub(r"å_", "", tweet)

    tweet = re.sub(r"\x89Û¢", "", tweet)

    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)

    tweet = re.sub(r"åÊ", "", tweet)

    tweet = re.sub(r"åÈ", "", tweet)   

    tweet = re.sub(r"å¨", "", tweet)

    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)

    tweet = re.sub(r"åÇ", "", tweet)

    tweet = re.sub(r"åÀ", "", tweet)

    tweet = re.sub(r'\n',' ', tweet) # Remove line breaks

    tweet = re.sub('\s+', ' ', tweet).strip() # Remove leading, trailing, and extra spaces

    

    return tweet



ds_train['text_cleaned'] = ds_train['text'].apply(lambda x : clean(x))

ds_test['text_cleaned'] = ds_test['text'].apply(lambda x : clean(x))
def hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'



def mentions(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'





ds_train['hashtags'] = ds_train['text'].apply(lambda x : hashtags(x))

ds_train['mentions'] = ds_train['text'].apply(lambda x : mentions(x))

ds_test['hashtags'] = ds_test['text'].apply(lambda x : hashtags(x))

ds_test['mentions'] = ds_test['text'].apply(lambda x : mentions(x))
ds_train.columns
from nltk.stem import PorterStemmer

from nltk.corpus import stopwords



corpus  = []

pstem = PorterStemmer()

for i in range(ds_train['text_cleaned'].shape[0]):

    #Remove unwanted words

    text = re.sub("[^a-zA-Z]", ' ', ds_train['text_cleaned'][i])

    #Transform words to lowercase

    text = text.lower()

    text = text.split()

    #Remove stopwords then Stemming it

    text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    #Append cleaned tweet to corpus

    corpus.append(text)



corpus2  = []

pstem2 = PorterStemmer()

for i in range(ds_test['text_cleaned'].shape[0]):

    #Remove unwanted words

    text = re.sub("[^a-zA-Z]", ' ', ds_test['text_cleaned'][i])

    #Transform words to lowercase

    text = text.lower()

    text = text.split()

    #Remove stopwords then Stemming it

    text = [pstem2.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    #Append cleaned tweet to corpus

    corpus2.append(text)
!pip install transformers
from sklearn import preprocessing,feature_extraction



count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(corpus)

y = ds_train['target']

test_vectors = count_vectorizer.transform(corpus2)
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

svc = SVC()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 1)



from xgboost import XGBClassifier

xgb = XGBClassifier()



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()



from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()



from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()



from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
score = cross_val_score(nb, train_vectors, ds_train["target"], cv=8, scoring="f1")

score
nb.fit(train_vectors,y)
import pickle

file = 'xgbmodel.pkl'

pickle.dump(svc, open(file, 'wb'))
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



sample_submission['target'] = nb.predict(test_vectors)



sample_submission.to_csv("Sub_NaveenV001.csv",index = False)