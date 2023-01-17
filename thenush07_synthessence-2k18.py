# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

df.head(4)
x_train = df

x_train.head(2)
x_train = x_train.drop(['restaurant_average_prices'],axis=1)
x_train = x_train.drop(x_train.loc[:,"restaurant_open_hours_monday":"restaurant_rating_atmosphere"],axis=1)
x_train = x_train.drop(['restaurant_open_hours_friday'],axis=1)
x_train.isnull().sum()
x_train.head()
#x_train_num = x_train.loc[:,"restaurant_rating_food":"restaurant_rating_value"]
#x_train_text = x_train.drop(x_train.loc[:,"restaurant_rating_food":"restaurant_rating_value"],axis=1)

x_train_text = x_train

x_train_text.head()
x_train_text = x_train_text.drop(['date','id'],axis=1)
x_train_text.info()
x_train_text = x_train_text.dropna(axis=0,subset=['title'])
x_train_text.info()
for a in x_train_text:

    x_train_text = x_train_text.dropna(axis=0,subset=[a])
x_train_text.info()
y_train = x_train_text.loc[:,'ratingValue_target']

y_train.head()
x_train_num = x_train_text.loc[:,"restaurant_rating_food":"restaurant_rating_value"]

x_train_text = x_train_text.drop(x_train_text.loc[:,"restaurant_rating_food":"restaurant_rating_value"],axis=1)
x_train_text = x_train_text.loc[:,'restaurant_cuisine':'title']
x_train_text.info()
for a in x_train_text.columns:

    x_train_text[a] = x_train_text[a].apply(lambda x: " ".join(x.lower() for x in x.split()))
for a in x_train_text.columns:

    x_train_text[a] = x_train_text[a].str.replace('[^\w\s]','')
x_train_text.head()
from nltk.corpus import stopwords
stop = stopwords.words('english')
for a in x_train_text.columns:

    x_train_text[a] = x_train_text[a].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
x_train_text.head()
freq = pd.Series(' '.join(x_train_text['text']).split()).value_counts()[:10]

freq
freq = freq.drop('good',axis=0)
freq = freq.drop(['great','nice'],axis=0)
freq
freq = list(freq.index)

x_train_text['text'] = x_train_text['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    
freq = pd.Series(' '.join(x_train_text['restaurant_good_for']).split()).value_counts()[-10:]

freq
freq = freq.drop(['dining','meetings','occasion','scene','bar','view','business'],axis=0)

freq
freq = list(freq.index)

x_train_text['restaurant_good_for'] = x_train_text['restaurant_good_for'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
freq = pd.Series(' '.join(x_train_text['text']).split()).value_counts()[-100:]

freq
freq = list(freq.index)

x_train_text['text'] = x_train_text['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
x_train_text.head()
from textblob import TextBlob
from nltk.stem import PorterStemmer
st = PorterStemmer()
from textblob import Word
for a in x_train_text.columns:

    x_train_text[a] = x_train_text[a].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



x_train_text.head()
x_train_text['sentiment_title'] = x_train_text['title'].apply(lambda x: TextBlob(x).sentiment[0]) 

x_train_text['sentiment_title'] = x_train_text['sentiment_title'].round(1)

x_train_text.head()
x_train_text['cnt_cuisine'] = x_train_text['restaurant_cuisine'].apply(lambda x: len(x.split()))

x_train_text.head()
x_train_text['cnt_features'] = x_train_text['restaurant_features'].apply(lambda x: len(x.split()))

x_train_text['cnt_good_for'] = x_train_text['restaurant_good_for'].apply(lambda x: len(x.split()))

x_train_text['cnt_meals'] = x_train_text['restaurant_meals'].apply(lambda x: len(x.split()))
x_train_text = x_train_text.drop(x_train_text.loc[:,"restaurant_cuisine":"restaurant_meals"],axis=1)
x_train_text['sentiment_text'] = x_train_text['text'].apply(lambda x: TextBlob(x).sentiment[0]) 

x_train_text['sentiment_text'] = x_train_text['sentiment_text'].round(1)

x_train_text['sentiment_text'][:10]
x_train_text = x_train_text.drop(['text','title'],axis=1)

x_train_text.head()
x_train_num.head()
x_train_text = x_train_text.join(x_train_num)

x_train = x_train_text

x_train.head()
del x_train_text

del x_train_num
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.metrics import f1_score
rf = GradientBoostingClassifier()
rf.fit(x_train,y_train)
#from sklearn import cross_validation



#num_folds = 10

#num_instances = len(x_train)

#KFold = cross_validation.KFold(n=num_instances, n_folds=num_folds)

#model = GradientBoostingClassifier()

#results = cross_validation.cross_val_score(model,x_train,y_train,cv=KFold)

#print(results)

#print(results.mean()*100)
test = pd.read_csv("../input/test.csv")
test.head()
test = test.drop(['restaurant_average_prices','date','restaurant_rating_atmosphere'],axis=1)
test.isnull().sum()
for a in test.columns:

    test = test.dropna(axis=0,subset=[a])
test_text = test.drop(test.loc[:,"restaurant_rating_food":"restaurant_rating_value"],axis=1)

test_num = test.loc[:,"restaurant_rating_food":"restaurant_rating_value"]
test_text.info()
test_text = test_text.drop(test_text.loc[:,"restaurant_open_hours_friday":"restaurant_open_hours_wednesday"],axis=1)
id_t = test_text.loc[:,'id']

test_text = test_text.drop(['id'],axis=1)
for a in test_text.columns:

    test_text[a] = test_text[a].apply(lambda x: " ".join(x.lower() for x in x.split()))
for a in test_text.columns:

    test_text[a] = test_text[a].str.replace('[^\w\s]','')
for a in test_text.columns:

    test_text[a] = test_text[a].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
for a in test_text.columns:

    test_text[a] = test_text[a].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
test_text['sentiment_title'] = test_text['title'].apply(lambda x: TextBlob(x).sentiment[0]) 

test_text['sentiment_title'] = test_text['sentiment_title'].round(1)
test_text['cnt_cuisine'] = test_text['restaurant_cuisine'].apply(lambda x: len(x.split()))
test_text['cnt_features'] = test_text['restaurant_features'].apply(lambda x: len(x.split()))

test_text['cnt_good_for'] = test_text['restaurant_good_for'].apply(lambda x: len(x.split()))

test_text['cnt_meals'] = test_text['restaurant_meals'].apply(lambda x: len(x.split()))
test_text = test_text.drop(test_text.loc[:,"restaurant_cuisine":"restaurant_meals"],axis=1)
test_text['sentiment_text'] = test_text['text'].apply(lambda x: TextBlob(x).sentiment[0]) 

test_text['sentiment_text'] = test_text['sentiment_text'].round(1)
test_text = test_text.drop(['text','title'],axis=1)
test_num.head()
test_text = test_text.join(test_num)

test = test_text

test.head()
x_train.head()
pred = rf.predict(test)

pred
result = pd.DataFrame(pred[:])

result.index.name = 'id'

result.columns = ['ratingValue_target']

result.to_csv('output.csv', index=True)