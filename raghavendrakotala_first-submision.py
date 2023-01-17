# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import ast

# Any results you write to the current directory are saved as output.
train_df= pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
len(train_df)
train_df.info()
test_df.head()
final_df = pd.concat([train_df,test_df])
final_df['cast_mod'] = final_df.cast.fillna('[]').apply(lambda x : [a['name'].replace(' ','') for a in ast.literal_eval(x)])
final_df.cast_mod = final_df.cast_mod.apply(lambda x:' '.join(x))
final_df['genres_mod'] = final_df.genres.fillna('[]').apply(lambda x : [a['name'] for a in ast.literal_eval(x)])
final_df['genres_mod'] = final_df['genres_mod'].apply(lambda x: ' '.join(x))
train_df = final_df[:len(train_df)]

test_df = final_df[len(train_df):]
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

from scipy.sparse import hstack
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

vectorizer = count.fit(train_df['genres_mod']+train_df.cast_mod)
train_features = vectorizer.transform(train_df['genres_mod']+train_df.cast_mod)

test_features = vectorizer.transform(test_df['genres_mod']+test_df.cast_mod)
train_features.shape
final_df.info()
X = final_df[['budget','popularity','original_language','runtime','release_date']]

X['release_date'] = pd.to_datetime(X.release_date)

X['release_year'] = X['release_date'].dt.year

X['release_motnh'] = X['release_date'].dt.month

X['release_day'] = X['release_date'].dt.day

X = pd.get_dummies(X.drop(['release_date'],axis=1),columns=['original_language'],drop_first=True)

test = X[len(train_df):]

X = X[:len(train_df)]

y = train_df.revenue
train_final = hstack([X,train_features]).tocsr()

test_final = hstack([test,test_features]).tocsr()

train_final.shape,test_final.shape
test.shape,test_df.shape
train_df.plot(kind="scatter",x='budget',y='revenue')
X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_final,y,test_size=0.2,random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor



lg = XGBRegressor(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=10)

lg.fit(X_train,y_train)

lg.score(X_test,y_test)
results = pd.DataFrame()

results['id'] = test_df['id']

results['revenue'] = lg.predict(test_final)
results.head()

results.to_csv('submission1.csv',index=False)


type(list(train_df.belongs_to_collection[1])),train_df.belongs_to_collection[1],ast.literal_eval(train_df.belongs_to_collection[1])

ast.literal_eval(train_df.belongs_to_collection[1])[0]['name']

train_df.belongs_to_collection[0]
for b in train_df.belongs_to_collection:

#     print(b)

    pass
train_df.crew[0]