# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# load files

train_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")

test_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")

att_data = pd.read_csv("../input/attributes.csv")

descriptions = pd.read_csv("../input/product_descriptions.csv")



from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords

stop = stopwords.words('english')

stemmer = SnowballStemmer('english')

def stm(s):

    return ' '.join([stemmer.stem(word)  for word in str(s).split() if word not in stop])
train_data['search_term']=train_data['search_term'].map(lambda x: stm(x))

descriptions['product_description']=descriptions['product_description'].map(lambda x: stm(x))

brands=att_data[['product_uid','value']][att_data.name=='MFG Brand Name']
att_data['value']=att_data['value'].map(lambda x: stm(x))

ser_att=pd.Series()

for p,v in zip(att_data['product_uid'],att_data['value']):

	s=' '.join([str(ser_att.get(p,'')),v])

	ser_att[p]=s
def search_in_str(search,s):

    return sum([s.count(term) for term in search.split()])

train_data=train_data.merge(brands,how='left',on='product_uid')

train_data.columns=['id', 'product_uid', 'product_title', 'search_term', 'relevance','brand']

train_data=train_data.merge(descriptions,how='left',on='product_uid')

train_data['search_in_title']=[search_in_str(x,y) for (x,y) in zip(train_data['search_term'],train_data['product_title'])]

train_data['search_in_brand']=[search_in_str(str(x),str(y)) for (x,y) in zip(train_data['search_term'],train_data['brand'])]

train_data['search_in_desc']=[search_in_str(str(x),str(y)) for (x,y) in zip(train_data['search_term'],train_data['product_description'])]

train_data['attr']=train_data['product_uid'].map(lambda x: ser_att.get(x,''))

train_data['search_in_att']=[search_in_str(str(x),str(y)) for (x,y) in zip(train_data['search_term'],train_data['attr'])]

#del train_data['attr']

#del train_data['product_title']

#del train_data['brand']

#del train_data['product_description']
test_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")

test_data['search_term']=test_data['search_term'].map(lambda x: stm(x))

test_data=test_data.merge(brands,how='left',on='product_uid')

test_data.columns=['id', 'product_uid', 'product_title', 'search_term','brand']

test_data=test_data.merge(descriptions,how='left',on='product_uid')

test_data['search_in_title']=[search_in_str(x,y) for (x,y) in zip(test_data['search_term'],test_data['product_title'])]

test_data['search_in_brand']=[search_in_str(str(x),str(y)) for (x,y) in zip(test_data['search_term'],test_data['brand'])]

test_data['search_in_desc']=[search_in_str(str(x),str(y)) for (x,y) in zip(test_data['search_term'],test_data['product_description'])]

test_data['attr']=test_data['product_uid'].map(lambda x: ser_att.get(x,''))

test_data['search_in_att']=[search_in_str(str(x),str(y)) for (x,y) in zip(test_data['search_term'],test_data['attr'])]

#del test_data['product_title']

#del test_data['brand']

#del test_data['product_description']

#del test_data['attr']

predictors=['search_in_title','search_in_brand','search_in_desc','search_in_att']

y_train = train_data['relevance'].values

X_train = train_data[predictors].values

X_test = test_data[predictors].values

id_test = test_data['id']



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
from sklearn.ensemble import RandomForestRegressor

from sklearn import cross_validation

import math

rf = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=0)



predictors=['search_in_title','search_in_brand','search_in_desc','search_in_att']

scores=cross_validation.cross_val_score(clf,train_data[predictors],train_data['relevance'],cv=5,scoring='mean_squared_error')

print(np.mean([math.sqrt(-x) for x in scores]))