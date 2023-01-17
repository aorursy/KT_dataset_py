# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
from matplotlib import pyplot as plt, style
style.use('ggplot')

import json
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = open('../input/train.json','r').read()
train_json = json.loads(train)
test = open('../input/test.json','r').read()
test_json = json.loads(test)
df_dict = dict()
df_dict['id']=[]
df_dict['cuisine']=[]
df_dict['ingredient_list'] = []
train_df = pd.DataFrame(df_dict,dtype=np.int64)
test_df = train_df.copy()
print(train_df.head())
import re
i = 0
for curr_json in train_json:    
    ingredient_list = " ".join([ re.sub('\s',"_",ingredient) for ingredient in curr_json['ingredients'] ])
    train_df.loc[i] =  [curr_json['id'],curr_json['cuisine'],ingredient_list]
    i+=1
print(train_df.head())
test_df.drop(['cuisine'],1,inplace=True)
i = 0
for curr_json in test_json:    
    ingredient_list = " ".join([ re.sub('\s',"_",ingredient) for ingredient in curr_json['ingredients'] ])
    test_df.loc[i] =  [curr_json['id'],ingredient_list]
    i+=1
print(test_df.head())
Y_train_all = train_df['cuisine']
train_df.drop('cuisine',1,inplace=True)
print(Y_train_all.head())
print(train_df.shape,test_df.shape)
X_all = train_df.append(test_df)
print(X_all.shape)
from sklearn.feature_extraction.text import CountVectorizer
ingredient_list = X_all['ingredient_list']
cv = CountVectorizer().fit(ingredient_list)
ingredient_list_count = cv.transform(ingredient_list)
print(ingredient_list_count.shape)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(ingredient_list_count)
ingredient_list_tfidf = tfidf_transformer.transform(ingredient_list_count)
print(ingredient_list_tfidf.shape)
X_train_all = ingredient_list_tfidf[:39774]
X_test = ingredient_list_tfidf[39774:]
from sklearn.model_selection import train_test_split
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train_all,Y_train_all,test_size=0.2)
X_test_index = test_df['id']
print("train :{}{}, test: {} index: {}, validation: {}{}".format(X_train.shape,Y_train.shape,
                                                                X_test.shape,X_test_index.shape,
                                                                X_validation.shape,Y_validation.shape))
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train,Y_train)
print(clf.score(X_validation,Y_validation))

predictions = clf.predict(X_test)
print(predictions.shape)

# Any results you write to the current directory are saved as output.
res = pd.DataFrame({
    "id":X_test_index,
    "cuisine":predictions
})
res.to_csv("sol.csv",header=True,index=False)
