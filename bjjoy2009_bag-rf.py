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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

print('read train')
train = pd.read_json('../input/train.json')
vectorizer = CountVectorizer(max_features = 1000)
ingredients = train['ingredients']
words_list = [' '.join(x) for x in ingredients]
print(len(words_list))
 
 
bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)
 
print('read test ')
test = pd.read_json('../input/test.json')
test_ingredients = test['ingredients']
test_ingredients_words = [' '.join(x) for x in test_ingredients]
test_ingredients_arrays = vectorizer.transform(test_ingredients_words).toarray()

print('fit')
rf = RandomForestClassifier(n_estimators=100)
rf_fit = rf.fit(bag_of_words,train['cuisine'])
print('predict')
result = rf_fit.predict(test_ingredients_arrays)
output = pd.DataFrame(data={"id":test["id"], "cuisine":result},columns=['id', 'cuisine'])
print('save')
result_filename = 'bag_rf.csv'
output.to_csv(result_filename, index=False, quoting=3 )
print('end')
