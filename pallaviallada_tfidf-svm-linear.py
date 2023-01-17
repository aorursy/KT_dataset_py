# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

# Any results you write to the current directory are saved as output.
def load_data(filename, flag):
    with open(filename,'r') as fp:
        json_data = json.load(fp)
    id_list = []
    ingredients_list = []
    cuisine_list = []
    for row in json_data:
        id_list.append(row['id'])
        ingredients_list.append(row['ingredients'])
        if(flag == 'train'):
            cuisine_list.append(row['cuisine'])
    if(flag == 'train'):
        df = pd.DataFrame(columns = ['id','ingredients','cuisine'])
        df['id'] = pd.Series(id_list)
        df['ingredients'] = pd.Series(ingredients_list)
        df['cuisine'] = pd.Series(cuisine_list)
    else:
        df = pd.DataFrame(columns = ['id','ingredients'])
        df['id'] = pd.Series(id_list)
        df['ingredients'] = pd.Series(ingredients_list)

    return df
train = load_data('../input/train.json','train')
test = load_data('../input/test.json','test')

len(train), len(test)


#preprocess the ingredients data by removing the [] brackets
def removespaces(ingredients):
    ingreds = []
    for ingred in ingredients:
        ingred_mod = ingred.replace(" ","_")
        ingreds.append(ingred_mod)
    return ingreds

def removebracs(ingredients):
    ingreds = ""
    for ingred in ingredients:
        ingred_mod = ingred.replace(" ","_")
        ingreds = ingreds + ingred_mod + ","
    return ingreds[:-1]
    
#recipes['ingredients_w2v'] = recipes['ingredients'].apply(removespaces)
train['ingredients_tfidf'] = train['ingredients'].apply(removebracs)
test['ingredients_tfidf'] = test['ingredients'].apply(removebracs)
train.head()
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['ingredients_tfidf'])
X_test = vectorizer.transform(test['ingredients_tfidf'])

lb = LabelEncoder()
y = lb.fit_transform(train['cuisine'])

from sklearn import svm
C = 1.0  # SVM regularization parameter
 
# SVC with linear kernel
svc = svm.SVC(kernel='linear', C=C)
# # LinearSVC (linear kernel)
# lin_svc = svm.LinearSVC(C=C).fit(X_tfidf, y)
# # SVC with RBF kernel
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_tfidf, y)
# # SVC with polynomial (degree 3) kernel
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_tfidf, y)

#svm_predictions = svc.predict(X_test)
 

ovr = OneVsRestClassifier(svc)
model = ovr.fit(X_train, y)
y_preds=model.predict(X_test)
preds = lb.inverse_transform(y_preds)

sub = pd.DataFrame({'id': test['id'], 'cuisine': preds}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)

