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
!pip install pymystem3
import gc

import nltk

from nltk.corpus import stopwords

from pymystem3 import Mystem

from string import punctuation



from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.metrics import accuracy_score



from scipy.sparse import hstack

from scipy.sparse import csr_matrix



nltk.download("stopwords")
df_train = pd.read_csv('../input/train.csv', index_col='item_id')

df_test = pd.read_csv('../input/test.csv', index_col='item_id')
train_target = df_train.category_id.copy()

del df_train['category_id']
mystem = Mystem() 

rus_stop = stopwords.words("russian")



def stem(text):

    tokens = mystem.lemmatize(text.lower())

    tokens = [token.strip() for token in tokens if token not in rus_stop and token != " " \

                      and token.strip() not in punctuation]

    return ' '.join(tokens)
def tfidf_features(train, test):

    tfidf = TfidfVectorizer(tokenizer=lambda text: stem(text).split(' '), analyzer='word', ngram_range=(1,2))

 

    train_data.append(tfidf.fit_transform(train))

    test_data.append(tfidf.transform(test))
train_data = []

test_data = []

for column in ['title', 'description']:

    tfidf_features(df_train[column], df_test[column])
log_price_train = csr_matrix(normalize(df_train['price'].apply(lambda price: np.log1p(price)).values.reshape(-1, 1)))

log_price_test = csr_matrix(normalize(df_test['price'].apply(lambda price: np.log1p(price)).values.reshape(-1, 1)))
train_data.append(log_price_train)

test_data.append(log_price_test)
train_data_all = hstack(train_data)

test_data_all = hstack(test_data)
X_train, X_valid, y_train, y_valid = train_test_split(train_data_all, train_target, test_size=0.2, random_state=42)
%%time

svc = LinearSVC(C=1, random_state=42)

svc.fit(X_train, y_train)
prediction = svc.predict(X_valid)

acc = accuracy_score(prediction, y_valid)
print('Accuracy = {0:.4f}'.format(acc))
gc.collect()
category = pd.read_csv('../input/category.csv', index_col='category_id')
max_depth = max([name.count('|')  for name in category.name]) + 1



def subcat(cat_list):

    while len(cat_list) != max_depth:

        cat_list.append(' '.join(cat_list))

    return cat_list
category = pd.DataFrame(category['name'].apply(lambda s: s.split('|')).apply(subcat).tolist())

tree = {i: category[i].to_dict() for i in range(max_depth)}
for subcat_i, mask in tree.items():

    acc = accuracy_score(pd.DataFrame(prediction).replace(mask).values.reshape(-1,1), pd.DataFrame(y_valid).replace(mask).values.reshape(-1,1))

    print('Уровень = {0}, Точность = {1:.3f}'.format(subcat_i, acc))
svc_all = LinearSVC(C=1, random_state=42)

svc_all.fit(train_data_all, train_target)
df_test = df_test.drop(['title', 'description', 'price'], axis=1)

df_test['category_id'] = svc_all.predict(test_data_all)

df_test.to_csv('submission.csv')