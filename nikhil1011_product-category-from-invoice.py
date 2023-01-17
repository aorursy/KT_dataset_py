# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/product-category-from-invoice/Dataset"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/product-category-from-invoice/Dataset/Train.csv')

test=pd.read_csv('../input/product-category-from-invoice/Dataset/Test.csv')
train.shape
train.info()
test.shape
test.info()
train.describe()
train.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

train['Product_Category'].value_counts().plot.bar()

plt.show()
train['Product_Category'].value_counts()
print(' # of unique classes ',train['Product_Category'].nunique())
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

train['GL_Code'].value_counts().plot.bar()

plt.show()
import seaborn as sns

plt.figure(1)

sns.distplot(train['Inv_Amt']);
from sklearn.preprocessing import LabelEncoder

for col in ['GL_Code', 'Vendor_Code']:

    

    le = LabelEncoder()

    le.fit(list(train[col]) + list(test[col]))

    train[col] = le.transform(train[col])

    test[col] = le.transform(test[col])
train.head()
# printing some random description

sent_0 = train['Item_Description'].values[0]

print(sent_0)

print("="*50)



sent_1000 = train['Item_Description'].values[1000]

print(sent_1000)

print("="*50)
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
from tqdm import tqdm

import re

preprocessed_text = []

# tqdm is for printing the status bar

for sentance in tqdm(train['Item_Description'].values):

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

    preprocessed_text.append(sentance.strip())
train['Item_Description_Preprocessed']=preprocessed_text

train.drop('Item_Description',axis=1,inplace=True)
train.head()
import string

import nltk

from nltk.corpus import stopwords 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)

tf_idf_vect.fit(preprocessed_text)

print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])

print('='*50)



final_tf_idf = tf_idf_vect.transform(preprocessed_text)

final_tf_idf_test = tf_idf_vect.transform(test['Item_Description'])

print("the type of count vectorizer ",type(final_tf_idf))

print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])
target = LabelEncoder()

y_endoded = target.fit_transform(train['Product_Category'])
X_train, X_valid, y_train, y_valid = train_test_split(final_tf_idf,y_endoded, test_size=0.3, random_state=1)
import xgboost as xgb

param = {}

# use softmax multi-class classification

param['objective'] = 'multi:softprob'

param['eta'] = 0.1

param['max_depth'] = 6

param['silent'] = 1

param['nthread'] = 4

param['num_class'] = len(target.classes_)

param['eval_metric'] = ['mlogloss']

param['seed'] = 1



dtrain = xgb.DMatrix(X_train, label=y_train)

dvalid = xgb.DMatrix(X_valid, label=y_valid)



evallist = [(dtrain, 'train'), (dvalid, 'eval')]



clf = xgb.train(param, dtrain, 100, evallist, verbose_eval=50)
y_pred_valid = clf.predict(dvalid)



print("Accuracy : ",accuracy_score(y_valid, np.argmax(y_pred_valid, axis=1)))
dtest = xgb.DMatrix(final_tf_idf_test)

y_test_pred = clf.predict(dtest)
output = test[['Inv_Id']].copy()

output['Product_Category'] = target.inverse_transform(np.argmax(y_test_pred, axis=1))
output['Product_Category'].nunique()
num_splits = 5

skf = StratifiedKFold(n_splits= num_splits, random_state=1, shuffle=True)
y_test_pred = np.zeros((test.shape[0], len(target.classes_)))

print(y_test_pred.shape)

y_valid_scores = []

X = train['Item_Description_Preprocessed']

fold_cnt = 1

dtest = xgb.DMatrix(final_tf_idf_test)



for train_index, test_index in skf.split(X, y_endoded):

    print("\nFOLD .... ",fold_cnt)

    fold_cnt += 1

    

    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]

    y_train, y_valid = y_endoded[train_index], y_endoded[test_index]

    

    X_train_bow = tf_idf_vect.transform(X_train)

    X_valid_bow = tf_idf_vect.transform(X_valid)

    

    dtrain = xgb.DMatrix(X_train_bow, label=y_train)

    dvalid = xgb.DMatrix(X_valid_bow, label=y_valid)



    evallist = [(dtrain, 'train'), (dvalid, 'eval')]



    clf = xgb.train(param, dtrain, 100, evallist, verbose_eval=50)

    #Predict validation data

    y_pred_valid = clf.predict(dvalid)

    y_valid_scores.append(accuracy_score(y_valid, np.argmax(y_pred_valid, axis=1)))

    

    #Predict test data

    y_pred = clf.predict(dtest)

    

    y_test_pred += y_pred
print("Validation Scores :", y_valid_scores)

print("Average Score: ",np.round(np.mean(y_valid_scores),3))
y_test_pred /= num_splits
output['Product_Category'] = target.inverse_transform(np.argmax(y_test_pred, axis=1))
output['Product_Category'].nunique()
output.to_csv("submission.csv", index=False)
output.head()