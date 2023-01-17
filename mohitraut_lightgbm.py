from sklearn.pipeline import Pipeline

import lightgbm as lgb 

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
train_merge = train_var.merge(train_text,left_on="ID",right_on="ID")

train_merge.head(5)



test_merge = test_var.merge(test_text,left_on="ID",right_on="ID")

test_merge.head(5)

import missingno as msno

%matplotlib inline

msno.bar(train_merge)
import missingno as msno

%matplotlib inline

msno.bar(test_merge)
import regex as re

def textClean(text):

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = text.lower().split()

    stops = {'so', 'his', 't', 'y', 'ours', 'herself', 

             'your', 'all', 'some', 'they', 'i', 'of', 'didn', 

             'them', 'when', 'will', 'that', 'its', 'because', 

             'while', 'those', 'my', 'don', 'again', 'her', 'if',

             'further', 'now', 'does', 'against', 'won', 'same', 

             'a', 'during', 'who', 'here', 'have', 'in', 'being', 

             'it', 'other', 'once', 'itself', 'hers', 'after', 're',

             'just', 'their', 'himself', 'theirs', 'whom', 'then', 'd', 

             'out', 'm', 'mustn', 'where', 'below', 'about', 'isn',

             'shouldn', 'wouldn', 'these', 'me', 'to', 'doesn', 'into',

             'the', 'until', 'she', 'am', 'under', 'how', 'yourself',

             'couldn', 'ma', 'up', 'than', 'from', 'themselves', 'yourselves',

             'off', 'above', 'yours', 'having', 'mightn', 'needn', 'on', 

             'too', 'there', 'an', 'and', 'down', 'ourselves', 'each',

             'hadn', 'ain', 'such', 've', 'did', 'be', 'or', 'aren', 'he', 

             'should', 'for', 'both', 'doing', 'this', 'through', 'do', 'had',

             'own', 'but', 'were', 'over', 'not', 'are', 'few', 'by', 

             'been', 'most', 'no', 'as', 'was', 'what', 's', 'is', 'you', 

             'shan', 'between', 'wasn', 'has', 'more', 'him', 'nor',

             'can', 'why', 'any', 'at', 'myself', 'very', 'with', 'we', 

             'which', 'hasn', 'weren', 'haven', 'our', 'll', 'only',

             'o', 'before'}

    ## I ketp getting errors on importing the stopwords and I have no clue why

    #stops = set(stopwords.words("English"))

    text = [w for w in text if not w in stops]    

    text = " ".join(text)

    text = text.replace("."," ").replace(","," ")

    return(text)
trainText = []

for it in train_merge['Text']:

    newT = textClean(it)

    trainText.append(newT)

testText = []

for it in test_merge['Text']:

    newT = textClean(it)

    testText.append(newT)
train_merge['Clean_text']=trainText

test_merge['Clean_text']=testText



train_merge=train_merge.drop('ID',axis=1)

train_merge=train_merge.drop('Text',axis=1)
test_merge=test_merge.drop(['ID','Text'],axis=1)

test_merge.head(5)
from sklearn.model_selection import train_test_split

train ,test = train_test_split(train_merge,test_size=0.2) 

np.random.seed(0)

train.head(5)
x_train = train['Clean_text'].values

x_test = test['Clean_text'].values

y_train = train['Class'].values

y_test = test['Class'].values
def my_tokenizer(X):

    newlist = []

    for alist in X:

        newlist.append(alist[0].split(' '))

    return newlist



maxFeats=500 



cvec = CountVectorizer(min_df=5, ngram_range=(1,3), max_features=maxFeats, 

                       strip_accents='unicode',

                       lowercase =True, analyzer='word', token_pattern=r'\w+',

                       stop_words = 'english',tokenizer=my_tokenizer)

tfidf = TfidfVectorizer(min_df=5, max_features=maxFeats, ngram_range=(1,3),

                        strip_accents='unicode',

                        lowercase =True, analyzer='word', token_pattern=r'\w+',

                        use_idf=True, smooth_idf=True, sublinear_tf=True, 

                        stop_words = 'english')
y_test=y_test-1

y_train=y_train-1



train_tran=tfidf.fit_transform(x_train)

test_tran=tfidf.fit_transform(x_test)
d_train = lgb.Dataset(train_tran, label=y_train)

d_val = lgb.Dataset(test_tran, label=y_test)
parms = {'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'multiclass',

    'num_class': 9,

    'metric': {'multi_logloss'},

    'learning_rate': 0.05, 

    'max_depth': 5,

    'num_iterations': 400, 

    'num_leaves': 95, 

    'min_data_in_leaf': 60, 

    'lambda_l1': 1.0,

    'feature_fraction': 0.8, 

    'bagging_fraction': 0.8, 

    'bagging_freq': 5}



rnds = 500

mod = lgb.train(parms, train_set=d_train, num_boost_round=rnds,

               valid_sets=[d_val], valid_names=['dval'], verbose_eval=20,early_stopping_rounds=20)
import matplotlib.pyplot as plt

%matplotlib inline

lgb.plot_importance(mod, max_num_features=30, figsize=(14,10))
test_data=test_merge['Clean_text']

test_data=tfidf.fit_transform(test_data)
pred = mod.predict(test_data)
pred1=pred

pred1=(pred1 == pred1.max(axis=1)[:,None]).astype(int)
submission=pd.DataFrame(pred1)

submission['ID']=test_var['ID']

submission.columns=["Class1","Class2","Class3","Class4","Class5","Class6","Class7","Class8","Class9","ID"]

submission.head(5)
submission.to_csv('submission1.csv', index=False)
import pandas as pd

temp=pd.read_csv("../input/submissionFile")
temp.to_csv("../output/result.csv",index=False)