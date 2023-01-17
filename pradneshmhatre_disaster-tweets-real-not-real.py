%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec

from tqdm import tqdm

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(train.info())

print('-'*50)

print(test.info())
print('Train data stats : \n',train.describe())

print('\n\nTest data stats : \n',test.describe())
def text_clean(text):

    cleanString=re.sub(r'http\S+', '', text)

    cleanString=re.sub(r'@\S+', '', cleanString)

    cleanString = re.sub('[^A-Za-z]+',' ', cleanString)

    return cleanString.lower()





def append_location(location,text):

    string=''

    if (not isinstance(location, str)):

        string=text

    else:

        string=string+text+' '+location 

    print(string)    

    return string    
train['cleaned_text'] = list(map(text_clean, train['text'].values))

train['cleaned_text_loc'] = list(map(append_location,train['location'], train['cleaned_text']))

train['cleaned_text_loc']=list(map(lambda x: x.lower(),train['cleaned_text_loc'].values))



test['cleaned_text'] = list(map(text_clean, test['text'].values))

test['cleaned_text_loc'] = list(map(append_location,test['location'], test['cleaned_text']))

test['cleaned_text_loc']=list(map(lambda x: x.lower(),test['cleaned_text_loc'].values))
train=train.drop(columns=['text','location','cleaned_text'])

train=train.fillna('')

train['keyword']=list(map(lambda x: x.replace('%20',' '), train['keyword']))

unique_keywords_train=list(train['keyword'].unique())

unique_keywords_train.remove('')



test=test.drop(columns=['text','location','cleaned_text'])

test=test.fillna('')

test['keyword']=list(map(lambda x: x.replace('%20',' '), test['keyword']))

unique_keywords_test=list(train['keyword'].unique())

unique_keywords_test.remove('')

unique_keywords= unique_keywords_train+unique_keywords_test
def fill_keywords(text,keyword):

    if keyword=='':

        for word in unique_keywords:

            if word in text:

                return word

    else:

        return keyword

    

train['keyword']=list(map(fill_keywords, train['cleaned_text_loc'], train['keyword']))

test['keyword']=list(map(fill_keywords, test['cleaned_text_loc'], test['keyword']))
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

train['cleaned_text_loc']=list(map(lambda x: ' '.join(e for e in x.split() if e.lower() not in stopwords), train['cleaned_text_loc'])) 

test['cleaned_text_loc']=list(map(lambda x: ' '.join(e for e in x.split() if e.lower() not in stopwords), test['cleaned_text_loc'])) 
from nltk.stem import PorterStemmer

ps = PorterStemmer() 

train['cleaned_text_loc']=list(map(lambda x: ps.stem(x), train['cleaned_text_loc']))

test['cleaned_text_loc']=list(map(lambda x: ps.stem(x), test['cleaned_text_loc']))
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

train['cleaned_text_loc']=list(map(lambda x: stemmer.stem(x), train['cleaned_text_loc']))

test['cleaned_text_loc']=list(map(lambda x: stemmer.stem(x), test['cleaned_text_loc']))
from numpy import array

keys=train['keyword'].astype(str)

print(keys)

label_encoder = LabelEncoder()

integer_encoded_tr = label_encoder.fit_transform(keys)

print(integer_encoded_tr)

train['label_encoder']= integer_encoded_tr
keys=test['keyword'].astype(str)

print(keys)

label_encoder = LabelEncoder()

integer_encoded_test = label_encoder.fit_transform(keys)

print(integer_encoded_test)

test['label_encoder']= integer_encoded_test
tf_idf_vect = TfidfVectorizer(min_df=0) 

tf_idf = tf_idf_vect.fit_transform(train['cleaned_text_loc'])

tf_idf_test = tf_idf_vect.transform(test['cleaned_text_loc'])
tf_idf_vect = TfidfVectorizer(min_df=0) 

tf_idf_1 = tf_idf_vect.fit_transform(train['keyword'].astype(str))

tf_idf_1_test = tf_idf_vect.transform(test['keyword'].astype(str))
from scipy.sparse import hstack

extra_features = hstack((tf_idf, tf_idf_1)).tocsr()
#Splitting the data into train and test

from sklearn.model_selection import train_test_split



X_tf=tf_idf

y_tf=train['target'].values

X_tr_tf, X_test_tf, y_tr_tf, y_test_tf = train_test_split(X_tf, y_tf, test_size=0.3, random_state=42)

print(X_tr_tf.shape,y_tr_tf.shape,X_test_tf.shape,y_test_tf.shape)
X_ef=extra_features

y_ef=train['target'].values

X_tr_ef, X_test_ef, y_tr_ef, y_test_ef = train_test_split(X_ef, y_ef, test_size=0.3, random_state=42)

print(X_tr_ef.shape,y_tr_ef.shape,X_test_ef.shape,y_test_ef.shape)


from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score



RANDOM_SEED = 0

rf_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=150,

                       n_jobs=None, oob_score=False, random_state=RANDOM_SEED,

                       verbose=0, warm_start=False)

lr_clf = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='warn', n_jobs=None, penalty='l2',

                   solver='warn', tol=0.0001, verbose=0,

                   warm_start=False, random_state=RANDOM_SEED)

knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=37, p=2,

                     weights='uniform')

svc_clf = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',

    kernel='rbf', max_iter=-1, probability=False, random_state=RANDOM_SEED,

    shrinking=True, tol=0.001, verbose=False)

nb_clf=MultinomialNB(alpha=1, class_prior=[0.1, 0.9], fit_prior=True)



classifier_array = [rf_clf, lr_clf, knn_clf, svc_clf, nb_clf]

labels = [clf.__class__.__name__ for clf in classifier_array]



normal_accuracy = []

normal_std = []

bagging_accuracy = []

bagging_std = []





for clf in classifier_array:

    #cv_scores = cross_val_score(clf, X_tr_tf, y_tr_tf, cv=3, n_jobs=-1)

    #cv_scores = f1_score(clf, X_tr_tf, y_tr_tf,average='weighted')

    clf1=clf.fit(X_tr_tf,y_tr_tf)

    y_pred1 = clf1.predict(X_test_tf)

    cv_scores = f1_score(y_test_tf,y_pred1,average='binary')

    bagging_clf = BaggingClassifier(clf, max_samples=0.9,random_state=RANDOM_SEED)

    #bagging_scores = cross_val_score(bagging_clf, X_tr_tf, y_tr_tf, cv=3, n_jobs=-1)

    clf2=bagging_clf.fit(X_tr_tf,y_tr_tf)

    y_pred2 = clf2.predict(X_test_tf)

    bagging_scores = f1_score(y_test_tf,y_pred2,average='binary')

    

    normal_accuracy.append(np.round(cv_scores.mean(),4))

    normal_std.append(np.round(cv_scores.std(),4))

    

    bagging_accuracy.append(np.round(bagging_scores.mean(),4))

    bagging_std.append(np.round(bagging_scores.std(),4))

    

    print("Accuracy: %0.4f (+/- %0.4f) [Normal %s]" % (cv_scores.mean(), cv_scores.std(), clf.__class__.__name__))

    print("Accuracy: %0.4f (+/- %0.4f) [Bagging %s]\n" % (bagging_scores.mean(), bagging_scores.std(), clf.__class__.__name__))



from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from mlxtend.classifier import EnsembleVoteClassifier

from xgboost import XGBClassifier



ada_boost = AdaBoostClassifier(n_estimators=5)

grad_boost = GradientBoostingClassifier(n_estimators=10)

xgb_boost = XGBClassifier(max_depth=5, learning_rate=0.001)



ensemble_clf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting='hard')



boosting_labels = ['Ada Boost', 'Gradient Boost', 'XG Boost', 'Ensemble']



for clf, label in zip([ada_boost, grad_boost, xgb_boost, ensemble_clf], boosting_labels):

    scores = cross_val_score(clf, X_tr_tf, y_tr_tf, cv=3, scoring='accuracy')

    print("Accuracy: {0:.3f}, Variance: (+/-) {1:.3f} [{2}]".format(scores.mean(), scores.std(), label))

#!pip install --upgrade scikit-learn
'''



from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import cross_val_score

RANDOM_SEED = 0



rf_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=150,

                       n_jobs=None, oob_score=False, random_state=RANDOM_SEED,

                       verbose=0, warm_start=False)

lr_clf = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='warn', n_jobs=None, penalty='l2',

                   solver='warn', tol=0.0001, verbose=0,

                   warm_start=False, random_state=RANDOM_SEED)

knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=37, p=2,

                     weights='uniform')

svc_clf = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3,

    kernel='rbf', max_iter=-1, probability=False, random_state=RANDOM_SEED,

    shrinking=True, tol=0.001, verbose=False)

nb_clf = MultinomialNB(alpha=1, class_prior=[0.1, 0.9], fit_prior=True)



lr = LogisticRegression(random_state=RANDOM_SEED) # meta classifier

sclf = StackingClassifier(estimators=[rf_clf, lr_clf, knn_clf, svc_clf, nb_clf])

classifier_array = [rf_clf, lr_clf, knn_clf, svc_clf, nb_clf, sclf]

labels = [clf.__class__.__name__ for clf in classifier_array]



acc_list = []

var_list = []



for clf, label in zip(classifier_array, labels):

    cv_scores = cross_val_score(clf, X_tf, y_tf, cv=3, scoring='accuracy')

    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (cv_scores.mean(), cv_scores.std(), label))

    acc_list.append(np.round(cv_scores.mean(),4))

    var_list.append(np.round(cv_scores.std(),4))

    #print("Accuracy: {} (+/- {}) [{}]".format(np.round(scores.mean(),4), np.round(scores.std(),4), label))

    

'''    
'''    

from sklearn.ensemble import StackingClassifier

clf=StackingClassifier(estimators=[('rf_clf',RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=150,

                       n_jobs=None, oob_score=False, random_state=RANDOM_SEED,

                       verbose=0, warm_start=False)), ('lr_clf',LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,n_jobs=None, penalty='l2', tol=0.0001, verbose=0,

                   warm_start=False, random_state=RANDOM_SEED)), ('knn_clf',KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=37, p=2,

                     weights='uniform')), ('svc_clf', SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3,kernel='rbf', max_iter=-1, probability=False, random_state=RANDOM_SEED,shrinking=True, tol=0.001, verbose=False))]).fit(X_tf, y_tf)

                     

'''                         
#test['target']=clf.predict(tf_idf_test)
#test_csv=test
#test_csv=test_csv.drop(columns=['keyword','cleaned_text_loc','label_encoder'])
#test_csv.to_csv('submission.csv', index=False)