import os
print(os.listdir("../input"))
import nltk
import re
import numpy as np # linear algebra
import pandas as pd # data processing
import random
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
stemmer=SnowballStemmer('english')

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize

%matplotlib inline
raw_data_train=pd.read_csv('../input/labeledTrainData.tsv',sep='\t')
raw_data_test=pd.read_csv('../input/testData.tsv',sep='\t')
print(raw_data_train.shape)
print(raw_data_test.shape)
raw_data_train.info()
raw_data_test.info()
raw_data_train.head()
raw_data_test.head()
raw_data_train['review'][0]
from bs4 import BeautifulSoup
soup=BeautifulSoup(raw_data_train['review'][0],'lxml').text
soup
import re
re.sub('[^a-zA-Z]',' ',raw_data_train['review'][0])
from nltk.tokenize import word_tokenize
word_tokenize((raw_data_train['review'][0]).lower())[0:20]
from nltk.corpus import stopwords
from string import punctuation
Cstopwords=set(stopwords.words('english')+list(punctuation))
[w for w in word_tokenize(raw_data_train['review'][0]) if w not in Cstopwords][0:20]
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
stemmer=SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
from nltk.corpus import stopwords
from string import punctuation
Cstopwords=set(stopwords.words('english')+list(punctuation))
def clean_review(df):
    review_corpus=[]
    for i in range(0,len(df)):
        review=df[i]
        review=BeautifulSoup(review,'lxml').text
        review=re.sub('[^a-zA-Z]',' ',review)
        review=str(review).lower()
        review=word_tokenize(review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower()) if w not in Cstopwords]
        review=[lemma.lemmatize(w) for w in review ]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus
df=raw_data_train['review']
clean_train_review_corpus=clean_review(df)
clean_train_review_corpus[0]
df1=raw_data_test['review']
clean_test_review_corpus=clean_review(df1)
clean_test_review_corpus[0]
df=raw_data_train
df['clean_review']=clean_train_review_corpus
df.head()
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=20000,min_df=5,ngram_range=(1,2))
X1=cv.fit_transform(df['clean_review'])
X1.shape
train_data_features = X1.toarray()
y=df['sentiment'].values
y.shape
X=train_data_features
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# average positive reviews in train and test
print('mean positive review in train : {0:.3f}'.format(np.mean(y_train)))
print('mean positive review in test : {0:.3f}'.format(np.mean(y_test)))
from sklearn.naive_bayes import MultinomialNB
model_nb=MultinomialNB()
model_nb.fit(X_train,y_train)
y_pred_nb=model_nb.predict(X_test)
print('accuracy for Naive Bayes Classifier :',accuracy_score(y_test,y_pred_nb))
print('confusion matrix for Naive Bayes Classifier:\n',confusion_matrix(y_test,y_pred_nb))
# get the feature names as numpy array
feature_names = np.array(cv.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_nb.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier(random_state=0)
# %%time
# from sklearn.model_selection import GridSearchCV
# parameters = {'n_estimators':[100,200],'criterion':['entropy','gini'],
#               'min_samples_leaf':[2,5,7],
#               'max_depth':[5,6,7]
#                }
# grid_search = GridSearchCV(estimator = model_rf,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search = grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print('Best Accuracy :',best_accuracy)
# print('Best parameters:\n',best_parameters)
%%time
model_rf=RandomForestClassifier()
model_rf.fit(X_train,y_train)
y_pred_rf=model_rf.predict(X_test)
print('accuracy for Random Forest Classifier :',accuracy_score(y_test,y_pred_rf))
print('confusion matrix for Random Forest Classifier:\n',confusion_matrix(y_test,y_pred_rf))
from sklearn.linear_model import LogisticRegression as lr
model_lr=lr(random_state=0)
%%time
model_lr=lr(penalty='l2',C=1.0,random_state=0)
model_lr.fit(X_train,y_train)
y_pred_lr=model_lr.predict(X_test)
print('accuracy for Logistic Regression :',accuracy_score(y_test,y_pred_lr))
print('confusion matrix for Logistic Regression:\n',confusion_matrix(y_test,y_pred_lr))
print('F1 score for Logistic Regression :',f1_score(y_test,y_pred_lr))
print('Precision score for Logistic Regression :',precision_score(y_test,y_pred_lr))
print('recall score for Logistic Regression :',recall_score(y_test,y_pred_lr))
print('AUC: ', roc_auc_score(y_test, y_pred_lr))
# get the feature names as numpy array
feature_names = np.array(cv.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_lr.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
