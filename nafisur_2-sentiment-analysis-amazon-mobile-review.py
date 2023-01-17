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


#import pandas_profiling

%matplotlib inline
raw_dataset=pd.read_csv('../input/Amazon_Unlocked_Mobile.csv')
raw_dataset.head()
df=raw_dataset
df.info()
#pandas_profiling.ProfileReport(df)
df.describe()
df=df[['Reviews','Rating']]

df.head()
df.info()
df=df.dropna()
df.info()
df=df[df['Rating']!=3]
df.info()
df=df.reset_index(drop=True)
df.info()
df['sentiment']=np.where(df['Rating'] > 3, 1, 0)
df.head()
df.tail()
Cstopwords=set(stopwords.words('english')+list(punctuation))
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
def clean_review(review_column):
    review_corpus=[]
    for i in range(0,len(review_column)):
        review=review_column[i]
        #review=BeautifulSoup(review,'lxml').text
        review=re.sub('[^a-zA-Z]',' ',review)
        review=str(review).lower()
        review=word_tokenize(review)
        #review=[stemmer.stem(w) for w in review if w not in Cstopwords]
        review=[lemma.lemmatize(w) for w in review ]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus
review_column=df['Reviews']
review_corpus=clean_review(review_column)
df['clean_review']=review_corpus
df.tail(20)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=20000,min_df=5,ngram_range=(1,2))
X1=cv.fit_transform(df['clean_review'])
X1.shape
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(min_df=5, max_df=0.95, max_features = 20000, ngram_range = ( 1, 2 ),
                              sublinear_tf = True)
tfidf=tfidf.fit(df['clean_review'])
X2=tfidf.transform(df['clean_review'])
X2.shape
y=df['sentiment'].values
y.shape
X=X2 #X1 for bag of words model and X2 for Tfidf model
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# average positive reviews in train and test
print('mean positive review in train : {0:.3f}'.format(np.mean(y_train)))
print('mean positive review in test : {0:.3f}'.format(np.mean(y_test)))
from sklearn.linear_model import LogisticRegression as lr
model_lr=lr(random_state=0)
# %%time
# from sklearn.model_selection import GridSearchCV
# parameters = {'C':[0.5,1.0, 10.0], 'penalty' : ['l1','l2']}
# grid_search = GridSearchCV(estimator = model_lr,
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
from sklearn.naive_bayes import MultinomialNB
model_nb=MultinomialNB()
model_nb.fit(X_train,y_train)
y_pred_nb=model_nb.predict(X_test)
print('accuracy for Naive Bayes Classifier :',accuracy_score(y_test,y_pred_nb))
print('confusion matrix for Naive Bayes Classifier:\n',confusion_matrix(y_test,y_pred_nb))
print('F1 score for Logistic Regression :',f1_score(y_test,y_pred_nb))
print('Precision score for Logistic Regression :',precision_score(y_test,y_pred_nb))
print('recall score for Logistic Regression :',recall_score(y_test,y_pred_nb))
print('AUC: ', roc_auc_score(y_test, y_pred_nb))
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
%%time
model_rf=RandomForestClassifier()
model_rf.fit(X_train,y_train)
y_pred_rf=model_rf.predict(X_test)
print('accuracy for Random Forest Classifier :',accuracy_score(y_test,y_pred_rf))
print('confusion matrix for Random Forest Classifier:\n',confusion_matrix(y_test,y_pred_rf))
# get the feature names as numpy array
feature_names = np.array(cv.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_rf.feature_importances_.argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
