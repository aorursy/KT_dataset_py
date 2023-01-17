# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn
import re
import os
print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read the csv file
dataset= pd.read_csv(r'''../input/social_media_clean_text.csv''')
drop_index= dataset[dataset['class_label']==2].index
dataset = dataset.drop(index=drop_index)

#cleaning the datato remove URLs and any other character other than texts
dataset["text"] = dataset["text"].apply(lambda elem : re.sub('[^a-zA-Z]', " ", elem ))
dataset["text"] = dataset["text"].apply(lambda elem : re.sub('r"http\S+', " ", elem ))
#converting all text to lowercase
dataset["text"]= dataset["text"].str.lower()

dataset.groupby("class_label").count()
# importing the nltk library and stopwords
import nltk

from nltk.corpus import stopwords

statements= dataset["text"]
statements;
#stemming the words will help the algorithm in grouping words which mean the same like "loved" and love
from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()

#removing the irrelevant words
from nltk.tokenize import word_tokenize
 #word_tokenize accepts string as an input, not file
stop_words=set(stopwords.words('english'))
#this list of list contains segmented words from each statements
total_list=[]
for statement in statements:
    filtered_sentence=[]
    word_tokens= word_tokenize(statement)
    for word in word_tokens: 
        if word not in stop_words: 
            word= ps.stem(word)
            filtered_sentence.append(word)
    total_list.append(filtered_sentence)          

new_tweet_list=[]
for tweet in total_list:
    tweet= ' '.join(tweet)
    new_tweet_list.append(tweet)


#creating a sparse matrix for the bag of words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv=CountVectorizer(max_features=10000)
#X=cv.fit_transform(new_tweet_list).toarray()
#X

#using tf-idf model(term frequency, inverse document frequency)
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb')
X= tv.fit_transform(new_tweet_list).toarray()
X =pd.DataFrame(X)
X
#Truncated SVD like PCA is used for dimesionality reduction of sparse matrices
from sklearn.decomposition import TruncatedSVD 
svd= TruncatedSVD(n_components= 100, n_iter= 5)
svd.fit(X)
#converting the target data into a data frame 
y= dataset.iloc[:, 2]
y=pd.DataFrame(y)

#splitting the dataset into training and test values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
#using Logistic Regression to classify the tweets 
#classifier = LogisticRegression()
#lg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
#grid_search = GridSearchCV(estimator= classifier,param_grid= lg_params, cv=5,  n_jobs= -1)
#from sklearn import svm, datasets
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svc = svm.SVC()
#grid_search = GridSearchCV(estimator= svc, param_grid= parameters, cv=5, n_jobs= -1)

from sklearn.ensemble import RandomForestClassifier
parameters = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
classifier= RandomForestClassifier()
grid_search= GridSearchCV(estimator=classifier, param_grid=parameters, cv= 5)
y_train =y_train.values


grid_search= grid_search.fit(X_train, y_train.ravel())
y_pred= grid_search.predict(X_test)
labels = ['Not relevant', 'Relevant']
print(classification_report(y_test, y_pred, target_names=labels))