! pip install mglearn #install mglearn#

import os

from sklearn.pipeline import make_pipeline # import a function for making an aggregation function

from sklearn.model_selection import GridSearchCV # this function can tune the parameter for finding the best one

from sklearn.metrics import confusion_matrix # evaluation

from sklearn.linear_model import LogisticRegression # machine learning method

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer #import two function to clean data and prepare data for trianing and testing

from sklearn.model_selection import train_test_split # split dataset into train and test function

import mglearn

import numpy as np #an wildly used package

import pandas as pd # an wildly used function

import seaborn as sn # an extension for matplotlib

import matplotlib as mpl

import matplotlib.pyplot as plt

import nltk # natural language package

from nltk.corpus import stopwords 

from string import punctuation

from gensim.sklearn_api import W2VTransformer

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS
data = pd.read_csv("../input/SPAM text message 20170820 - Data.csv")
def strip_punctuation(s):

    return ''.join(c for c in s if c not in punctuation)

# this is the function to remove punction
def digit_punctuation(s):

    return ''.join([i for i in s if not i.isdigit()])

# this is the function to remove digits
def cleanupDoc(s):

 stopset = set(stopwords.words('english'))

 tokens = nltk.word_tokenize(s)

 cleanup = [token.lower() for token in tokens if token.lower() not in stopset and  len(token)>2]

 return cleanup

# this is the function to remove stop words 
texts = []

labels = []

for i, label in enumerate(data['Category']):

    strip_no_punt=strip_punctuation(data['Message'][i].lower())

    sttrip_nodigit=digit_punctuation(strip_no_punt)

    texts.append(cleanupDoc(sttrip_nodigit))

    if label == 'ham':

        labels.append(0)

    else:

        labels.append(1)

# this is the pipeline from remove punction, digits, to remove stop words and tonkenize
wordcloud = WordCloud(background_color='white',max_words=200,max_font_size=200,width=1000, height=860, random_state=42).generate(str(texts))

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=1000)

# this is the commend to create word cloud
model = W2VTransformer(size=10, min_count=1, seed=1)

wordvecs = model.fit(texts)

# this is the commend to create word2vector
print(texts[1])

print (wordvecs.transform(texts[1]))

# print the first line of texts and its corresponding w2v array
w2varray = []

for i in range(len(texts)):

    transformob=np.sum(wordvecs.transform(texts[i]),axis=0)

    w2varray.append(transformob)

# create the whole w2v dataset
X_train, X_test, y_train, y_test = train_test_split(w2varray , labels, test_size=0.1, random_state=42)

# split train and test dataset
logreg = LogisticRegression() # call M.L. methods 

param_grid = {'C': [0.01, 0.1, 1, 10, 100]} # call M.L. parameters

grid = GridSearchCV(logreg, param_grid, cv=5) # set cross validation for fining appropriate parameter

logreg_train = grid.fit(X_train, y_train) # fit training dataset

pred_logreg = logreg_train.predict(X_test) # predict dataset

confusion = confusion_matrix(y_test, pred_logreg) # using confusion matrix to evaluate

df_cm = pd.DataFrame(confusion, ['ham','spam'],['ham','spam']) 

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}) #create heatmap
texts2 = []

labels = []

for i, label in enumerate(data['Category']):

    strip_no_punt=strip_punctuation(data['Message'][i].lower())

    sttrip_nodigit=digit_punctuation(strip_no_punt)

    texts2.append(sttrip_nodigit)

    if label == 'ham':

        labels.append(0)

    else:

        labels.append(1)
X_train, X_test, y_train, y_test = train_test_split(texts2 , labels, test_size=0.1, random_state=42)
vect = CountVectorizer().fit(X_train)

X_train_cv_without_stop_word = vect.transform(X_train)

X_test_cv_without_stop_word = vect.transform(X_test)
logreg = LogisticRegression()

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=5)

logreg_train = grid.fit(X_train_cv_without_stop_word, y_train)

pred_logreg = logreg_train.predict(X_test_cv_without_stop_word )

confusion = confusion_matrix(y_test, pred_logreg)

df_cm = pd.DataFrame(confusion, ['ham','spam'],['ham','spam'])

sn.set(font_scale=1.4)#for label size

sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
logreg = LogisticRegression()

pipe = make_pipeline(CountVectorizer(), logreg)

param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(pipe, param_grid, cv=5)

logreg_train = grid.fit(X_train, y_train)

pred_logreg = logreg_train.predict(X_test)

confusion = confusion_matrix(y_test, pred_logreg)

df_cm = pd.DataFrame(confusion, ['ham','spam'],['ham','spam'])

sn.set(font_scale=1.4)#for label size

sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})

# alternative codes
max_value = X_train_cv_without_stop_word.max(axis=0).toarray().ravel()

sorted_by_tfidf = max_value.argsort()



feature_names = np.array(vect.get_feature_names())



print("features with lowest cv_without_stop_word")

print(feature_names[sorted_by_tfidf[:20]], '\n')



print("features with highest cv_without_stop_word")

print(feature_names[sorted_by_tfidf[-20:]])
mglearn.tools.visualize_coefficients(grid.best_estimator_.coef_, feature_names, n_top_features=20)

plt.title("CountVectorizer_without_stop_word-cofficient")
vect = TfidfVectorizer().fit(X_train)

X_train_tf_without_stop_word = vect.transform(X_train)

X_test_tf_without_stop_word = vect.transform(X_test)
logreg = LogisticRegression()

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=5)

logreg_train = grid.fit(X_train_tf_without_stop_word, y_train)

pred_logreg = logreg_train.predict(X_test_tf_without_stop_word)

confusion = confusion_matrix(y_test, pred_logreg)

df_cm = pd.DataFrame(confusion, ['ham','spam'],['ham','spam'])

sn.set(font_scale=1.4)#for label size

sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
max_value = X_train_tf_without_stop_word.max(axis=0).toarray().ravel()

sorted_by_tfidf = max_value.argsort()



feature_names = np.array(vect.get_feature_names())



print("features with lowest tf_without_stop_word")

print(feature_names[sorted_by_tfidf[:20]], '\n')



print("features with highest tf_without_stop_word")

print(feature_names[sorted_by_tfidf[-20:]])
mglearn.tools.visualize_coefficients(grid.best_estimator_.coef_, feature_names, n_top_features=20)

plt.title("TfidfVectorizer_without_stop_word-cofficient")
vect = TfidfVectorizer(stop_words='english',min_df=3).fit(X_train)

X_train_tf_with_stop_word_3 = vect.transform(X_train)

X_test_tf_with_stop_word_3 = vect.transform(X_test)
logreg = LogisticRegression()

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=5)

logreg_train = grid.fit(X_train_tf_with_stop_word_3, y_train)

pred_logreg = logreg_train.predict(X_test_tf_with_stop_word_3)

confusion = confusion_matrix(y_test, pred_logreg)

df_cm = pd.DataFrame(confusion, ['ham','spam'],['ham','spam'])

sn.set(font_scale=1.4)#for label size

sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
max_value = X_train_tf_with_stop_word_3 .max(axis=0).toarray().ravel()

sorted_by_tfidf = max_value.argsort()



feature_names = np.array(vect.get_feature_names())



print("features with lowest tf_without_stop_word_3")

print(feature_names[sorted_by_tfidf[:20]], '\n')



print("features with highest tf_without_stop_word_3")

print(feature_names[sorted_by_tfidf[-20:]])
mglearn.tools.visualize_coefficients(grid.best_estimator_.coef_, feature_names, n_top_features=20)

plt.title("TfidfVectorizer_with_stop_word-cofficient_3")