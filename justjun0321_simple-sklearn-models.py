# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Set up matplotlib style 

plt.style.use('ggplot')



# And libraries for data transformation

from string import punctuation

from nltk.tokenize import word_tokenize 

from nltk.corpus import stopwords
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(10)
train.target = train.target.apply(lambda x: 1 if x>0.45 else 0)
train_sub = train.drop(['id','comment_text'],axis=1)
sns.set(rc={'figure.figsize':(8,8)})



matrix = train_sub.corr()

sns.heatmap(matrix)
train.target.value_counts()
train.shape
train.fillna(0,inplace=True)
stop_words = set(stopwords.words('english')) 



train['comment_text'] = train.comment_text.apply(lambda x: x.lower())



train['cleaned_comment'] = train.comment_text.apply(lambda x: word_tokenize(x))



train['cleaned_comment'] = train.cleaned_comment.apply(lambda x: [w for w in x if w not in stop_words])



train['cleaned_comment'] = train.cleaned_comment.apply(lambda x: ' '.join(x))



train.drop('comment_text',axis=1,inplace=True)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



y = train.target



X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33,random_state=53)



# Initialize a CountVectorizer object: count_vectorizer

count_vectorizer = CountVectorizer(stop_words="english")



# Transform the training data using only the 'text' column values: count_train 

count_train = count_vectorizer.fit_transform(X_train["cleaned_comment"])



y_train = np.asarray(y_train.values)



# Pick up the most effective words

ch2 = SelectKBest(chi2, k = 300)



X_new = ch2.fit_transform(count_train, y_train)



# Transform the test data using only the 'text' column values: count_test 

count_test = count_vectorizer.transform(X_test["cleaned_comment"])



X_test_new = ch2.transform(X=count_test)
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

# Fit the classifier to the training data

clf.fit(X_new, y_train)



# Create the predicted tags: pred

pred = clf.predict(X_test_new)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test, pred)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, pred)

print('F score is:',f1)
sns.heatmap(metrics.confusion_matrix(pred,y_test),annot=True,fmt='2.0f')
from itertools import compress



features = count_vectorizer.get_feature_names()

mask = ch2.get_support()

features = list(compress(features, mask))

importances = clf.feature_importances_

indices = np.argsort(importances)



sns.set(rc={'figure.figsize':(11,50)})



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
import eli5

from eli5.sklearn import PermutationImportance



X_test_new = X_test_new.toarray()



perm = PermutationImportance(clf, random_state=1).fit(X_test_new, y_test)

eli5.show_weights(perm, feature_names = features)
features = count_vectorizer.get_feature_names()

mask = ch2.get_support()

features = list(compress(features, mask))



train_df = pd.DataFrame(X_new.todense(), columns=features)



test_df = pd.DataFrame(X_test_new.todense(), columns=features)
train_data = train_df.join(X_train.drop(['target','cleaned_comment','created_date','publication_id','parent_id', 'article_id,rating'],axis=1))



test_data = test_df.join(X_test.drop(['target','cleaned_comment','created_date','publication_id','parent_id', 'article_id,rating'],axis=1))
clf = RandomForestClassifier()

# Fit the classifier to the training data

clf.fit(train_data, y_train)



# Create the predicted tags: pred

pred = clf.predict(test_data)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test, pred)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, pred)

print('F score is:',f1)
test['comment_text'] = test.comment_text.apply(lambda x: x.lower())



test['cleaned_comment'] = test.comment_text.apply(lambda x: word_tokenize(x))



test['cleaned_comment'] = test.cleaned_comment.apply(lambda x: [w for w in x if w not in stop_words])



test['cleaned_comment'] = test.cleaned_comment.apply(lambda x: ' '.join(x))
test = test['cleaned_comment']
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(test)



test = ch2.transform(count_test)



prediction = clf.predict(test)
submission = pd.read_csv('../input/sample_submission.csv')



submission['prediction'] = prediction



submission.to_csv('submission.csv',index=False)