# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import string

import nltk

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.svm import LinearSVC

from sklearn import svm, linear_model

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from math import sqrt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.ensemble import VotingClassifier 

sns.set(color_codes=True)

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import random

from sklearn.metrics import accuracy_score

from collections import Counter

from sklearn.metrics import accuracy_score
review_data0 = pd.read_csv('../input/boardgamegeek-reviews/bgg-13m-reviews.csv', index_col=0)

review_data0.head()
review_data0.shape
review_data2=review_data0[~review_data0.comment.str.contains("NaN",na=True)]

review_data2.head()
review_data2.shape
review_data2.describe()
#plot histogram of ratings

num_bins = 70

n, bins, patches = plt.hist(review_data2.rating, num_bins, facecolor='green', alpha=0.9)



#plt.xticks(range(9000))

plt.title('Histogram of Ratings')

plt.xlabel('Ratings')

plt.ylabel('Count')

plt.show()
review_data2.head()

review_data3=review_data2.sample(n=30000)

review_data3.head()
review_data3.dtypes
review_data3.isna().sum()
review_data3['word_count']  = review_data3.comment.str.len()



num_bins = 70

n, bins, patches = plt.hist(review_data3.word_count, num_bins, facecolor='green', alpha=0.9)



#plt.xticks(range(9000))

plt.title('Histogram of Word Count')

plt.xlabel('Word Count')

plt.ylabel('Count')

plt.show()
#lowercase and remove punctuation

review_data3['cleaned'] = review_data3['comment'].str.lower().apply(lambda x:''.join([i for i in x if i not in string.punctuation]))



# stopword list to use

stopwords_list = stopwords.words('english')

stopwords_list.extend(('game','play','played','players','player','people','really','board','games','one','plays','cards','would')) 



stopwords_list[-10:]



#remove stopwords

review_data3['cleaned'] = review_data3['cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_list)]))

review_data3.head()
num_bins = 70

n, bins, patches = plt.hist(review_data3.rating, num_bins, facecolor='green', alpha=0.9)



#plt.xticks(range(9000))

plt.title('Histogram of Ratings')

plt.xlabel('Ratings')

plt.ylabel('Count')

plt.show()
Counter(" ".join(review_data3["cleaned"]).split()).most_common(50)[:50]
from wordcloud import WordCloud

from collections import Counter



neg = review_data3.loc[review_data3['rating'] < 3]

pos = review_data3.loc[review_data3['rating'] > 8]





words = Counter([w for w in " ".join(pos['cleaned']).split()])



wc = WordCloud(width=400, height=350,colormap='plasma',background_color='white').generate_from_frequencies(dict(words.most_common(100)))

plt.figure(figsize=(20,15))

plt.imshow(wc, interpolation='bilinear')

plt.title('Common Words in Positive Reviews', fontsize=20)

plt.axis('off');

plt.show()





words = Counter([w for w in " ".join(neg['cleaned']).split()])



wc = WordCloud(width=400, height=350,colormap='plasma',background_color='white').generate_from_frequencies(dict(words.most_common(100)))

plt.figure(figsize=(20,15))

plt.imshow(wc, interpolation='bilinear')

plt.title('Common Words in Negative Reviews', fontsize=20)

plt.axis('off');

plt.show()
print('Mean: ', review_data3.rating.mean())

print('Median: ', review_data3.rating.median())

print('Mode: ', review_data3.rating.mode())
def calc_rmse(errors, weights=None):

    n_errors = len(errors)

    if weights is None:

        result = sqrt(sum(error ** 2 for error in errors) / n_errors)

    else:

        result = sqrt(sum(weight * error ** 2 for weight, error in zip(weights, errors)) / sum(weights))

    return result



#if the score is far from mean (high or low scores), weight those reviews and ratings more when assessing model accuracy

def calc_weights(scores):

    peak = 6.851

    return tuple((10 ** (0.3556 * (peak - score))) if score < peak else (10 ** (0.2718 * (score - peak))) for score in scores)





def assess_model( model_name, test, predicted):

    error = test - predicted

    rmse = calc_rmse(error)

    mae = mean_absolute_error(test, predicted)

    weights = calc_weights(test)

    weighted_rmse = calc_rmse(error, weights = weights)

    

    

    print(model_name)

    print('RMSE:',rmse)

    print('Weighed RMSE:', weighted_rmse)

    print('MAE:', mae)
X_train, X_test, y_train, y_test = train_test_split(review_data3.cleaned, review_data3.rating, random_state=44,test_size=0.20)



model_nb = Pipeline([

    ('count_vectorizer', CountVectorizer(lowercase = True, stop_words = stopwords.words('english'))), 

    ('tfidf_transformer',  TfidfTransformer()), #weighs terms by importance to help with feature selection

    ('classifier', MultinomialNB()) ])

    

model_nb.fit(X_train,y_train.astype('int'))

labels = model_nb.predict(X_test)

mat = confusion_matrix(y_test.astype('int'), labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()



assess_model("Multinomial NB", y_test,labels)

acc = accuracy_score(y_test.astype('int'),labels, normalize=True) * float(100)

print('\n****Test accuracy is',(acc))
#Experimented with adding different numbers of n-grams, 1-2 seems to have best performance

model_nb2 = Pipeline([

    ('count_vectorizer', CountVectorizer( ngram_range=(1,2), lowercase = True, stop_words = stopwords.words('english'))), 

    ('tfidf_transformer',  TfidfTransformer()), #weighs terms by importance to help with feature selection

    ('classifier', MultinomialNB()) ])

    

model_nb2.fit(X_train,y_train.astype('int'))

labels = model_nb2.predict(X_test)

mat = confusion_matrix(y_test.astype('int'), labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()



assess_model("Multinomial NB n-grams 1-2", y_test,labels)

acc = accuracy_score(y_test.astype('int'),labels, normalize=True) * float(100)

print('\n****Test accuracy is',(acc))
# Convert the data in vector fpormate

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))

tf_idf_train = tf_idf_vect.fit_transform(X_train)

tf_idf_test = tf_idf_vect.transform(X_test)



alpha_range = list(np.arange(0,30,1))

len(alpha_range)
from sklearn.naive_bayes import MultinomialNB

y_train=y_train.astype('int')



alpha_scores=[]



for a in alpha_range:

    clf = MultinomialNB(alpha=a)

    scores = cross_val_score(clf, tf_idf_train, y_train, cv=5, scoring='accuracy')

    alpha_scores.append(scores.mean())

    print(a,scores.mean())
# Plot b/w misclassification error and CV mean score.

import matplotlib.pyplot as plt



MSE = [1 - x for x in alpha_scores]





optimal_alpha_bnb = alpha_range[MSE.index(min(MSE))]



# plot misclassification error vs alpha

plt.plot(alpha_range, MSE)



plt.xlabel('hyperparameter alpha')

plt.ylabel('Misclassification Error')

plt.show()
optimal_alpha_bnb
model_nb = Pipeline([

    ('count_vectorizer', CountVectorizer(lowercase = True, stop_words = stopwords.words('english'))), 

    ('tfidf_transformer',  TfidfTransformer()), #weighs terms by importance to help with feature selection

    ('classifier', MultinomialNB(alpha=optimal_alpha_bnb)) ])

    

model_nb.fit(X_train,y_train.astype('int'))

labels = model_nb.predict(X_test)

mat = confusion_matrix(y_test.astype('int'), labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()



assess_model("Multinomial NB", y_test,labels)

acc = accuracy_score(y_test.astype('int'),labels, normalize=True) * float(100)

print('\n****Test accuracy is',(acc))
model_svc = make_pipeline(TfidfVectorizer(ngram_range=(1,3)), svm.SVC(kernel="linear",probability=True))

model_svc.fit(X_train, y_train.astype('int'))

labels = model_svc.predict(X_test)



mat = confusion_matrix(y_test.astype('int'), labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()



assess_model("Linear SVC model", y_test,labels)

acc = accuracy_score(y_test.astype('int'),labels, normalize=True) * float(100)

print('\n****Test accuracy is',(acc))
review_data2.head()

rating1_subset = review_data2[review_data2['rating']==1] 

rating1_subset.head()



# Slect 100 sample that have rating =1

r1=rating1_subset.sample(2000)

r1.head()





rating2_subset = review_data2[review_data2['rating']==2] 

rating2_subset.head()

# Slect 100 sample that have rating =2

r2=rating2_subset.sample(2000)

r2.head()



rating3_subset = review_data2[review_data2['rating']==3] 

rating3_subset.head()

# Slect 100 sample that have rating =3

r3=rating3_subset.sample(2000)

r3.head()



rating4_subset = review_data2[review_data2['rating']==4] 

rating4_subset.head()

# Slect 100 sample that have rating =4

r4=rating4_subset.sample(2000)

r4.head()



rating5_subset = review_data2[review_data2['rating']==5] 

rating5_subset.head()

# Slect 100 sample that have rating =5

r5=rating5_subset.sample(2000)

r5.head()



rating6_subset = review_data2[review_data2['rating']==6] 

rating6_subset.head()

# Slect 100 sample that have rating =6

r6=rating6_subset.sample(2000)

r6.head()



rating7_subset = review_data2[review_data2['rating']==7] 

rating7_subset.head()

# Slect 100 sample that have rating =7

r7=rating7_subset.sample(2000)

r7.head()



rating8_subset = review_data2[review_data2['rating']==8] 

rating8_subset.head()

# Slect 100 sample that have rating =8

r8=rating8_subset.sample(2000)

r8.head()



rating9_subset = review_data2[review_data2['rating']==9] 

rating9_subset.head()

# Slect 100 sample that have rating=9

r9=rating9_subset.sample(2000)

r9.head()



rating10_subset = review_data2[review_data2['rating']==10] 

rating10_subset.head()

# Slect 100 sample that have rating=10

r10=rating10_subset.sample(2000)

r10.head()
review_balance=df = r1.append([r2, r3,r4,r5,r6,r7,r8,r9,r10])

review_balance.head()
review_balance.shape
#lowercase and remove punctuation

review_balance['cleaned'] = review_balance['comment'].str.lower().apply(lambda x:''.join([i for i in x if i not in string.punctuation]))



# stopword list to use

stopwords_list = stopwords.words('english')

stopwords_list.extend(('game','play','played','players','player','people','really','board','games','one','plays','cards','would')) 



stopwords_list[-10:]



#remove stopwords

review_balance['cleaned'] = review_balance['cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_list)]))

review_balance.head()
#plot histogram of ratings

num_bins = 70

n, bins, patches = plt.hist(review_balance.rating, num_bins, facecolor='green', alpha=0.9)



#plt.xticks(range(9000))

plt.title('Histogram of Ratings')

plt.xlabel('Ratings')

plt.ylabel('Count')

plt.show()
X_train1, X_test1, y_train1, y_test1 = train_test_split(review_balance.cleaned, review_balance.rating, test_size=0.20)

model_svc_balance = make_pipeline(TfidfVectorizer(ngram_range=(1,3)), svm.SVC(kernel="linear",probability=True))

model_svc_balance.fit(X_train1, y_train1.astype('int'))

labels = model_svc_balance.predict(X_test1)



mat = confusion_matrix(y_test1.astype('int'), labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()



assess_model("Linear SVC Balanced model", y_test1,labels)

acc = accuracy_score(y_test1.astype('int'),labels, normalize=True) * float(100)

print('\n****Test accuracy is',(acc))
X_train, X_test, y_train, y_test = train_test_split(review_data3.cleaned, review_data3.rating, test_size=0.20)

labels = model_svc_balance.predict(X_test)



mat = confusion_matrix(y_test.astype('int'), labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()



assess_model("Linear SVC model", y_test,labels)

acc = accuracy_score(y_test.astype('int'),labels, normalize=True) * float(100)

print('\n****Test accuracy of re-trained SVC is',(acc))


X_train, X_test, y_train, y_test = train_test_split(review_data3.cleaned, review_data3.rating, test_size=0.20)



Ensemble = VotingClassifier(estimators=[('model_svc_unbalance',model_svc), ('model_svc_balance', model_svc_balance )],

                        voting='soft',

                        weights=[3, 1])



Ensemble.fit(X_train,y_train.astype(int))





labels = Ensemble.predict(X_test)

mat = confusion_matrix(y_test.astype(int), labels)

ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()

assess_model("Ensemble model", y_test,labels)

acc = accuracy_score(y_test.astype('int'),labels, normalize=True) * float(100)

print('\n****Test accuracy of Ensemble SVC is',(acc))
X_train, X_test, y_train, y_test = train_test_split(review_data3.cleaned, review_data3.rating, test_size=0.20)



labels = model_svc.predict(X_test)

labels_2 = model_svc_balance.predict(X_test)





pred = pd.concat([pd.DataFrame(y_test).reset_index().rating,pd.Series(labels),pd.Series(labels_2)],axis=1)

pred.columns = ['rating','model_1','model_2']



pred = pd.concat([pd.DataFrame(y_test).reset_index().rating,pd.Series(labels),pd.Series(labels_2)],axis=1)

pred.columns = ['rating','model_1','model_2']



pred['final'] = np.where(pred.model_2 >= 3, np.where(pred.model_2 <= 9, pred.model_1, pred.model_2), pred.model_2)

#pred['final'] = np.where(pred.model_2 <= 9, pred.model_1, pred.model_2)

pred.tail()
mat = confusion_matrix(pred.rating.astype(int), pred.final)

ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False )

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.show()

assess_model("Ensemble model", pred.rating,pred.final)



acc = accuracy_score(pred.rating.astype(int),pred.final, normalize=True) * float(100)

print('\n****Test accuracy of Ensemble SVC is',(acc))