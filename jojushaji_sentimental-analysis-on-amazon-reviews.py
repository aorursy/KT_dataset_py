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
#load the data
data=pd.read_csv("/kaggle/input/amazon-unlocked-mobilecsv/Amazon_Unlocked_Mobile.csv")
# Sample the data to speed up computation
# Comment out this line to match with lecture
data = data.sample(frac=0.1, random_state=10)
#drop nan values and rows with  rating =3
data.dropna(inplace=True)
data=data[data.Rating!=3]
# adding a new column called Postively rated "1" for Ratings greater than 3 & "0"for others
data['Postively rated']=np.where(data.Rating>3,1,0)
data.head()
data['Postively rated'].mean()
#lets only take the Reviews & Postively rated columns for the review analysis
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.Reviews,data['Postively rated'],random_state=0)
print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)
from sklearn.feature_extraction.text import CountVectorizer

# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)

#looking at the vocaboulry of 2000 features,messy words with N.O's and miss spelling
vect.get_feature_names()[::2000]
len(vect.get_feature_names())
# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)

X_train_vectorized
from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
from sklearn.metrics import roc_auc_score

# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
#et's take a look at a few tricks for reducing the number of 
#features that might help improve our model's performance or reduce a refitting.
#CountVectorizor and tf–idf Vectorizor both take an argument, 
#mindf, which allows us to specify a minimum number of documents in which a token needs to appear to become part of the vocabulary.

from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())

#This helps us remove some words that might appear in only a few and are unlikely to be useful predictors.
#For example, here we'll pass in min_df = 5, which will remove any words from our vocabulary that 
#appear in fewer than five documents.

#Looking at the length, we can see we've reduced the number of features 
#now lets fit the model and check the accuracy

X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

#no improvment in the auc score
feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

#List of features with the smallest tf–idf either commonly appeared across all reviews or only 
#appeared rarely in very long reviews.
#List of features with the largest tf–idf contains words which appeared frequently in a review, 
#but did not appear commonly across all reviews.
#Looking at the smallest and largest coefficients from our new model, we can again see which words our 
#model has connected to negative and positive reviews.

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

#prob with our previous model is that 
#One problem with our previous bag-of-words approach is word order is disregarded. So, not an issue, phone is 
#working is seen the same as an issue, phone is not working
#Our current model sees both of these reviews as negative reviews.
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())

#To create these n-gram features, we'll pass in a tuple to the parameter ngram_range, where the values 
#correspond to the minimum length and maximum lengths of sequences.

#Keep in mind that, although n-grams can be powerful in capturing meaning, longer sequences 
#can cause an explosion of the number of features.
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
# These reviews are now correctly identified
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))