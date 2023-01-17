import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import nltk

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from nltk.probability import FreqDist



dataset = pd.read_csv('../input/kindle_reviews.csv', na_filter=False)

newdf = dataset[:10000]
newdf.head()
newdf.columns
newdf.dtypes
print ("Shape of the dataset - ", newdf.shape)

#check for the missing values

newdf.apply(lambda x: sum(x.isnull()))
newdf['overall'].value_counts()
# Remove neutral rated

newdf = newdf[newdf['overall'] != 3]

newdf['Positively Rated'] = np.where(newdf['overall'] > 3, 1, 0)



# 22 rows from reviewText are blank. Lets add sample review for it

#newdf['reviewText']=newdf['reviewText'].fillna("No Review", inplace=True)

#newdf = newdf.replace(np.nan, '', regex=True)

#newdf.apply(lambda x: sum(x.isnull()))

#print (newdf['reviewText'].head(10))
# Number of rating which are positively rated 

newdf['Positively Rated'].mean()
from  sklearn.model_selection import train_test_split



# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(newdf['reviewText'],newdf['Positively Rated'], random_state=0)

print('X_train first entry: ', X_train.iloc[1])

print('\nX_train shape: ', X_train.shape)
from  sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from  sklearn.metrics import roc_auc_score



# Fit the CountVectorizer to the training data

vect = CountVectorizer().fit(X_train)

# transform the documents in the training data to a document-term matrix

X_train_vectorized = vect.transform(X_train)

# Train the model

model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

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
df = newdf.groupby('asin', as_index=False).agg({'Positively Rated': 'sum'})

#df.sort_values(by=['Positively Rated'], ascending=False)

print ("PRODUCT HAVING THE LARGEST POSTIVE RATING - ",df.loc[df['Positively Rated'].idxmax()][0])
X = np.concatenate((X_train, X_test), axis=0)

y = np.concatenate((y_train, y_test), axis=0)
# summarize size

print("Training data: ")

print(X.shape)

print(y.shape)
# Summarize number of classes

print("Classes: ")

print(np.unique(y))
# Summarize number of words

print("Number of words: ")

print(len(np.unique(np.hstack(X))))
# Summarize review length

print("Review length: ")

result = map(len, X)

print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))

# plot review length as a boxplot and histogram

pyplot.subplot(121)

pyplot.boxplot(result)

pyplot.subplot(122)

pyplot.hist(result)

pyplot.show()