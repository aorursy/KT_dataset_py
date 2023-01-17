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
df = pd.read_csv('/kaggle/input/amazon-reviews-unlocked-mobile-phones/Amazon_Unlocked_Mobile.csv')

df = df.sample(frac=0.1, random_state=10)

df.head()
#Drop the missing values.

df.dropna(inplace=True)



#remove any neutral rating equals to 3.

df = df[df['Rating']!=3]



#Encode 4 star and 5 star as positively rated 1.

#Encode 1 star and 2 star as poorely rated 0.

df['Positively Rated'] = np.where(df['Rating']>3,1,0)

df.head(10)
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

fig,(ax0,ax1) = plt.subplots(nrows=1,

                            ncols=2,

                            figsize = (20,12))

ax0.scatter(df['Rating'],df['Review Votes'])

ax0.set(title='Rating vs Review Votes',

       xlabel='Rating',

       ylabel='Review Votes')

ax1.plot(df['Rating']==4,df['Rating']==5)

ax1.set(title='Rating 4 vs Rating 5',

       xlabel='Rating 4',

       ylabel='Rating 5');
# most ratings are positive

df['Positively Rated'].mean()
from sklearn.model_selection import train_test_split



X = df['Reviews']

y = df['Positively Rated']

#spliting data into training and test set.

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

print('X_train first entry:\n',X_train.iloc[0])

print('\nX_train shape',X_train.shape)
df['Positively Rated'].value_counts()
from sklearn.feature_extraction.text import CountVectorizer

#fit the countVectorizer to the training data.

vect = CountVectorizer()

vect.fit(X_train)



#getting every 2000 vocabulay features.

vect.get_feature_names()[::2000]
len(vect.get_feature_names())
#transform the document in the training data to a document term matrix.

X_train_vectorized = vect.transform(X_train)

X_train_vectorized
from sklearn.linear_model import LogisticRegression

#train the model.

model = LogisticRegression()

model.fit(X_train_vectorized,y_train)
from sklearn.metrics import roc_auc_score,roc_curve



#predict the transform test document.

predictions = model.predict(vect.transform(X_test))

print('AUC: ',roc_auc_score(y_test,predictions))
#get the feature names as numpy array.

feature_names = np.array(vect.get_feature_names())



#sort the coffecient from the model.

sorted_coef_index = model.coef_[0].argsort()



'''Find the 10 smallest and 10 largest coefficients.

 The 10 largest coefficients are being indexed using [:-11:-1] 

 so the list returned is in order of largest to smallest.'''



print('Smallest Coefficient(Negative reviews): \n{}\n'.format(feature_names[sorted_coef_index[:10]]))

print('Largest Coeffiecient(Positive reviews): \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
from sklearn.feature_extraction.text import TfidfVectorizer



# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5

vect = TfidfVectorizer(min_df=5).fit(X_train)

len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)



model = LogisticRegression()

model.fit(X_train_vectorized, y_train)



predictions = model.predict(vect.transform(X_test))



print('AUC: ', roc_auc_score(y_test, predictions))
feature_names = np.array(vect.get_feature_names())



sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()



print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))

print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
sorted_coef_index = model.coef_[0].argsort()



print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))

print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
# These reviews are treated the same by our current model

print(model.predict(vect.transform(['not an issue, phone is working',

                                    'an issue, phone is not working'])))
# Fit the CountVectorizer to the training data specifiying a minimum 

# document frequency of 5 and extracting 1-grams and 2-grams

vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)



X_train_vectorized = vect.transform(X_train)



len(vect.get_feature_names())
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