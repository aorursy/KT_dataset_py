import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



spam_data = pd.read_csv('../input/spam.csv', encoding='latin-1')

spam_data.head()
spam_data = spam_data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)

spam_data = spam_data.rename(columns = {'v1': 'target','v2': 'text'})



spam_data.head()
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

from nltk.stem import SnowballStemmer
stop_words = set(stopwords.words('english'))

stemmer = SnowballStemmer('english')
spam_data['parsed'] = spam_data['text'].apply(lambda x: x.lower())

spam_data['parsed'] = spam_data['text'].apply(lambda x: word_tokenize(x))

spam_data['parsed'] = spam_data['text'].apply(lambda x: [word for word in str(x).split() if word not in stop_words])

spam_data['parsed'] = spam_data['parsed'].apply(lambda x: [stemmer.stem(word) for word in x])

spam_data['parsed'] = spam_data['parsed'].apply(lambda x: ' '.join(x))
spam_data.head()
s = spam_data['target'].value_counts()

sns.barplot(x=s.values, y=s.index)

plt.title('Data Distribution')
s1 = spam_data[spam_data['target'] == 'ham']['parsed'].str.len()

sns.distplot(s1, label='Ham')

s2 = spam_data[spam_data['target'] == 'spam']['parsed'].str.len()

sns.distplot(s2, label='Spam')

plt.title('Lenght Distribution')

plt.legend()
s1 = spam_data[spam_data['target'] == 'ham']['parsed'].str.replace(r'\D+', '').str.len()

sns.distplot(s1, label='Ham')

s2 = spam_data[spam_data['target'] == 'spam']['parsed'].str.replace(r'\D+', '').str.len()

sns.distplot(s2, label='Spam')

plt.title('Digits Distribution')

plt.legend()
s1 = spam_data[spam_data['target'] == 'ham']['parsed'].str.replace(r'\w+', '').str.len()

sns.distplot(s1, label='Ham')

s2 = spam_data[spam_data['target'] == 'spam']['parsed'].str.replace(r'\w+', '').str.len()

sns.distplot(s2, label='Spam')

plt.title('Non-Digits Distribution')

plt.legend()
spam_data.groupby('target').describe()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(spam_data['parsed'], 

                                                    spam_data['target'], 

                                                    random_state=0)
from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer().fit(X_train)

print('Vocabulary len:', len(vect.get_feature_names()))

print('Longest word:', max(vect.vocabulary_, key=len))



X_train_vectorized = vect.transform(X_train)
from sklearn.naive_bayes import MultinomialNB



model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectorized, y_train)
# get the feature names as numpy array

feature_names = np.array(vect.get_feature_names())



# Sort the coefficients from the model

sorted_coef_index = model.coef_[0].argsort()



# Find the 10 smallest and 10 largest coefficients

# The 10 largest coefficients are being indexed using [:-11:-1] 

# so the list returned is in order of largest to smallest

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))

print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
from sklearn.metrics import accuracy_score



y_pred = model.predict(vect.transform(X_test))

print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
from sklearn.feature_extraction.text import TfidfVectorizer



vect = TfidfVectorizer(min_df=3).fit(X_train)

print('Vocabulary len:', len(vect.get_feature_names()))

print('Longest word:', max(vect.vocabulary_, key=len))



X_train_vectorized = vect.transform(X_train)
model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectorized, y_train)
feature_names = np.array(vect.get_feature_names())



sorted_coef_index = model.coef_[0].argsort()



print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))

print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
from sklearn.metrics import accuracy_score



y_pred = model.predict(vect.transform(X_test))

print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
def add_feature(X, feature_to_add):

    """

    Returns sparse feature matrix with added feature.

    feature_to_add can also be a list of features.

    """

    from scipy.sparse import csr_matrix, hstack

    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
vect = TfidfVectorizer(min_df=5).fit(X_train)

print('Vocabulary len:', len(vect.get_feature_names()))

print('Longest word:', max(vect.vocabulary_, key=len))



X_train_vectorized = vect.transform(X_train)



X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectorized, y_train)
index = np.array(vect.get_feature_names() + ['length_of_doc'])

values  = model.coef_[0]

features_series = pd.Series(data=values,index=index)



print('Smallest Coefs:\n{}\n'.format(features_series.nsmallest(10).index.values.tolist()))

print('Largest Coefs: \n{}'.format(features_series.nlargest(10).index.values.tolist()))
X_test_vectorized = vect.transform(X_test)

X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())

    

y_pred = model.predict(X_test_vectorized)

print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
vect = TfidfVectorizer(min_df=5, ngram_range=(1, 3)).fit(X_train)

print('Vocabulary len:', len(vect.get_feature_names()))

print('Longest word:', max(vect.vocabulary_, key=len))



X_train_vectorized = vect.transform(X_train)



X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())

X_train_vectorized = add_feature(X_train_vectorized, X_train.str.replace(r'\D+', '').str.len())
model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectorized, y_train)
index = np.array(vect.get_feature_names() + ['length_of_doc', 'digit_count'])

values  = model.coef_[0]

features_series = pd.Series(data=values,index=index)



print('Smallest Coefs:\n{}\n'.format(features_series.nsmallest(10).index.values.tolist()))

print('Largest Coefs: \n{}'.format(features_series.nlargest(10).index.values.tolist()))
X_test_vectorized = vect.transform(X_test)

X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())

X_test_vectorized = add_feature(X_test_vectorized, X_test.str.replace(r'\D+', '').str.len())

    

y_pred = model.predict(X_test_vectorized)

print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
vect = CountVectorizer(min_df=5, ngram_range=(2, 5), analyzer='char_wb').fit(X_train)

print('Vocabulary len:', len(vect.get_feature_names()))

print('Longest word:', max(vect.vocabulary_, key=len))



X_train_vectorized = vect.transform(X_train)



X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())

X_train_vectorized = add_feature(X_train_vectorized, X_train.str.replace(r'\D+', '').str.len())

X_train_vectorized = add_feature(X_train_vectorized, X_train.str.replace(r'\w+', '').str.len())
model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectorized, y_train)
index = np.array(vect.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])

values = model.coef_[0]

features_series = pd.Series(data=values,index=index)



print('Smallest Coefs:\n{}\n'.format(features_series.nsmallest(10).index.values.tolist()))

print('Largest Coefs: \n{}'.format(features_series.nlargest(10).index.values.tolist()))
X_test_vectorized = vect.transform(X_test)

X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())

X_test_vectorized = add_feature(X_test_vectorized, X_test.str.replace(r'\D+', '').str.len())

X_test_vectorized = add_feature(X_test_vectorized, X_test.str.replace(r'\w+', '').str.len())

    

y_pred = model.predict(X_test_vectorized)

print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))