import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns

import warnings; warnings.simplefilter('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_train= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print('The shape of train dataset: ', df_train.shape)

print('The shape of test dataset: ', df_test.shape)
df_train.head()
df_test.head()
df_train.sample(n=10, replace=False, random_state=1) # We will look at 10 random samples
df_train.isnull().sum()
df_test.isnull().sum()
sns.set_style('darkgrid')

sns.countplot(x='target', data= df_train)

Labels= ('No Disaster', 'Real Disaster')

plt.xticks(range(2), Labels)
df_train[df_train['target']== 1]['keyword'].value_counts().head()
df_train['text_length']= df_train['text'].apply(len)

df_train.sample(n=5, replace=False, random_state=1)
sns.boxplot(x= 'target', y= 'text_length', data= df_train, palette= 'rainbow')
correlation= df_train['target'].corr(df_train['text_length'])

correlation
missing_cols = ['keyword', 'location']



fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)



sns.barplot(x=df_train[missing_cols].isnull().sum().index, y=df_train[missing_cols].isnull().sum().values, ax=axes[0])

sns.barplot(x=df_test[missing_cols].isnull().sum().index, y=df_test[missing_cols].isnull().sum().values, ax=axes[1])



axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)

axes[0].tick_params(axis='x', labelsize=15)

axes[0].tick_params(axis='y', labelsize=15)

axes[1].tick_params(axis='x', labelsize=15)

axes[1].tick_params(axis='y', labelsize=15)



axes[0].set_title('Training Set', fontsize=13)

axes[1].set_title('Test Set', fontsize=13)
def cleaned_tweet(text):

    import re

    from nltk.corpus import stopwords

    from nltk.stem.porter import PorterStemmer

    tweets = re.sub("[^a-zA-Z]", ' ', text)

    tweets = tweets.lower()

    tweets = tweets.split()

    ps = PorterStemmer()

    tweets = [ps.stem(word) for word in tweets if not word in set(stopwords.words('english'))]

    tweets = ' '.join(tweets)

    return tweets

df_train['clean_tweet'] = df_train['text'].apply(cleaned_tweet)

df_test['clean_tweet'] = df_test['text'].apply(cleaned_tweet)
from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer(max_features = 1500)

X_train_vector= cv.fit_transform(df_train['clean_tweet'])

y_train= df_train.iloc[:, 4].values



'''note that I'm NOT using .fit_transform() here. Using just .transform() makes sure

 that the tokens in the train vectors are the only ones mapped to the test vectors -

 i.e. that the train and test vectors use the same set of tokens.'''



X_test_vector= cv.transform(df_test['clean_tweet'])
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
from sklearn import model_selection

scores= model_selection.cross_val_score(classifier, X_train_vector, df_train['target'], cv=5, scoring="f1")

scores
# Fitting to the train set

classifier.fit(X_train_vector, y_train)
y_pred = classifier.predict(X_test_vector)
submit_df = pd.DataFrame()

submit_df['id'] = df_test['id']

submit_df['target'] = y_pred



submit_df.to_csv('submission_final.csv', index=False)