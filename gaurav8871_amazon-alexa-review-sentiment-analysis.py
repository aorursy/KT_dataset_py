import re  # regular expressions



# for mathemactical computation, oraganization and visualization of data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# for data preprocessing, model training and model evaluation

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix



import nltk # for stemming and stopwords removal
data = pd.read_csv('../input/amazon_alexa.tsv',sep='\t')

data.shape
data.head()
plt.rcParams["figure.figsize"] = (20,10)

data.groupby(['variation']).feedback.value_counts().plot(kind='bar')
data['variation'].nunique()
data.isna().sum()
data.drop(['rating','date'], inplace=True, axis=1)

data.head()
import collections

collections.Counter(data['feedback'])
stop_words = set(nltk.corpus.stopwords.words('english'))

sno = nltk.stem.SnowballStemmer('english')



def cleanpunc(sentence):

    cleaned = re.sub(r"[?|!|\'|\"|#]", r"",sentence)

    cleaned =  re.sub(r"[.|,|)|(|\|/]",r"",cleaned)

    return cleaned.lower()



sentences = data['verified_reviews'].values

reviews = []

for sent in sentences:

    cleaned_sent = cleanpunc(sent)

    sent_arr = cleaned_sent.split()

    output_sent = ''

    for word in sent_arr:

        if word not in stop_words:

            stemmed_word = sno.stem(word)

            output_sent = output_sent + ' ' + stemmed_word

    reviews.append(output_sent)



reviews_text = pd.DataFrame({'reviews': reviews})

data = pd.concat([data,reviews_text], axis=1)

data.head()
data.drop(['verified_reviews'], axis=1,inplace=True)



from sklearn.utils import resample

data_majority = data[data.feedback == 1]

data_minority = data[data.feedback == 0]

data_minority_upsampled = resample(data_minority,

                                   replace=True,n_samples=2500,random_state=123)

data_upsampled = pd.concat([data_majority, data_minority_upsampled])

final = pd.concat([data_upsampled,

                   pd.get_dummies(data_upsampled['variation'],sparse=True)], axis=1)



final.shape



final.drop(['variation'], axis=1, inplace=True)

final.head()
count_vect = CountVectorizer(ngram_range=(1,2))

final_counts = count_vect.fit_transform(final['reviews'].values)

print(final_counts.get_shape())

print(final.shape)

final.drop(['reviews'],axis=1,inplace=True)
rev_df = pd.DataFrame(final_counts.todense(),columns=count_vect.get_feature_names())
rev_df.shape
final.reset_index(inplace=True, drop=True)

final_df = pd.concat([final,rev_df], axis=1)
final_df.shape
X = final_df.iloc[:,1:].values

y = np.ravel(final_df.iloc[:,0:1].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
clf = MultinomialNB()

param_disb = { 'alpha': [10**-4,10**-3,10**-2,10**-1,10,1,10**2,10**3,10**4]}

search = GridSearchCV(clf, param_grid=param_disb, cv=5)

search.fit(X_train,y_train)



print(search.best_estimator_)
clf = MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)

clf. fit(X_train, y_train)

print(confusion_matrix(y_test, clf.predict(X_test)))

print(f"Accuracy Score -> {clf.score(X_test,y_test)}")