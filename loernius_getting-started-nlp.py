import numpy as np

import pandas as pd

import nltk

from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split, cross_validate, LeaveOneOut, KFold, ShuffleSplit

from sklearn.linear_model import LogisticRegression, PassiveAggressiveRegressor, LinearRegression, RidgeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from math import floor
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
print('Train has {0} rows and {1} columns'.format(train.shape[0], train.shape[1]))

print('Test has {0} rows and {1} columns'.format(test.shape[0], test.shape[1]))
positive_prcnt = train['target'].value_counts()[0]/train['target'].count()*100

negative_prcnt = train['target'].value_counts()[1]/train['target'].count()*100

print('Percentage of positive cases: {0}% '.format(round(positive_prcnt, 2)))

print('Percentage of negative cases: {0}% '.format(round(negative_prcnt, 2)))
stopwords = set(nltk.corpus.stopwords.words('english'))
for punct in punctuation:

    stopwords.add(punct)
SEED = 321321

np.random.seed(SEED)


vect = TfidfVectorizer(lowercase=True, max_features=100, ngram_range=(1,2))

raw_tfidf = vect.fit_transform(train.text).toarray()

# treino, teste, classe_treino, classe_teste = train_test_split(tfidf_bruto, train.target)

classes = train.target
def validador_de_modelos(model, train, test, class_train, class_test, cross_validation=0.2):

    model.fit(train, class_train)

    

    predict = model.predict(test)

    

    print("O modelo passado teve acuracia de {0}".format(model.score(test,class_test)))
def cross_validation(model, x, y, validation_splitter):

    print(model)

    print('\n')

    print(validation_splitter)

    cv = cross_validate(model, x, y, cv=validation_splitter)

    st_deviation = cv['test_score'].std()

    mean  = cv['test_score'].mean()

    conf_interval = [(mean-2 * st_deviation)*100, (mean+2 * st_deviation)*100]



    print('\n################################\n')

    print('Mean score: {0}'.format(np.mean(cv['test_score'])))

    print('Median score: {0}'.format(np.median(cv['test_score'])))

    print('Min score: {0}'.format(np.amin(cv['test_score'])))

    print('Max score: {0}'.format(np.amax(cv['test_score'])))

    print('\n################################\n')

    print('Confident interval: [{0}% , {1}%]'.format(round(conf_interval[0],2), round(conf_interval[1], 2)))

    print('\n')
rc = RidgeClassifier()

lor = LogisticRegression()

par = PassiveAggressiveRegressor()

dtr = DecisionTreeRegressor()

dtc = DecisionTreeClassifier(criterion = 'entropy')

gnb = GaussianNB()

classifiers_list= [rc, lor, par, dtr, dtc, gnb]
kf = KFold(100, True)

ss = ShuffleSplit(n_splits=10)



split_list = [kf, ss]
# for classifier in classifiers_list:

#     for splitter in split_list:

#         cross_validation(classifier, raw_tfidf, classes, splitter)

# Commenting this routine due to slow processing



# chosen methods so far: RidgeClassifier with KFold splitter - 60% to 80% of confident interval

cross_validation(rc, raw_tfidf, classes, kf)
train['token'] = train.text.apply(nltk.tokenize.word_tokenize)

train.head()
def remove_stopwords(text):

    token_without_stopwords = [w for w in text if w not in stopwords]

    return token_without_stopwords



train['token_without_stopwords'] = train.token.apply(remove_stopwords)
def stemmer(text):

    stemmer = nltk.stem.PorterStemmer()

    stemmed_text = [stemmer.stem(w) for w in text]

    return stemmed_text



train['stemmed_tokens'] = train.token_without_stopwords.apply(stemmer)
word_list = list()

for list_of_words in train['stemmed_tokens']:

    word_list.append(' '.join(map(str, list_of_words)))



train['string_stemmed_tokens'] = word_list
cv = CountVectorizer(max_features=1000)

X = cv.fit_transform([w.lower() for w in train['string_stemmed_tokens']]).toarray()

y = train.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
score_list = list()

for train_index, test_index in kf.split(X, y):

    rc = RidgeClassifier()

    cv_X_train, cv_y_train, cv_X_test, cv_y_test = X[train_index], y[train_index], X[test_index], y[test_index]

    rc.fit(cv_X_train, cv_y_train)

    predict = rc.predict(cv_X_test)

    score_list.append(accuracy_score(predict, cv_y_test))

score_array = np.array(score_list)



fig = plt.figure()



plt.title("Variation on accuracy score with cross-validation")

plt.tick_params(

    axis='x',

    which='both',

    bottom=False,

    top=False,

    labelbottom=False)

plt.plot(score_array)
fig = plt.figure()



plt.title("Variation on accuracy score with cross-validation")

plt.tick_params(

    axis='x',

    which='both',

    bottom=False,

    top=False,

    labelbottom=False)

plt.plot(np.sort(score_array))
st_deviation = score_array.std()

mean  = score_array.mean()

median  = np.median(score_array)

conf_interval = [(mean-2 * st_deviation)*100, (mean+2 * st_deviation)*100]



print('\n################################\n')

print('Mean score: {0}'.format(mean))

print('Median score: {0}'.format(median))

print('Min score: {0}'.format(score_array.min()))

print('Max score: {0}'.format(score_array.max()))

print('\n################################\n')

print('Confident interval: [{0}% , {1}%]'.format(round(conf_interval[0],2), round(conf_interval[1], 2)))

print('\n')
rc = RidgeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42225420)

rc.fit(X_train, y_train)

pr = rc.predict(X_test)

print("Final model accuracy: {:.2%}".format(accuracy_score(pr, y_test)))
test['token'] = test.text.apply(nltk.tokenize.word_tokenize)

test['token_without_stopwords'] = test.token.apply(remove_stopwords)

test['stemmed_tokens'] = test.token_without_stopwords.apply(stemmer)



word_list = list()

for list_of_words in test['stemmed_tokens']:

    word_list.append(' '.join(map(str, list_of_words)))



test['string_stemmed_tokens'] = word_list

test_x = cv.transform([w.lower() for w in test['string_stemmed_tokens']]).toarray()
test['target'] = rc.predict(test_x)
submit = test[['id', 'target']]

submit.head()
submit.to_csv('submit.csv', index=False)