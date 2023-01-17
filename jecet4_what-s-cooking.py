import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_json('/kaggle/input/ml1920-whats-cooking/cooking_train.json')



print('recipes: ', train.shape[0], '\n')

print('cuisines:\n', sorted(set(train.cuisine)), '\n')

print('Head:\n', train.head(), '\n')

print('Sample Ingredients:\n', train.ingredients[:10])

print(train.shape)
from sklearn.feature_extraction.text import CountVectorizer



def preprocessor(line):

    tabbed = ' '.join(line).lower()

    return tabbed



recipes = train.ingredients[:4]



for r in recipes:

    print(preprocessor(r))

    print()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_extraction.text import TfidfVectorizer



vect = TfidfVectorizer(preprocessor=preprocessor, strip_accents='unicode')



data = pd.DataFrame(data=vect.fit_transform(train.ingredients).todense(), columns=sorted(vect.vocabulary_))

data.head()
cuisine_occurances = train.cuisine.value_counts()

cuisine_occurances.plot.bar(figsize=(20, 7), title='Cuisines\' occurances in train data')
cuisine = train.cuisine

cuisine.rename('original_cuisine', inplace=True)



data = pd.concat([data, cuisine], axis=1)

data = data.groupby('original_cuisine').agg(np.mean)



def get_freq(row):

    return row[row['most_frequent']]



data['most_frequent'] = data.idxmax(axis=1)

data['freq'] = data.apply(get_freq, axis=1)



data[['most_frequent', 'freq']]
labels = sorted(train['cuisine'].unique())

freqs = data.freq.to_numpy()



x = np.arange(len(labels))  # the label locations

width = 0.8  # the width of the bars



fig, ax = plt.subplots(figsize=(20, 7), dpi=256)



rects1 = ax.bar(x, freqs, width)



# Add some text for labels, title and custom x-axis tick labels

ax.set_ylabel('Frequence')

ax.set_title('Most frequent words for each cuisine (according to tfidf formula)')

ax.set_xticks(x)

ax.set_xticklabels(labels)



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying connected word"""

    for i in range(0, len(rects)):

        height = data.most_frequent.to_numpy()[i]

        rect = rects[i]

        ax.annotate(height,

                    xy=(rect.get_x(), rect.get_height() * 1.02))



autolabel(rects1)



fig.tight_layout()



plt.show()
from nltk.stem.snowball import EnglishStemmer

from nltk.stem import WordNetLemmatizer



stemmer = EnglishStemmer()

lemmatizer = WordNetLemmatizer()



count_analyzer = CountVectorizer().build_analyzer()



def stem(doc):

    line = str.lower(' '.join(doc))

    return (stemmer.stem(w) for w in count_analyzer(line))



def lemmatize(doc):

    line = str.lower(' '.join(doc))

    return (lemmatizer.lemmatize(w) for w in count_analyzer(line))



stopwords = ['a', 'the', 'in', 'of', 'off', 'up', 'down',

             'fresh', 'virgin', 'sliced', 'leaf', 'leaves',

             'chopped', 'cooked', 'baby', 'yellow', 'red',

             'blue', 'white', 'black', 'purple', 'violet',

             'over']
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



X_train = train['ingredients']

y_train = train['cuisine']



vectorization = Pipeline([('vectorizer', CountVectorizer(analyzer=lemmatize, strip_accents='unicode', stop_words=stopwords)),

                          ('tfidf', TfidfTransformer())])



preprocessing = Pipeline([('vectorization', vectorization),

                          ('poly', PolynomialFeatures(degree=2))])



pipeline = Pipeline([('preprocessing', preprocessing), 

                     ('model', LinearSVC(random_state=0, C=1.2, max_iter=4000))])



scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print(np.mean(scores))
ngrams_pipeline = Pipeline([('vectorizer', TfidfVectorizer(analyzer=stem, stop_words='english', ngram_range = (1,2))),

                            ('model', LinearSVC(random_state=0, C=1, max_iter=4000))])



print(np.mean(cross_val_score(ngrams_pipeline, X_train, y_train, cv=5)))
from sklearn.ensemble import ExtraTreesClassifier



extra_trees_pipeline = Pipeline([('preprocessing', vectorization),

                                 ('model', ExtraTreesClassifier(n_estimators=200, random_state=0))])



scores = cross_val_score(extra_trees_pipeline, X_train, y_train, cv=5)

print(np.mean(scores))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=200, random_state=0)



rf_pipeline = Pipeline([('preprocessing', vectorization),

                        ('model', rf)])



scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5)



print(np.mean(scores))
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import ExtraTreesClassifier



vc = VotingClassifier(estimators=[('linearSVC', Pipeline([('poly', PolynomialFeatures(degree=2)), ('svc', LinearSVC(random_state=0, C=1, max_iter=4000))])), 

                                  ('extraTrees', ExtraTreesClassifier(n_estimators=150, random_state=0)),

                                  ('rf', RandomForestClassifier(n_estimators=150, random_state=0))], voting='hard')



vc_pipeline = Pipeline([('preprocessing', vectorization),

                        ('model', vc)])



scores = cross_val_score(vc_pipeline, X_train, y_train, cv=5)



print(np.mean(scores))
test = pd.read_json('/kaggle/input/ml1920-whats-cooking/cooking_test.json')



X_test = test['ingredients']



pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)



submission = test.copy()

submission['cuisine'] = prediction

submission.to_csv('et_submission.csv', index=False, columns=['id', 'cuisine'])