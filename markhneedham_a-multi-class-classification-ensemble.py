import pandas as pd

import numpy as np

from sklearn import linear_model, metrics

from sklearn.ensemble import VotingClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
Y_COLUMN = "author"

TEXT_COLUMN = "text"

train_df = pd.read_csv("../input/train.csv", usecols=[Y_COLUMN, TEXT_COLUMN])

train_df.head()
tfidf_pipe = Pipeline([

    ('tfidf', TfidfVectorizer(min_df=3, max_features=None,

                              strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',

                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,

                              stop_words='english')),

    ('mnb', MultinomialNB())

])



unigram_pipe = Pipeline([

    ('cv', CountVectorizer()),

    ('mnb', MultinomialNB())

])



ngram_pipe = Pipeline([

    ('cv', CountVectorizer(ngram_range=(1, 2))),

    ('mnb', MultinomialNB())



])
def test_pipeline(df, nlp_pipeline):

    y = df[Y_COLUMN].copy()

    X = pd.Series(df[TEXT_COLUMN])

    rskf = StratifiedKFold(n_splits=5, random_state=1)

    losses = []

    accuracies = []

    for train_index, test_index in rskf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        nlp_pipeline.fit(X_train, y_train)

        losses.append(metrics.log_loss(y_test, nlp_pipeline.predict_proba(X_test)))

        accuracies.append(metrics.accuracy_score(y_test, nlp_pipeline.predict(X_test)))



    print("kfolds log losses: {0}, mean log loss: {1} mean accuracy: {2}".format(

        str([str(round(x, 3)) for x in sorted(losses)]),

        round(np.mean(losses), 3),

        round(np.mean(accuracies), 3)

    ))
test_pipeline(train_df, unigram_pipe)
test_pipeline(train_df, ngram_pipe)
test_pipeline(train_df, tfidf_pipe)
classifiers = [

    ("tfidf", tfidf_pipe),

    ("ngram", ngram_pipe),

    ("unigram", unigram_pipe),

]



mixed_pipe = Pipeline([

    ("voting", VotingClassifier(classifiers, voting="soft"))

])
test_pipeline(train_df, mixed_pipe)
# This function generates all possible combinations of the classifiers

# e.g. 

# [0 0 0] all turned off

# [1 1 1] all turned on

# [1 0 1] the first and last ones turned on, the middle one turned off

def combinations_on_off(num_classifiers):

    return [[int(x) for x in list("{0:0b}".format(i).zfill(num_classifiers))]

            for i in range(1, 2 ** num_classifiers)]



param_grid = dict(

        voting__weights=combinations_on_off(len(classifiers)),

)



grid_search = GridSearchCV(mixed_pipe, param_grid=param_grid, n_jobs=-1, verbose=10, scoring="neg_log_loss")



y = train_df[Y_COLUMN].copy()

X = pd.Series(train_df[TEXT_COLUMN])



grid_search.fit(X, y)



cv_results = grid_search.cv_results_



for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):

    print(params, mean_score)



print("Best score: %0.3f" % grid_search.best_score_)

print("Best parameters set:")

best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))