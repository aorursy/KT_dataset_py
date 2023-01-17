import functools

import json

import itertools

import re

import shutil

import tempfile



import matplotlib.pyplot as plt

import nltk

import numpy as np

import pandas as pd

import scipy.stats as stats



from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import (

    ENGLISH_STOP_WORDS,

    TfidfVectorizer,

)

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.model_selection import (

    train_test_split,

    RandomizedSearchCV,

    StratifiedShuffleSplit,

)

from sklearn.pipeline import (

    make_pipeline,

    Pipeline,

)

from sklearn.preprocessing import LabelEncoder
def load_dataframe(filepath, is_train=True):

    """Load data.



    Arguments

    ---------

    filepath : str, file-like, or path-like

        Path to JSON file.

    is_train : bool

        Load training data if True, else testing data.



    Returns

    -------

    data_tuple : tuple

        data : pd.DataFrame

            Input features.

        target : None or pd.Series

            Targets (training data).

    """

    with open(filepath) as fh:

        deserialized_data = json.load(fh)



        data_frame = pd.DataFrame(

            data={

                'ingredients': [

                    ' '.join(entry['ingredients'])

                    for entry in deserialized_data

                ],

            },

            index=[entry['id'] for entry in deserialized_data],

        )



        target_series = None

        if is_train:

            target_series = pd.Series(

                data=[entry['cuisine'] for entry in deserialized_data],

                name='cuisine',

            )



        return data_frame, target_series
data_train, target = load_dataframe('../input/train.json')

data_train.head()
value_counts = target.value_counts()



plt.figure(figsize=(20, 7))



plt.bar(range(value_counts.shape[0]), value_counts,

        color='SkyBlue')



plt.xticks(range(value_counts.shape[0]), value_counts.index,

           rotation=60, ha='right')

plt.title('Label Distribution')

plt.xlabel('Labels')

plt.ylabel('Total')

plt.show()
plt.figure(figsize=(16, 5))



ingredient_distribution = nltk.FreqDist(

    itertools.chain(*data_train['ingredients'].apply(str.split)))

ingredient_distribution.plot(

    50, cumulative=False, title='Most Common Word Occurences')
STOP_WORDS = frozenset(

    nltk.corpus.stopwords.words('english')

    + list(ENGLISH_STOP_WORDS)

)



print(f'The number of stop words: {len(STOP_WORDS)}.')
class IngredientTokenizer:

    """Custom tokenizer used in document transformations.



    Arguments

    ---------

    token_pattern_re : str or regex pattern

        Pattern indicating tokens.

    lemmatizer : nltk tokenizer or nltk lemmatizer

        Tokenizer or lemmatizer class transforming words into tokens.

    """



    def __init__(self,

                 token_pattern_re=None,

                 lemmatizer=None):

        self.token_pattern_re = token_pattern_re or re.compile(

            r'(?iu)\b\w\w+\b')

        self.lemmatizer = lemmatizer or nltk.wordnet.WordNetLemmatizer()



    def __repr__(self):

        return (

            f'{self.__class__.__name__}(\n\t'

            f'{self.token_pattern_re!r},\n\t{self.lemmatizer!r})'

        )



    def __call__(self, document):

        """Retrieve and tokenize/lemmatize valid words.



        Arguments

        ---------

        document : array-like

            Iterable containing texts.



        Returns

        -------

        lemmas : list

            Lemmatized/Tokenized words.

        """

        words = self.token_pattern_re.findall(document)

        call_fn = getattr(self.lemmatizer, 'lemmatize', 'tokenize')



        lemmas = []

        for word in words:

            word = word.lower()

            if word not in STOP_WORDS:

                lemmas.append(call_fn(word))



        return lemmas
vectorizer = TfidfVectorizer(

    min_df=20,

    max_df=0.95,

    stop_words=STOP_WORDS,

    tokenizer=IngredientTokenizer(),

)

bag_of_words = vectorizer.fit_transform(data_train['ingredients'])

bag_of_words
vectorizer.get_feature_names()[:10]
label_encoder = LabelEncoder()

target_encoded = label_encoder.fit_transform(target)



x_train, x_test, y_train, y_test = train_test_split(

    bag_of_words, target_encoded, random_state=0)



estimator = PassiveAggressiveClassifier(

    early_stopping=True,

    loss='hinge',

    average=True,

    class_weight='balanced',

    n_jobs=-1,

    verbose=False,

    random_state=0,

)

estimator.fit(x_train, y_train)



print(f'Train accuracy: {estimator.score(x_train, y_train):.3f}')

print(f'Test accuracy: {estimator.score(x_test, y_test):.3f}')
def visualize_coefficients(feature_names, estimator_coefficients,

                           class_=0, n_top=20, **kwargs):

    """Plot coefficient magnitude for a specified label.



    Parameters

    ----------

    feature_names : array-like

        Vectorized features.

    estimator_coefficients : array-like

        Estimator weights/importances.

    class_ : int, default 0

        Label.

    n_top : int, default 20

        The number of positive/negative coefficients to plot.

    kwargs : dict-like

        Additional key-word arguments for plot function.

    """

    coefficients = np.ravel(estimator_coefficients[class_])

    positive_coef = np.argsort(coefficients)[-n_top:]

    negative_coef = np.argsort(coefficients)[:n_top]

    coef_matrix = np.hstack([negative_coef, positive_coef])



    plt.figure(figsize=(22, 6))

    plt.bar(np.arange(2 * n_top), coefficients[coef_matrix],

            color=['b' if c < 0 else 'r' for c in coefficients[coef_matrix]])

    plt.xticks(np.arange(2 * n_top), feature_names[coef_matrix],

               rotation=60, ha='right')

    plt.subplots_adjust(bottom=0.3)

    plt.xlabel('Feature values')

    plt.ylabel('Coefficient magnitude')

    plt.title(kwargs.get('target_value'))

    plt.show()





feature_names = np.array(vectorizer.get_feature_names())

coefficients = estimator.coef_

visualize_coefficients_default = functools.partial(

    visualize_coefficients, feature_names, coefficients)
label = 15

label_encoded = label_encoder.inverse_transform([label])[0]



visualize_coefficients_default(

    class_=label, n_top=15, target_value=label_encoded)
cachedir = tempfile.mkdtemp()



pipe = Pipeline(

    steps=[

        ('tfidf', TfidfVectorizer()),

        ('tsvd', TruncatedSVD(algorithm='arpack')),

        ('pa', PassiveAggressiveClassifier(

            max_iter=1000,

            average=True,

            early_stopping=True,

            validation_fraction=0.1,

            n_jobs=-1,

            n_iter_no_change=20,

            random_state=0,

        ))

    ],

    memory=cachedir)



param_distributions = {

    'tfidf__min_df': range(1, 51),

    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],

    'tsvd__n_components': range(100, 501),



    'pa__C': stats.expon(1e-4, 0.5), # The regularization term C

    'pa__loss': ['hinge', 'squared_hinge'], # PA-I or PA-II

    'pa__class_weight': [None, 'balanced'],

}



cv = StratifiedShuffleSplit(

    n_splits=3,

    test_size=0.2,

    random_state=0,

)



grid = RandomizedSearchCV(

    estimator=pipe,

    param_distributions=param_distributions,

    n_iter=5,

    cv=cv,

    scoring='accuracy',

    n_jobs=-1,

    iid=True,

    refit=True,

    error_score=np.nan,

    verbose=True,

)



# Uncomment the line below to run hyperparameter optimization.

# grid.fit(data_train['ingredients'], target)



shutil.rmtree(cachedir)
%%time



best_pipe = make_pipeline(

    TfidfVectorizer(min_df=20),

    TruncatedSVD(n_components=500, algorithm='arpack'),

    PassiveAggressiveClassifier(

        C=0.1,

        loss='hinge',

        class_weight='balanced',

        max_iter=1000,

        early_stopping=True,

        validation_fraction=0.2,

        n_jobs=-1,

        random_state=0,

        average=True,

    )

)

# best_pipe = grid.best_estimator_

best_pipe.fit(data_train['ingredients'], target)
data_test, _ = load_dataframe("../input/test.json", is_train=False)



predictions = best_pipe.predict(data_test['ingredients'])



submission = pd.DataFrame(data={"id": data_test.index, "cuisine": predictions})

submission.to_csv("submission.csv", index=None)

submission.head(n=10).T