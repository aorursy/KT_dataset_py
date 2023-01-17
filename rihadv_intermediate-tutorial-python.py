import pandas as pd
import spacy

from multiprocessing import cpu_count
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from spacy import attrs
from spacy.symbols import VERB, NOUN, ADV, ADJ
TEXT_COLUMN = 'text'
Y_COLUMN = 'author'
def test_pipeline(df, nlp_pipeline, pipeline_name=''):
    y = df[Y_COLUMN].copy()
    X = pd.Series(df[TEXT_COLUMN])
    # If you've done EDA, you may have noticed that the author classes aren't quite balanced.
    # We'll use stratified splits just to be on the safe side.
    rskf = StratifiedKFold(n_splits=5, random_state=1)
    losses = []
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nlp_pipeline.fit(X_train, y_train)
        losses.append(metrics.log_loss(y_test, nlp_pipeline.predict_proba(X_test)))
    print(f'{pipeline_name} kfolds log losses: {str([str(round(x, 3)) for x in sorted(losses)])}')
    print(f'{pipeline_name} mean log loss: {round(pd.np.mean(losses), 3)}')
train_df = pd.read_csv("../input/train.csv", usecols=[TEXT_COLUMN, Y_COLUMN])
unigram_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('mnb', MultinomialNB())
                        ])
test_pipeline(train_df, unigram_pipe, "Unigrams only")
class UnigramPredictions(TransformerMixin):
    def __init__(self):
        self.unigram_mnb = Pipeline([('text', CountVectorizer()), ('mnb', MultinomialNB())])

    def fit(self, x, y=None):
        # Every custom transformer requires a fit method. In this case, we want to train
        # the naive bayes model.
        self.unigram_mnb.fit(x, y)
        return self
    
    def add_unigram_predictions(self, text_series):
        # Resetting the index ensures the indexes equal the row numbers.
        # This guarantees nothing will be misaligned when we merge the dataframes further down.
        df = pd.DataFrame(text_series.reset_index(drop=True))
        # Make unigram predicted probabilities and label them with the prediction class, aka 
        # the author.
        unigram_predictions = pd.DataFrame(
            self.unigram_mnb.predict_proba(text_series),
            columns=['naive_bayes_pred_' + x for x in self.unigram_mnb.classes_]
                                           )
        # We only need 2 out of 3 columns, as the last is always one minus the 
        # sum of the other two. In some cases, that colinearity can actually be problematic.
        del unigram_predictions[unigram_predictions.columns[0]]
        df = df.merge(unigram_predictions, left_index=True, right_index=True)
        return df

    def transform(self, text_series):
        # Every custom transformer also requires a transform method. This time we just want to 
        # provide the unigram predictions.
        return self.add_unigram_predictions(text_series)
NLP = spacy.load('en', disable=['parser', 'ner'])
class PartOfSpeechFeatures(TransformerMixin):
    def __init__(self):
        self.NLP = NLP
        # Store the number of cpus available for when we do multithreading later on
        self.num_cores = cpu_count()

    def part_of_speechiness(self, pos_counts, part_of_speech):
        if eval(part_of_speech) in pos_counts:
            return pos_counts[eval(part_of_speech).numerator]
        return 0

    def add_pos_features(self, df):
        text_series = df[TEXT_COLUMN]
        """
        Parse each sentence with part of speech tags. 
        Using spaCy's pipe method gives us multi-threading 'for free'. 
        This is important as this is by far the single slowest step in the pipeline.
        If you want to test this for yourself, you can use:
            from time import time 
            start_time = time()
            (some code)
            print(f'Code took {time() - start_time} seconds')
        For faster functions the timeit module would be standard... but that's
        meant for situations where you can wait for the function to be called 1,000 times.
        """
        df['doc'] = [i for i in self.NLP.pipe(text_series.values, n_threads=self.num_cores)]
        df['pos_counts'] = df['doc'].apply(lambda x: x.count_by(attrs.POS))
        # We get a very minor speed boost here by using pandas built in string methods
        # instead of df['doc'].apply(len). String processing is generally slow in python,
        # use the pandas string methods directly where possible.
        df['sentence_length'] = df['doc'].str.len()
        # This next step generates the fraction of each sentence that is composed of a 
        # specific part of speech.
        # There's admittedly some voodoo in this step. Math can be more highly optimized in python
        # than string processing, so spaCy really stores the parts of speech as numbers. If you
        # try >>> VERB in the console you'll get 98 as the result.
        # The monkey business with eval() here allows us to generate several named columns
        # without specifying in advance that {'VERB': 98}.
        for part_of_speech in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            df[f'{part_of_speech.lower()}iness'] = df['pos_counts'].apply(
                lambda x: self.part_of_speechiness(x, part_of_speech))
            df[f'{part_of_speech.lower()}iness'] /= df['sentence_length']
        df['avg_word_length'] = (df['doc'].apply(
            lambda x: sum([len(word) for word in x])) / df['sentence_length'])
        return df

    def fit(self, x, y=None):
        # since this transformer doesn't train a model, we don't actually need to do anything here.
        return self

    def transform(self, df):
        return self.add_pos_features(df.copy())
class DropStringColumns(TransformerMixin):
    # You may have noticed something odd about this class: there's no __init__!
    # It's actually inherited from TransformerMixin, so it doesn't need to be declared again.
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype == object:
                del df[col]
        return df
logit_all_features_pipe = Pipeline([
        ('uni', UnigramPredictions()),
        ('nlp', PartOfSpeechFeatures()),
        ('clean', DropStringColumns()), 
        ('clf', LogisticRegression())
                                     ])
test_pipeline(train_df, logit_all_features_pipe)
def generate_submission_df(trained_prediction_pipeline, test_df):
    predictions = pd.DataFrame(
        trained_prediction_pipeline.predict_proba(test_df.text),
        columns=trained_prediction_pipeline.classes_
                               )
    predictions['id'] = test_df['id']
    predictions.to_csv("submission.csv", index=False)
    return predictions
