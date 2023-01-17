import os

from time import time



import pandas as pd

import numpy as np



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report



from sklearn.base import BaseEstimator, TransformerMixin



import warnings

warnings.simplefilter(action='ignore')



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
# config

DATA_DIR = '../input/tweet-sentiment-extraction'

TRAIN_DATA_FILE = 'train.csv'

TEST_DATA_FILE = 'test.csv'

SUBMISSION_FILE = 'submission.csv'



RANDOM_STATE = 0
train_data = pd.read_csv(os.path.join(DATA_DIR, TRAIN_DATA_FILE)).fillna('')

test_data = pd.read_csv(os.path.join(DATA_DIR, TEST_DATA_FILE)).fillna('')
train_data = train_data[['textID', 'text', 'sentiment', 'selected_text']]

train_data[17:22]
test_data.head()
def create_model():

    clf = Pipeline([

        ('vect', CountVectorizer(tokenizer=lambda x: x.split())

        ),

        ('clf', LogisticRegression(

                                   max_iter=1000,

                                   random_state=RANDOM_STATE

                                  )

        )

    ])

    return clf



X = pd.concat([train_data['text'], test_data['text']], axis=0)

Y = pd.concat([train_data['sentiment'], test_data['sentiment']], axis=0)

Y = Y.map({'negative': -1, 'positive': 1, 'neutral': 0})

print(X.shape, Y.shape)



model = create_model()



model.fit(X, Y)



pred = model.predict(X)

print(classification_report(y_true=Y, y_pred=pred))
# Grid search with CV doesn't seem to make sense here because the model

# is not used to make predictions on the test data

# parameters = {

#     'clf__C': [0.001, 0.009, 0.01, 0.09, 1, 5, 10, 25],

#     'clf__class_weight': [None, 'balanced']

# }



# gs_clf = GridSearchCV(model,

#                       parameters,

#                       cv=5,

#                       verbose=2,

#                       n_jobs=-1

#                      )



# gs_clf = gs_clf.fit(X, Y)



# gs_clf.best_params_
VOCAB = model.named_steps['vect'].vocabulary_

COEF = model.named_steps['clf'].coef_



class WordSelector(BaseEstimator, TransformerMixin):

    """Provide a class to select words supporing the sentiment."""

    

    def __init__(self, pos_class_std=2.2, neg_class_std=2.2):

        # number of standard deviations to select words 

        # with unusual weights from a tweet

        self.pos_class_std = pos_class_std

        self.neg_class_std = neg_class_std

        

        self.vocabulary_ = VOCAB  # word vocab from CountVectorizer

        self.coef_ = COEF         # word importance weight from LogRegression

        

        self.weights_by_classes = {

            'negative': list(enumerate(self.coef_[0])),

            'neutral':  list(enumerate(self.coef_[1])),

            'positive': list(enumerate(self.coef_[2]))

        }

        

        # Translation dict from vocab indexes to words

        # Only used for plotting

        self.index_to_word = {

            ind: word

            for (word, ind)

            in self.vocabulary_.items()

            }

    

    ##### PLOTTING-RELATED METHODS###############################

    def get_words_from_idx(self, indexes):

        """Return a list of words from indexes for plotting."""

        return [self.index_to_word[index] for index in indexes]



    

    def plot_top_features(self, class_label, max_top_feat):

        """Plot a bar chart of top features for a given label."""

        idx_coef_list = sorted(self.weights_by_classes[class_label],

                               key=lambda pair: pair[1], 

                               reverse=True

                              )

        idx, coef = zip(*idx_coef_list)

        top_words = self.get_words_from_idx(idx[:max_top_feat])

        plt.figure(figsize=(12,4))

        plt.bar(top_words, coef[:max_top_feat])

        plt.title(f'Top-{max_top_feat} features in category {class_label}')

        plt.xlabel('Features')

        plt.ylabel('Weights')

        plt.xticks(rotation = '45')

        plt.show()

    ############################################################

    

    

    def get_weights(self, text_list, class_weights):

        """Return a list of weights for text."""

        text_idx = [self.vocabulary_[tok] for tok in text_list]



        return [class_weights[idx][1] for idx in text_idx]



    def get_top_words(self, words_list, weights_list, num_std):

        """Return a string of words with unusually high weights."""

        mean = np.mean(weights_list)

        std = np.std(weights_list)

        top_words = []

        for word, weight in zip(words_list, weights_list):

            if weight > (mean +  num_std * std):

                top_words.append(word)

        return ' '.join(top_words)





    def select_words(self, text_sentiment):

        """Select words given sentiment label by calling get_top_words()."""

        text, sentiment = text_sentiment



        if sentiment == 'neutral':

            return text



        text = text.lower().split()

        weights = self.get_weights(text,

                              self.weights_by_classes[sentiment]  

                             )



        if sentiment == 'positive':

            res = self.get_top_words(text, weights, num_std=self.pos_class_std)



        if sentiment == 'negative':

            res = self.get_top_words(text, weights, num_std=self.neg_class_std)



        if res == '':

            res = ' '.join(text)



        return res

    

    def fit(self, X, y=None):

        return self

    

    def predict(self, X):

        """Return predicted 'selected_text' as a Pandas series.

        

        X: Pandas dataframe with columns 'text', sentiment'

        """



        res = pd.DataFrame()

        res['sentiment'] = X['sentiment']

        res['text'] = X['text'].map(lambda s: s.lower())

        res['selected_text'] = X[['text', 'sentiment']].apply(self.select_words, axis=1)



        return res['selected_text']

    

    def jaccard(self, predicted_selected):

        """Provide an evaluation metric.

        

        predicted_selected: a tuple (predicted text, true selected text)

        """

        str1, str2 = predicted_selected

        a = set(str1.lower().split()) 

        b = set(str2.lower().split())

        if (len(a) == 0) & (len(b) == 0):

            return 0.5

        c = a.intersection(b)

        return float(len(c)) / (len(a) + len(b) - len(c))



    def score(self, X, y):

        """Return a mean Jaccard score.

        

        X: Pandas dataframe with columns 'text', sentiment'

        y: Pandas series containing true 'selected_text'

        """

        

        res = pd.DataFrame()

        

        res['selected_text'] = y

        res['sentiment'] = X['sentiment']

        res['text'] = X['text']



        res['predictions'] = self.predict(res[['text', 'sentiment']])

        

        res['score'] = res[['predictions', 'selected_text']].apply(self.jaccard, axis=1)



        return res['score'].mean()
word_selector = WordSelector(pos_class_std=1.8, neg_class_std=2.4)



print('Jaccard score for training data:')

word_selector.score(train_data[['text', 'sentiment']], train_data['selected_text'])



# [0. , 0.8, 1.6, 2.4, 3.2, 4. ]

# pos_class_std=1.6, neg_class_std=2.4

# 0.6535060973409171



# [1. , 1.4, 1.8, 2.2, 2.6, 3. ]

# {'neg_class_std': 2.2, 'pos_class_std': 1.8}

# 0.6552466246349189



#     'pos_class_std': [1.6, 1.7, 1.8, 1.9],  

#     'neg_class_std': [2.1, 2.2, 2.3, 2.4]

# {'neg_class_std': 2.4, 'pos_class_std': 1.8}

# 0.6553365154317554
# Jaccard score by sentiment

temp_df = pd.DataFrame()

temp_df['sentiment'] = train_data['sentiment']

temp_df['text'] = train_data['text']  #.map(lambda s: s.lower())



temp_df['predicted_text'] = word_selector.predict(train_data[['text', 'sentiment']])



temp_df['selected_text'] = train_data['selected_text']



temp_df['score'] = temp_df[['predicted_text', 'selected_text']].apply(word_selector.jaccard, axis=1)



print('Jaccard score by sentiment:')

print(temp_df.groupby('sentiment')['score'].mean())

print('\nTotal score:', temp_df['score'].mean())
for sentiment in ['negative', 'neutral', 'positive']:

    word_selector.plot_top_features(sentiment, 20)
# print('Performing GridSearch for WordSelect parameters...')

# parameters = {

#     'pos_class_std': [1.6, 1.7, 1.8, 1.9],  

#     'neg_class_std': [2.1, 2.2, 2.3, 2.4]

# }



# gs = GridSearchCV(WordSelector(

#                               ),

#                   parameters,

#                   cv=5,

#                   verbose=1,

#                   n_jobs=-1

#                  )





# gs = gs.fit(train_data[['text', 'sentiment']], train_data['selected_text'])



# gs.best_params_
submission_df = pd.DataFrame() 

submission_df['textID'] = test_data['textID']



submission_df['selected_text']= word_selector.predict(test_data[['text', 'sentiment']])

submission_df.to_csv(SUBMISSION_FILE, index = False)
pd.set_option('max_colwidth', 80)

test_data.head()
submission_df.head(5)
plt.hist(train_data['selected_text'].map(len), alpha=0.8, bins=40)

plt.title('Distribution of Lengths of TRUE Substrings in Training Data')

plt.xlabel('Char length')

plt.ylabel('How often')

plt.show()



plt.hist(temp_df['predicted_text'].map(len), alpha=0.8, bins=40)

plt.title('Distribution of Lengths of PREDICTED Substrings in Training Data')

plt.xlabel('Char length')

plt.ylabel('How often')

plt.show()



plt.hist(submission_df['selected_text'].map(len), alpha=0.8, bins=40)

plt.title('Distribution of Lengths of PREDICTED Substrings in Testing Data')

plt.xlabel('Char length')

plt.ylabel('How often')

plt.show()