import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print('done printing available files...')
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

# create an analyzer which can be used in combination with a stemmer
def get_analyzer():
    pattern=r'\b[a-z][a-z]+\b' # at least two a-z characters
    #pattern=r'\b[^\d\W]+\b' # at least one non-numeric character
    word_count_min=2 # words are ignored if appearing in less than 2 texts
    word_freq_max=0.6 # words are ignored if appearing in more than 60% of the texts
    
    return CountVectorizer(
        token_pattern=pattern,
        stop_words=stopwords.words('english'),
        ngram_range=(1, 2), # use "phrases" of 1 and 2 words,
        min_df=word_count_min,
        max_df=word_freq_max,
    ).build_analyzer()

def stem_words(doc):
    analyzer = get_analyzer()
    stemmer = SnowballStemmer('english');
    return (stemmer.stem(w) for w in analyzer(doc))

vectorizer_text = CountVectorizer(analyzer = stem_words)
vectorizer_keywords = OneHotEncoder()

print('vectorizers created')
import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

print('we would like to add a few additional features to the original dataset...')

hashtag_pattern = re.compile(r'(^|\s+)(#)\w+') # matches #something
user_mention_pattern = re.compile(r'(^|\s+)(@)\w+') # matches @username123
sentiment_analyzer = SentimentIntensityAnalyzer()

def add_features(dataframe):
    dataframe['has_keyword'] = 1 - dataframe['keyword'].isnull() * 1
    dataframe['has_location'] = 1 - dataframe['location'].isnull() * 1
    dataframe['has_hashtag'] = dataframe['text'].str.contains(hashtag_pattern) * 1
    dataframe['has_mention'] = dataframe['text'].str.contains(user_mention_pattern) * 1
    
    polarity = dataframe['text'].apply(lambda text: sentiment_analyzer.polarity_scores(text))
    dataframe['sentiment_negative'] = polarity.apply(lambda p: p['neg'])
    dataframe['sentiment_neutral'] = polarity.apply(lambda p: p['neu'])
    dataframe['sentiment_positive'] = polarity.apply(lambda p: p['pos'])
    dataframe['sentiment_total'] = polarity.apply(lambda p: p['compound'])
    
    dataframe['key'] = dataframe['keyword'].fillna(value='EMPTY')
    
    return dataframe

train_df = add_features(pd.read_csv("/kaggle/input/nlp-getting-started/train.csv"))
test_df = add_features(pd.read_csv("/kaggle/input/nlp-getting-started/test.csv"))

print('a few additional features were added, current columns:')
for c in train_df.columns: print('- ' + str(c))

train_df[train_df['has_mention'] == 1].head()
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split

print('splitting into training and validation sets...')
COL_LABEL = 'target'
X = train_df.drop(COL_LABEL, axis=1)
y = train_df[COL_LABEL]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=0)
X_test = test_df # has no target column :)

print('transforming text to a matrix representing the words...')
COL_TEXT = 'text'
X_train_vectors = vectorizer_text.fit_transform(X_train[COL_TEXT])
X_valid_vectors = vectorizer_text.transform(X_valid[COL_TEXT])
X_test_vectors = vectorizer_text.transform(X_test[COL_TEXT])

print('transforming keywords to a matrix...')
COL_KEYWORD = 'key'
X_train_key_vec = csr_matrix(vectorizer_keywords.fit_transform(X_train[COL_KEYWORD].to_numpy().reshape(-1, 1)))
X_valid_key_vec = csr_matrix(vectorizer_keywords.transform(X_valid[COL_KEYWORD].to_numpy().reshape(-1, 1)))
X_test_key_vec = csr_matrix(vectorizer_keywords.transform(X_test[COL_KEYWORD].to_numpy().reshape(-1, 1)))

print('size of vectorized text: ' + str(X_train_vectors.shape))
print('size of vectorized keywords: ' + str(X_train_key_vec.shape))

# merge the vectors
X_train_vectors = hstack((X_train_vectors, X_train_key_vec))
X_valid_vectors = hstack((X_valid_vectors, X_valid_key_vec))
X_test_vectors = hstack((X_test_vectors, X_test_key_vec))

##print(vectorizer_text.get_feature_names())
print('done reading data into training, validation and test vectors')
print('training set size is ' + str(X_train_vectors.shape))
import numpy as np
from scipy.sparse import csr_matrix, hstack

def to_csr_column(dataframe, col):
    # col can be string or list of strings
    # if it is a single string, we need to transpose the output matrix
    matrix = csr_matrix(dataframe[col].values)
    return matrix.transpose() if col is str else matrix

print('lets add some of the new features to the vector')
ADD_FEATURES = ['has_keyword', 
                'has_location', 
                'has_hashtag', 
                'has_mention', 
                'sentiment_negative', 
                'sentiment_neutral', 
                'sentiment_positive', 
                'sentiment_total']

train_columns = to_csr_column(X_train, ADD_FEATURES)
valid_columns = to_csr_column(X_valid, ADD_FEATURES)
test_columns = to_csr_column(X_test, ADD_FEATURES)

X_train_vectors = hstack((X_train_vectors, train_columns))
X_valid_vectors = hstack((X_valid_vectors, valid_columns))
X_test_vectors = hstack((X_test_vectors, test_columns))

print('the new training set shape is: ' + str(X_train_vectors.shape))
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier

model = LogisticRegression(max_iter=200) # log-reg seems to do slightly better
#model = MultinomialNB()
#model = RidgeClassifier()
model.fit(X_train_vectors, y_train)

print('creating a prediction on the validation set using model: ' + str(model))
y_valid_pred = model.predict(X_valid_vectors)

print('accuracy score: ' + str(metrics.accuracy_score(y_valid, y_valid_pred)))
print('f1 score: ' + str(metrics.f1_score(y_valid, y_valid_pred)))

y_valid_np = y_valid.to_numpy()
X_valid_np = X_valid.to_numpy()

DISPLAY_COUNT = 50
COL_PREDICTION = 'prediction'
print('lets have a look at ' + str(DISPLAY_COUNT) + ' of the texts that are not classified correctly...')
pd.set_option('display.max_colwidth', -1) # no max for column width - display entire text

df_pred = X_valid.copy()
df_pred[COL_LABEL] = y_valid
df_pred[COL_PREDICTION] = y_valid_pred
df_pred.loc[df_pred[COL_LABEL] != df_pred[COL_PREDICTION]].head(DISPLAY_COUNT) # show incorrectly classified examples

df_pred.loc[(1 == df_pred[COL_LABEL]) & (df_pred[COL_LABEL] == df_pred[COL_PREDICTION])].head(DISPLAY_COUNT) # show correctly classified disasters

df_pred.loc[(0 == df_pred[COL_LABEL]) & (df_pred[COL_LABEL] == df_pred[COL_PREDICTION])].head(DISPLAY_COUNT) # show correctly classified non-disasters
import pandas as pd

print('creating submission file...')
y_test_pred = model.predict(X_test_vectors)

# insert prediction into sample submission file
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
submission["target"] = y_test_pred
submission.to_csv("submission.csv", index=False)

print('submission created')
print('lets print the words, sorted from most to least frequent...')

sum_words = X_train_vectors.sum(axis=0)
words_freq = [(word, sum_words[0, i]) for word, i in vectorizer_text.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

print(words_freq)