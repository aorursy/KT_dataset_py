import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
def load_data(path):
    data = pd.read_csv(path, sep='\t', quoting=3, index_col=0)
    return data
train = load_data('../input/labeledTrainData.tsv')
test = load_data('../input/testData.tsv')
unlabeled = load_data('../input/unlabeledTrainData.tsv')
def print_sample(sample):
    print('Id: {}'.format(sample.name))
    print('Sentiment: {}'.format(sample['sentiment']))
    print('Text:')
    print(sample['review'])
def get_a_sample_review():
    obs = train.sample().iloc[0]
    print_sample(obs)
get_a_sample_review()
train.review.str.len().hist(bins=100);
sample = train.loc[train.review.str.contains('bad')].sample().iloc[0]
print_sample(sample)
def check_keywords(keywords):
    result = pd.DataFrame(columns=['not_present', 'positive'])
    for word in keywords:
        df = train.loc[train.review.str.contains(word, case=False)]
        result.loc[word] = [train.shape[0] - df.shape[0], df.sentiment.sum()]
    result = result / train.shape[0]
    result['negative'] = 1- result['not_present'] - result['positive']
    result.plot(kind='bar', stacked=True)
    plt.legend(loc='best')
    return result
keywords = ['good', 'bad', 'great', 'disaster', 'fun', 'phenomenal']
check_keywords(keywords)
train['clean_review'] = train.review
%%time
train['clean_review'] = train.clean_review.str.replace('<.+? />','')
train.review.iloc[0]
train.clean_review.iloc[0]
train['clean_review'] = train.clean_review.str.lower()
train.clean_review.iloc[0]
# Everything not a alphabet character replaced with a space
train['clean_review'] = train.clean_review.str.replace('[^a-zA-Z]', ' ')
# Remove double space
train['clean_review'] = train.clean_review.str.replace(' +', ' ')
# Remove trailing space at the beginning or end
train['clean_review'] = train.clean_review.str.strip()
train.clean_review.iloc[0]
import nltk
stemmer = nltk.stem.SnowballStemmer('english')
stemmer.stem('people')
stemmer.stem('guys')
stemmer.stem('closed')
def review_cleaning(reviewSeries):
    result = reviewSeries.copy()
    # Remove HTML tags

    # Convert to lower case
    
    # Remove non alphabetic characters
    
    # Remove double space and strip spaces
    
    return result
train['clean_review'] = review_cleaning(train.review)
test['clean_review'] = review_cleaning(test.review)
unlabeled['clean_review'] = review_cleaning(unlabeled.review)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
test_corpus = [ 
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',     
    'Is this the first document?',]
count_vectorizer = CountVectorizer(ngram_range=(1,1), analyzer='word')
count_vectorizer.fit(test_corpus)
count_vectorizer.vocabulary_
count_vectorizer.transform(test_corpus)
count_vectorizer.transform(test_corpus).todense()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), analyzer='word')
tfidf_vectorizer.fit(test_corpus)
tfidf_vectorizer.vocabulary_
tfidf_vectorizer.transform(test_corpus)
tfidf_vectorizer.transform(test_corpus).todense()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train.clean_review, train.sentiment, test_size=0.2, 
                                                          stratify=train.sentiment)
from sklearn.linear_model import LogisticRegression
vectorizer = TfidfVectorizer(analyzer='word', # word or char
                             ngram_range=(1,1), # ngram setting  
                             binary=True, # set tf to binary instead of count
                             max_df=0.9, # remove too frequent words
                             min_df=10, # remove too rare words
                             max_features = None, # max words in vocabulary, will keep most frequent words
                             ) 
X_vect_train = vectorizer.fit_transform(X_train)
lreg = LogisticRegression()
lreg.fit(X_vect_train, y_train)
from sklearn.metrics import accuracy_score, roc_auc_score
p_valid = lreg.predict_proba(
            vectorizer.transform(X_valid)
            )
accuracy_score(y_valid, p_valid.argmax(axis=1))
def test_prediction(p_test):
    target = test.index.str.slice(-2,-1).isin(['7','8','9','0']).astype(np.int8)
    print('accuracy: {}'.format(accuracy_score(target, p_test[:,1]>=0.5)))
    print('roc auc: {}'.format(roc_auc_score(target, p_test[:,1])))
p_test = lreg.predict_proba(
            vectorizer.transform(test.clean_review)
            )
test_prediction(p_test)
# vectorizer.vocabulary_ is a dictionary of word -> column index
vectorizer.vocabulary_['good']
# Let's store them in a DataFrame
coefs = pd.DataFrame(columns=['word'])
for word, ind in vectorizer.vocabulary_.items():
    coefs.loc[ind, 'word'] = word
coefs.sort_index(inplace=True)
coefs.head()
# the coefficient are stored in a 1xn array
lreg.coef_
# Once sorted, the word order correspond to coefficient order
coefs['coefs'] = lreg.coef_[0,:]
most_relevant_words = coefs.iloc[np.argsort(coefs.coefs.abs())].tail(20)
most_relevant_words.sort_values('coefs', inplace=True)
def plot_impact(words, impacts):
    pos_ind = (impacts > 0)
    position = np.arange(len(words))
    plt.barh(bottom=position[pos_ind], width=impacts[pos_ind], color='green')
    plt.barh(bottom=position[~pos_ind], width=impacts[~pos_ind], color='red')
    plt.yticks(position + 0.4 ,words)
    plt.show()
plot_impact(most_relevant_words.word.values, most_relevant_words.coefs.values)
from sklearn.pipeline import Pipeline
vectorizer = TfidfVectorizer(analyzer='word', # word or char
                             ngram_range=(1,1), # ngram setting  
                             binary=True, # set tf to binary instead of count
                             max_df=0.9, # remove too frequent words
                             min_df=10, # remove too rare words
                             max_features = None, # max words in vocabulary, will keep most frequent words
                             ) 
lreg = LogisticRegression()
pipeline = Pipeline([
        ('vectorizer', vectorizer), 
        ('lreg', lreg)
    ])
pipeline.fit(X_train, y_train)
! conda install -yq seaborn
import seaborn as sns
def words_impacts(text):
    baseline_ = pipeline.predict_proba([text])[:,1]
    words_list = text.split()
    text_excl_words = list()
    for i, word in enumerate(words_list):
        new_words_list = text.split()
        new_words_list.pop(i)
        text_excl_words.append(' '.join(new_words_list)) 
    impacts = baseline_ - pipeline.predict_proba(text_excl_words)[:, 1]
    return words_list, impacts, baseline_
def reshape_pad(iterable, shape, pad_value=0):
    n_ = len(iterable)
    pad_length = shape[0] * shape[1] - n_
    data = list(iterable)
    data.extend([pad_value for _ in range(pad_length)])
    assert len(data) == shape[0] * shape[1]
    data = np.reshape(data, shape)
    return data

def plot_text_with_impacts(words_list, impacts, n_cols=10):
    assert len(words_list) == len(impacts)
    n_rows = (len(words_list) // n_cols) + 1
    words = reshape_pad(words_list, (n_rows, n_cols), pad_value='')
    impact_data = reshape_pad(impacts, (n_rows, n_cols))
    plt.figure(figsize=(20, n_rows//2))
    sns.heatmap(impact_data, annot=words, square=False, fmt='')
    sns.set(font_scale=1)
review_sample = X_valid.sample()
words_list, impacts, baseline = words_impacts(review_sample.iloc[0])
print('Ground_truth: {}'.format(y_valid.loc[review_sample.index].iloc[0]))
print('Predicted probability: {}'.format(baseline[0]))
plot_text_with_impacts(words_list, impacts, n_cols=15)
!conda install -yq -c amueller wordcloud
from wordcloud import WordCloud, wordcloud, ImageColorGenerator
def wcloud_color(word, font_size, position, orientation, random_state=None, **kwargs):
        i = words_list.index(word)
        if impacts[i] > 0:
            return 'Tomato'
        else:
            return 'DodgerBlue'
wcloud = WordCloud(max_words=100, 
                   background_color='white', 
                   color_func=wcloud_color, 
                   relative_scaling=1)
img = wcloud.generate_from_frequencies(zip(words_list, np.abs(impacts)))
plt.imshow(img);
from sklearn.pipeline import Pipeline
vectorizer = TfidfVectorizer(analyzer='word', # word or char
                             ngram_range=(1,1), # ngram setting  
                             binary=True, # set tf to binary instead of count
                             max_df=0.9, # remove too frequent words
                             min_df=10, # remove too rare words
                             max_features = None, # max words in vocabulary, will keep most frequent words
                             ) 
lreg = LogisticRegression()
pipeline = Pipeline([
        ('vectorizer', vectorizer), 
        ('lreg', lreg)
    ])
pipeline.get_params()['vectorizer__analyzer']
from sklearn.model_selection import RandomizedSearchCV
## answer
params_grid = {
    'lreg__C': [10, 1, 1e-1, 1e-2, 1e-3],
    'vectorizer__ngram_range': [(1,1), (1,2), (1,3)],
    'vectorizer__stop_words':['english', None],
    'vectorizer__min_df': [5, 10, 20, 50, 100],
    'vectorizer__max_df': [0.6, 0.7, 0.8, 0.9, 1.0],
    'vectorizer__binary': [True, False],
    'vectorizer__use_idf': [True, False]
}
search_ = RandomizedSearchCV(pipeline, params_grid, 
                             n_iter=4, n_jobs=4, verbose=1, cv=5)
search_.fit(X_train, y_train)
search_.best_score_
search_.best_params_
